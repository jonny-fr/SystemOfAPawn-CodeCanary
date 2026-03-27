"""
classifier/scoring.py
=====================
Täglicher Mood-Score auf Basis des Pipeline-Outputs.

Score-Bereich: -100 … +100
  Negativ  →  depressive Richtung
  0        →  Baseline / stabil
  Positiv  →  manische Richtung

Formel (kontinuierlich-additiv)
--------------------------------
Jeder Score setzt sich aus zwei Teilen zusammen:

  raw   = tanh(net / NET_SCALE) * RAW_MAX
          net = M_t − D_t   (kontinuierliches Richtungssignal)
          Misst die Nettoabweichung von der persönlichen Baseline.

  boost = ± tanh(dominant / BOOST_SCALE) * BOOST_MAX * conf_weight
          dominant = D_t bei Depressions-State, M_t bei Manie-State
          Verstärkt den Score nur dann, wenn der Algorithmus einen
          bestätigten State-Change detektiert hat (CUSUM-Alarm).
          conf_weight wächst mit CUSUM-Konfidenz → sanfte Rampe, kein Sprung.

  score = clip(raw + boost, -100, +100) * quality_damping

Warum keine festen Basen (DEP_BASE / MAN_BASE) mehr?
  Die alten Konstanten (BASE=55) erzeugten einen harten Sprung beim
  Schwellenwert-Crossing: von max ±12 (stabil) auf min ±55 (confirmed).
  Die neue Formel hat diesen Sprung nicht mehr – an der Schwelle ist
  boost ≈ 0 und raw setzt sich stetig fort.

Kalibrierungsbeispiele (D_t / M_t in Einheiten der persönlichen Baseline):
  Stabil, net=−2:                   raw ≈ −12,   boost = 0     → −12
  Depression-Onset, net=−5, D_t=3:  raw ≈ −27,   boost ≈ −10  → −37
  Depression confirmed, D_t=10:     raw ≈ −43,   boost ≈ −30  → −73
  Depression stark, D_t=20:         raw ≈ −49,   boost ≈ −47  → −96
  Manie confirmed, M_t=8:           raw ≈ +31,   boost ≈ +24  → +55
  Manie stark, M_t=18:              raw ≈ +47,   boost ≈ +43  → +90
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Kalibrierungskonstanten
# ---------------------------------------------------------------------------

# Raw signal: tanh(net / NET_SCALE) * RAW_MAX
# NET_SCALE ≈ Signalstärke (D_t - M_t) bei der tanh 0.76 erreicht
_NET_SCALE: float = 10.0
_RAW_MAX:   float = 55.0

# State boost: tanh(dominant / BOOST_SCALE) * BOOST_MAX * conf_weight
# Nur aktiv bei bestätigtem State-Change (dep / man shift)
_BOOST_SCALE: float = 8.0
_BOOST_MAX:   float = 55.0

# Mindest-Gewichtung des Boosts bei Konfidenz = 0  (verhindert Sprung an der Schwelle)
_CONF_MIN: float = 0.3

# Halb-Boost bei "changed but unclear" (Richtung unklar, aber Veränderung vorhanden)
_UNCLEAR_BOOST_FRACTION: float = 0.45

# Dämpfung für schlechte Audioqualität  [0, 1]
# Wird auf den Gesamtscore angewendet, wenn quality_reliability < 1.0
# Achtung: reliability ist bei der v2-Pipeline bereits in D_t/M_t eingerechnet,
# daher nur leichte Zusatz-Dämpfung für explizit "degraded" Qualität.
_QUALITY_FULL_WEIGHT:     float = 1.0
_QUALITY_DEGRADED_WEIGHT: float = 0.85


# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------

@dataclass
class DailyScore:
    """Tages-Score-Ergebnis.

    Attributes
    ----------
    day : int
        Tagnummer (1-basiert).
    score : float or None
        Mood-Score in [−100, +100] oder None bei Abstain.
    label : str
        Deutsches Kurz-Label.
    confidence : float
        CUSUM-Konfidenz [0, 1].
    is_baseline : bool
        True für Tage 1–3.
    abstain_reason : str or None
        Grund für Abstain, sonst None.
    """
    day: int
    score: Optional[float]
    label: str
    confidence: float
    is_baseline: bool
    abstain_reason: Optional[str]

    def to_dict(self) -> dict:
        return {
            "day": self.day,
            "score": self.score,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "is_baseline": self.is_baseline,
            "abstain_reason": self.abstain_reason,
        }


# ---------------------------------------------------------------------------
# Kern-Scoring-Funktion
# ---------------------------------------------------------------------------

def compute_daily_score(
    *,
    day: int,
    final_status: str,
    quality_status: str,
    D_t: float,
    M_t: float,
    A_t: float,
    C_D_t: float = 0.0,
    C_M_t: float = 0.0,
    quality_reliability: float = 1.0,
    confidence: float = 0.0,
) -> DailyScore:
    """Berechnet den täglichen Mood-Score.

    Parameters
    ----------
    day : int
        Tagnummer (1-basiert). Tage 1–3 → Score = 0 (Baseline).
    final_status : str
        Klassifikation aus dem Pipeline-Algorithmus.
    quality_status : str
        Audioqualität: "clean", "degraded", "reject".
    D_t : float
        Geglättete (3-Tage-Median) gewichtete Depressions-Projektion.
        Bereits mit quality_reliability multipliziert.
    M_t : float
        Geglättete gewichtete Manie-Projektion.
        Bereits mit quality_reliability multipliziert.
    A_t : float
        L2-Norm der Z-Score-Abweichungen (Gesamtmagnitude der Änderung).
    C_D_t : float
        CUSUM-Depressions-Akkumulator (informativ).
    C_M_t : float
        CUSUM-Manie-Akkumulator (informativ).
    quality_reliability : float
        Zuverlässigkeit [0, 1] der Audioqualität.
    confidence : float
        CUSUM-Konfidenz [0, 1]: wie weit liegt der Score über der Schwelle.
        0 = gerade an der Schwelle, 1 = weit darüber.
    """
    # ── Abstain-Fälle ────────────────────────────────────────────────────
    if final_status == "abstain_insufficient_baseline":
        return DailyScore(
            day=day, score=None, label="Keine Baseline",
            confidence=0.0, is_baseline=False,
            abstain_reason="insufficient_baseline",
        )

    if final_status == "abstain_due_to_quality":
        return DailyScore(
            day=day, score=None, label="Unbrauchbare Audioqualität",
            confidence=0.0, is_baseline=False,
            abstain_reason="quality_rejected",
        )

    # ── Baseline-Periode (Tage 1–3) ──────────────────────────────────────
    if day < 4:
        return DailyScore(
            day=day, score=0.0, label="Baseline",
            confidence=float(quality_reliability),
            is_baseline=True, abstain_reason=None,
        )

    # ── NaN-Schutz ───────────────────────────────────────────────────────
    if not all(math.isfinite(v) for v in [D_t, M_t, A_t]):
        return DailyScore(
            day=day, score=None, label="Fehlende Daten",
            confidence=0.0, is_baseline=False,
            abstain_reason="nan_inputs",
        )

    # ── Kontinuierlich-additive Scoring-Formel ───────────────────────────

    # 1) Netto-Richtungssignal (negativ = depressiv, positiv = manisch)
    net = M_t - D_t

    # 2) Raw score: kontinuierliches Signal, begrenzt auf ±RAW_MAX
    raw = math.tanh(net / _NET_SCALE) * _RAW_MAX

    # 3) Konfidenz-Gewichtung für den Boost
    #    Rampe: _CONF_MIN (an der Schwelle) bis 1.0 (bei hoher Konfidenz)
    conf_clamped = max(0.0, min(1.0, float(confidence)))
    conf_weight = _CONF_MIN + (1.0 - _CONF_MIN) * conf_clamped

    # 4) State-Boost: verstärkt in der bestätigten Richtung
    if final_status == "depression-like shift":
        # Negativer Boost, Stärke aus D_t
        boost = -math.tanh(max(0.0, D_t) / _BOOST_SCALE) * _BOOST_MAX * conf_weight

    elif final_status == "mania-like shift":
        # Positiver Boost, Stärke aus M_t
        boost = +math.tanh(max(0.0, M_t) / _BOOST_SCALE) * _BOOST_MAX * conf_weight

    elif final_status == "changed but unclear":
        # Halber Boost in der Nettorichtung (Veränderung vorhanden, Richtung unklar)
        boost = (
            math.tanh(net / (_NET_SCALE * 1.4))
            * _BOOST_MAX * _UNCLEAR_BOOST_FRACTION
            * conf_weight
        )

    else:
        # "stable" / "normal": kein Boost
        boost = 0.0

    # 5) Qualitätsdämpfung (additiv zu der bereits in D_t/M_t eingerechneten)
    if quality_status == "degraded":
        quality_damping = _QUALITY_DEGRADED_WEIGHT
    else:
        quality_damping = _QUALITY_FULL_WEIGHT

    # 6) Zusammensetzen und begrenzen
    score = (raw + boost) * quality_damping
    score = round(max(-100.0, min(100.0, score)), 1)

    return DailyScore(
        day=day,
        score=score,
        label=_label(score),
        confidence=conf_clamped,
        is_baseline=False,
        abstain_reason=None,
    )


# ---------------------------------------------------------------------------
# Convenience-Wrapper für Pipeline-Ausgabe
# ---------------------------------------------------------------------------

def score_from_pipeline_row(row: dict) -> DailyScore:
    """Score aus einem Row-Dict der Pipeline-Ausgabe berechnen."""
    return compute_daily_score(
        day=int(row.get("day", 0)),
        final_status=str(row.get("final_status", "stable")),
        quality_status=str(row.get("quality_status", "clean")),
        D_t=float(row.get("D_t", 0.0)),
        M_t=float(row.get("M_t", 0.0)),
        A_t=float(row.get("A_t", 0.0)),
        C_D_t=float(row.get("C_D_t", 0.0)),
        C_M_t=float(row.get("C_M_t", 0.0)),
        quality_reliability=float(row.get("quality_reliability", 1.0)),
        confidence=float(row.get("confidence", 0.0)),
    )


def score_dataframe(df: "pd.DataFrame") -> "pd.Series":
    """Score für jede Zeile eines pandas DataFrames."""
    import pandas as pd  # noqa: PLC0415

    def _row_score(row: "pd.Series") -> float:
        result = score_from_pipeline_row(row.to_dict())
        return result.score if result.score is not None else float("nan")

    return df.apply(_row_score, axis=1)


def score_series_for_patient(rows: List[dict]) -> List[DailyScore]:
    """Tages-Scores für einen einzelnen Patienten (nach Tag sortiert)."""
    return [
        score_from_pipeline_row(row)
        for row in sorted(rows, key=lambda r: int(r.get("day", 0)))
    ]


# ---------------------------------------------------------------------------
# Label-Hilfsfunktion
# ---------------------------------------------------------------------------

_SCORE_BANDS: List[tuple] = [
    ( 60.0, "Stark manisch"),
    ( 25.0, "Manisch"),
    (  8.0, "Leicht manisch"),
    ( -7.9, "Normal / Stabil"),
    (-25.0, "Leicht depressiv"),
    (-60.0, "Depressiv"),
]


def _label(score: float) -> str:
    for threshold, text in _SCORE_BANDS:
        if score >= threshold:
            return text
    return "Stark depressiv"


def interpret_score(score: Optional[float]) -> str:
    """Öffentliche Interpretation eines Score-Werts."""
    if score is None or not math.isfinite(score):
        return "Keine Bewertung (unzureichende Daten)"
    return _label(score)


# ---------------------------------------------------------------------------
# CLI-Demo: python scoring.py <path/to/day_level_scores.json>
# ---------------------------------------------------------------------------

def _cli_demo(json_path: str) -> None:
    data: List[dict] = json.loads(Path(json_path).read_text(encoding="utf-8"))
    patients: Dict[str, List[dict]] = {}
    for row in data:
        pid = str(row.get("patient_id", "unknown"))
        patients.setdefault(pid, []).append(row)

    for patient_id, rows in sorted(patients.items()):
        scores = score_series_for_patient(rows)
        print(f"\n{'=' * 62}")
        print(f"  Patient: {patient_id}")
        print(f"{'=' * 62}")
        print(f"  {'Tag':>4}  {'Score':>7}  {'Konfidenz':>9}  Label")
        print(f"  {'-'*4}  {'-'*7}  {'-'*9}  {'-'*25}")
        for ds in scores:
            if ds.score is None:
                score_str = "   None"
                label_str = f"[{ds.abstain_reason}]"
            else:
                score_str = f"{ds.score:+7.1f}"
                label_str = ds.label
            marker = " (Baseline)" if ds.is_baseline else ""
            print(f"  {ds.day:>4}  {score_str}  {ds.confidence:>9.3f}  {label_str}{marker}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Verwendung: python scoring.py <path/to/day_level_scores.json>")
        sys.exit(1)
    _cli_demo(sys.argv[1])
