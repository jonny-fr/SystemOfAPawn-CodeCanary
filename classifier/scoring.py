"""
classifier/scoring.py
=====================
Unified daily mood score based on the pipeline's final_status and signal values.

Score range
-----------
  -100 … -1  →  depressive Richtung
      0       →  Baseline / stabil
  +1 … +100   →  manische Richtung

Wie der Score berechnet wird
-----------------------------
Der `final_status` aus dem Algorithmus ist der primäre Treiber:

  "depression-like shift"  →  Score -75 bis -100  (Stärke aus D_t)
  "mania-like shift"       →  Score +60 bis +90   (Stärke aus M_t)
  "changed but unclear"    →  Score ±5 bis ±50    (Richtung aus M_t − D_t)
  "stable"                 →  Score ±0 bis ±12    (sehr nah an 0)
  Baseline (Tage 1–3)      →  Score = 0
  Abstain                  →  None (keine Bewertung möglich)
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Kalibrierungskonstanten
# ---------------------------------------------------------------------------

# Depression: Score = -(DEP_BASE + tanh(D_t / DEP_SCALE) * DEP_RANGE)
# Kalibriert auf: D_t=3.1 → -88, D_t=3.7 → -92, D_t=5.2 → -96
_DEP_BASE: float = 55.0
_DEP_SCALE: float = 3.5
_DEP_RANGE: float = 45.0

# Manie: Score = +(MAN_BASE + tanh(M_t / MAN_SCALE) * MAN_RANGE)
# Kalibriert auf: M_t=0.5 → +74, M_t=0.8 → +86, M_t=0.9 → +89
_MAN_BASE: float = 55.0
_MAN_SCALE: float = 0.9
_MAN_RANGE: float = 40.0

# "Changed but unclear": Score = tanh(net / UNCLEAR_SCALE) * UNCLEAR_MAX
# net = M_t - D_t; Bereich ±5 bis ±50
_UNCLEAR_SCALE: float = 2.5
_UNCLEAR_MAX: float = 50.0

# "Stable": Score = tanh(net / STABLE_SCALE) * STABLE_MAX
# Bleibt nah an 0, max ±12
_STABLE_SCALE: float = 4.0
_STABLE_MAX: float = 12.0


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
        Zuverlässigkeit [0, 1] basierend auf Audioqualität.
    is_baseline : bool
        True für Tage 1–3 (Score = 0.0).
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
) -> DailyScore:
    """Berechnet den täglichen Mood-Score.

    Parameters
    ----------
    day : int
        Tagnummer (1-basiert). Tage 1–3 → Score = 0 (Baseline).
    final_status : str
        Klassifikation aus dem Pipeline-Algorithmus.
        Mögliche Werte: "stable", "depression-like shift", "mania-like shift",
        "changed but unclear", "abstain_due_to_quality",
        "abstain_insufficient_baseline"
    quality_status : str
        Audioqualität: "valid", "degraded", "reject".
    D_t : float
        Gewichtete Depression-Projektionssumme des Tages.
    M_t : float
        Gewichtete Manie-Projektionssumme des Tages.
    A_t : float
        Gesamtänderungsbetrag (Betrag der Z-Score-Vektors).
    C_D_t : float
        Kumulierter CUSUM-Depressions-Akkumulator.
    C_M_t : float
        Kumulierter CUSUM-Manie-Akkumulator.
    quality_reliability : float
        Zuverlässigkeitswert der Audioqualität [0, 1].

    Returns
    -------
    DailyScore
    """
    # --- Abstain: keine Baseline ---
    if final_status == "abstain_insufficient_baseline":
        return DailyScore(
            day=day, score=None, label="Keine Baseline",
            confidence=0.0, is_baseline=False,
            abstain_reason="insufficient_baseline",
        )

    # --- Abstain: schlechte Audioqualität ---
    if final_status == "abstain_due_to_quality":
        return DailyScore(
            day=day, score=None, label="Unbrauchbare Audioqualität",
            confidence=0.0, is_baseline=False,
            abstain_reason="quality_rejected",
        )

    # --- Baseline-Periode (Tage 1–3): immer 0 ---
    if day < 4:
        return DailyScore(
            day=day, score=0.0, label="Baseline",
            confidence=float(quality_reliability),
            is_baseline=True, abstain_reason=None,
        )

    # --- Ungültige Eingaben ---
    if not all(math.isfinite(v) for v in [D_t, M_t, A_t]):
        return DailyScore(
            day=day, score=None, label="Fehlende Daten",
            confidence=0.0, is_baseline=False,
            abstain_reason="nan_inputs",
        )

    # ------------------------------------------------------------------
    # Berechnung nach final_status
    # ------------------------------------------------------------------

    if final_status == "depression-like shift":
        # Stärke aus D_t: je höher D_t, desto näher an -100
        mag = _DEP_BASE + math.tanh(max(0.0, D_t) / _DEP_SCALE) * _DEP_RANGE
        score = -round(min(100.0, mag), 1)

    elif final_status == "mania-like shift":
        # Stärke aus M_t: je höher M_t, desto näher an +100
        mag = _MAN_BASE + math.tanh(max(0.0, M_t) / _MAN_SCALE) * _MAN_RANGE
        score = +round(min(100.0, mag), 1)

    elif final_status == "changed but unclear":
        # Richtung und Stärke aus netto M_t - D_t
        # Positiv = manische Richtung, negativ = depressive Richtung
        net = M_t - D_t
        score = round(math.tanh(net / _UNCLEAR_SCALE) * _UNCLEAR_MAX, 1)

    else:
        # "stable" (inkl. Fälle die nicht den obigen entsprechen)
        net = M_t - D_t
        score = round(math.tanh(net / _STABLE_SCALE) * _STABLE_MAX, 1)

    return DailyScore(
        day=day,
        score=score,
        label=_label(score),
        confidence=min(1.0, max(0.0, float(quality_reliability))),
        is_baseline=False,
        abstain_reason=None,
    )


# ---------------------------------------------------------------------------
# Convenience-Wrapper für Pipeline-Ausgabe
# ---------------------------------------------------------------------------

def score_from_pipeline_row(row: dict) -> DailyScore:
    """Score aus einem Row-Dict der Pipeline-Ausgabe berechnen.

    Erwartet Keys aus ``day_level_scores.json``:
    ``day``, ``final_status``, ``quality_status``, ``quality_reliability``,
    ``D_t``, ``M_t``, ``A_t``, ``C_D_t``, ``C_M_t``
    """
    return compute_daily_score(
        day=int(row.get("day", 0)),
        final_status=str(row.get("final_status", "stable")),
        quality_status=str(row.get("quality_status", "valid")),
        D_t=float(row.get("D_t", 0.0)),
        M_t=float(row.get("M_t", 0.0)),
        A_t=float(row.get("A_t", 0.0)),
        C_D_t=float(row.get("C_D_t", 0.0)),
        C_M_t=float(row.get("C_M_t", 0.0)),
        quality_reliability=float(row.get("quality_reliability", 1.0)),
    )


def score_dataframe(df: "pd.DataFrame") -> "pd.Series":
    """Score für jede Zeile eines pandas DataFrames.

    Gibt eine ``pd.Series`` mit float-Werten zurück (NaN bei Abstain).
    """
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
    (60.0,  "Stark manisch"),
    (25.0,  "Manisch"),
    (8.0,   "Leicht manisch"),
    (-7.9,  "Normal / Stabil"),
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
