from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import opensmile
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

EPS = 1e-8
TARGET_SR = 16000
WINDOW_SEC = 2.0

ROBUST_FEATURES: Tuple[str, ...] = (
    "tempo_proxy",
    "pause_ratio",
    "f0_median",
    "f0_variability",
    "loudness_median",
    "loudness_variability",
    "voiced_ratio",
    "spectral_flux_median",
    "spectral_flux_variability",
    "hnr_median",
    "alpha_ratio_median",
    "hammarberg_index_median",
)

GLOBAL_WEIGHTS: Dict[str, float] = {
    "tempo_proxy": 0.9,
    "pause_ratio": 1.1,
    "f0_median": 0.7,
    "f0_variability": 0.8,
    "loudness_median": 0.75,
    "loudness_variability": 0.8,
    "voiced_ratio": 0.45,
    "spectral_flux_median": 0.45,
    "spectral_flux_variability": 0.6,
    "hnr_median": 0.35,
    "alpha_ratio_median": 0.35,
    "hammarberg_index_median": 0.35,
}

DEGRADED_MULTIPLIERS: Dict[str, float] = {
    "tempo_proxy": 0.80,
    "pause_ratio": 0.92,
    "f0_median": 0.72,
    "f0_variability": 0.65,
    "loudness_median": 0.78,
    "loudness_variability": 0.72,
    "voiced_ratio": 0.90,
    "spectral_flux_median": 0.68,
    "spectral_flux_variability": 0.68,
    "hnr_median": 0.62,
    "alpha_ratio_median": 0.70,
    "hammarberg_index_median": 0.70,
}

DEP_SIGNS: Dict[str, float] = {
    "tempo_proxy": -1.0,
    "pause_ratio": 1.0,
    "f0_variability": -1.0,
    "loudness_median": -0.6,
    "loudness_variability": -0.8,
    "spectral_flux_variability": -0.4,
}

MAN_SIGNS: Dict[str, float] = {
    "tempo_proxy": 0.8,
    "pause_ratio": -0.8,
    "f0_variability": 1.0,
    "loudness_median": 0.3,
    "loudness_variability": 0.9,
    "spectral_flux_variability": 0.5,
}

FEATURE_FLOOR_DEFAULTS: Dict[str, float] = {
    "tempo_proxy": 0.30,
    "pause_ratio": 0.10,
    "f0_median": 0.70,
    "f0_variability": 0.50,
    "loudness_median": 0.22,
    "loudness_variability": 0.16,
    "voiced_ratio": 0.10,
    "spectral_flux_median": 0.06,
    "spectral_flux_variability": 0.06,
    "hnr_median": 0.80,
    "alpha_ratio_median": 0.15,
    "hammarberg_index_median": 0.25,
}

CUSUM_K_DEP = 0.25
CUSUM_K_MAN = 0.25
CUSUM_TRIGGER = 1.35
DIRECTION_DAY_TRIGGER = 0.50
SEPARATION_MARGIN = 0.40
CHANGE_MAGNITUDE_TRIGGER = 1.00
STABLE_MAGNITUDE_MAX = 0.75
TREND_ALPHA = 0.15
TREND_MAX_ABS = 2.20
MIN_DIRECTION_DAYS = 3


@dataclass(frozen=True)
class QualityMetrics:
    voiced_fraction: float
    snr_proxy_db: float
    clipping_ratio: float
    pitch_success_ratio: float
    spectral_flatness: float
    quality_score: float
    status: str
    reliability: float


class EGeMAPSExtractor:
    def __init__(self) -> None:
        self.lld = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        self.functionals = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def extract(self, wav_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
        lld_df = self.lld.process_file(str(wav_path))
        fun_df = self.functionals.process_file(str(wav_path))
        if fun_df.empty:
            fun_row = pd.Series(dtype=float)
        else:
            fun_row = fun_df.iloc[0]
        return lld_df, fun_row


def _mad(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def _iqr(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    q75, q25 = np.percentile(values, [75, 25])
    return float(q75 - q25)


def _median(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def _safe_float(value: float | int | np.floating | None) -> float:
    if value is None:
        return float("nan")
    out = float(value)
    if not np.isfinite(out):
        return float("nan")
    return out


def _resolve_dataset_root(dataset: str | None) -> Path:
    if dataset:
        p = Path(dataset)
        if p.exists():
            return p
        rel = Path.cwd() / dataset
        if rel.exists():
            return rel

    this_dir = Path(__file__).resolve().parent
    candidates = [
        this_dir / "Hackathon_Dataset_Final",
        this_dir / "hackathon dataset final",
        this_dir / "hackathon_dataset_final",
        Path.cwd() / "hackathon dataset final",
        Path.cwd() / "Hackathon_Dataset_Final",
        Path.cwd() / "classifier" / "Hackathon_Dataset_Final",
        Path.cwd() / "classifier" / "hackathon dataset final",
    ]

    for cand in candidates:
        if cand.exists():
            return cand

    normalized_target = "hackathondatasetfinal"
    for root in [Path.cwd(), this_dir, this_dir.parent]:
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            norm = re.sub(r"[^a-z0-9]", "", child.name.lower())
            if norm == normalized_target:
                return child

    raise FileNotFoundError(
        "Could not locate dataset directory. Checked common names including "
        "'hackathon dataset final' and 'Hackathon_Dataset_Final'."
    )


def _parse_day(stem: str) -> int | None:
    match = re.search(r"day[_\-\s]*(\d+)", stem, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _collect_audio_files(dataset_root: Path) -> List[Tuple[str, int, Path]]:
    rows: List[Tuple[str, int, Path]] = []
    for wav_path in sorted(dataset_root.rglob("*.wav")):
        patient_id = wav_path.parent.name
        day = _parse_day(wav_path.stem)
        if day is None:
            continue
        rows.append((patient_id, day, wav_path))
    return rows


def _load_audio_mono_resampled(wav_path: Path, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(wav_path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float64), target_sr
    peak = np.max(np.abs(audio))
    if np.isfinite(peak) and peak > 0:
        audio = audio / max(peak, 1.0)

    if sr != target_sr and audio.size > 0:
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        audio = resample_poly(audio, up, down)
        sr = target_sr

    return np.asarray(audio, dtype=np.float64), sr


def _frame_signal(signal: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if signal.size == 0:
        return np.zeros((0, frame_len), dtype=np.float64)
    if signal.size < frame_len:
        pad = frame_len - signal.size
        signal = np.pad(signal, (0, pad), mode="constant")
    n_frames = 1 + (signal.size - frame_len) // hop_len
    if n_frames <= 0:
        n_frames = 1
    end = (n_frames - 1) * hop_len + frame_len
    usable = signal[:end]
    shape = (n_frames, frame_len)
    strides = (usable.strides[0] * hop_len, usable.strides[0])
    return np.lib.stride_tricks.as_strided(usable, shape=shape, strides=strides).copy()


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    existing = set(df.columns)
    for cand in candidates:
        if cand in existing:
            return cand
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lowered:
            return lowered[key]
    return None


def _compute_quality_metrics(audio: np.ndarray, sr: int, lld_df: pd.DataFrame) -> QualityMetrics:
    if audio.size == 0:
        return QualityMetrics(0.0, 0.0, 1.0, 0.0, 1.0, 0.0, "reject", 0.0)

    frame_len = max(1, int(0.025 * sr))
    hop_len = max(1, int(0.010 * sr))
    frames = _frame_signal(audio, frame_len=frame_len, hop_len=hop_len)
    if frames.size == 0:
        return QualityMetrics(0.0, 0.0, 1.0, 0.0, 1.0, 0.0, "reject", 0.0)

    rms = np.sqrt(np.mean(frames * frames, axis=1) + EPS)
    rms_db = 20.0 * np.log10(rms + EPS)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)

    window = np.hanning(frame_len)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)) + EPS
    geo = np.exp(np.mean(np.log(spec), axis=1))
    arith = np.mean(spec, axis=1) + EPS
    flatness = geo / arith

    noise_floor_db = np.percentile(rms_db, 20)
    voiced_mask = (rms_db > noise_floor_db + 6.0) & (zcr < 0.25) & (flatness < 0.60)
    if float(np.mean(voiced_mask)) < 0.02:
        voiced_mask = (rms_db > np.percentile(rms_db, 65)) & (flatness < 0.75)

    voiced_fraction = float(np.mean(voiced_mask))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.99))
    spectral_flatness = float(np.median(flatness))

    if np.any(voiced_mask) and np.any(~voiced_mask):
        snr_proxy_db = float(np.percentile(rms_db[voiced_mask], 75) - np.percentile(rms_db[~voiced_mask], 25))
    elif np.any(voiced_mask):
        snr_proxy_db = float(np.percentile(rms_db[voiced_mask], 75) - np.percentile(rms_db, 20))
    else:
        snr_proxy_db = 0.0

    pitch_col = _find_column(lld_df, ["F0semitoneFrom27.5Hz_sma3nz"])
    pitch_success_ratio = 0.0
    if pitch_col:
        f0 = lld_df[pitch_col].to_numpy(dtype=np.float64)
        valid_pitch = np.isfinite(f0) & (f0 > 0.0)
        if valid_pitch.size > 0:
            if voiced_mask.size > 0:
                n = min(valid_pitch.size, voiced_mask.size)
                denom = max(int(np.sum(voiced_mask[:n])), 1)
                pitch_success_ratio = float(np.sum(valid_pitch[:n]) / denom)
            else:
                pitch_success_ratio = float(np.mean(valid_pitch))

    duration_sec = audio.size / max(sr, 1)
    median_rms = float(np.median(rms))

    score_components = {
        "voiced": np.clip((voiced_fraction - 0.05) / 0.35, 0.0, 1.0),
        "snr": np.clip((snr_proxy_db - 2.0) / 12.0, 0.0, 1.0),
        "clip": np.clip((0.08 - clipping_ratio) / 0.08, 0.0, 1.0),
        "pitch": np.clip((pitch_success_ratio - 0.10) / 0.75, 0.0, 1.0),
        "flatness": np.clip((0.65 - spectral_flatness) / 0.45, 0.0, 1.0),
    }
    quality_score = float(
        0.30 * score_components["voiced"]
        + 0.25 * score_components["snr"]
        + 0.15 * score_components["clip"]
        + 0.20 * score_components["pitch"]
        + 0.10 * score_components["flatness"]
    )

    reject = (
        duration_sec < 1.0
        or median_rms < 8e-4
        or voiced_fraction < 0.06
        or snr_proxy_db < 3.0
        or clipping_ratio > 0.10
        or pitch_success_ratio < 0.15
        or quality_score < 0.28
    )
    degraded = (
        voiced_fraction < 0.16
        or snr_proxy_db < 8.0
        or clipping_ratio > 0.02
        or pitch_success_ratio < 0.40
        or spectral_flatness > 0.55
        or quality_score < 0.62
    )

    if reject:
        status = "reject"
        reliability = 0.0
    elif degraded:
        status = "degraded"
        reliability = float(np.clip(quality_score, 0.35, 0.75))
    else:
        status = "valid"
        reliability = float(np.clip(max(quality_score, 0.80), 0.0, 1.0))

    return QualityMetrics(
        voiced_fraction=voiced_fraction,
        snr_proxy_db=snr_proxy_db,
        clipping_ratio=clipping_ratio,
        pitch_success_ratio=pitch_success_ratio,
        spectral_flatness=spectral_flatness,
        quality_score=quality_score,
        status=status,
        reliability=reliability,
    )


def _count_segments(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    starts = mask & np.concatenate((np.array([True]), ~mask[:-1]))
    return int(np.sum(starts))


def _window_feature_lists(lld_df: pd.DataFrame, window_sec: float = WINDOW_SEC) -> Dict[str, List[float]]:
    results: Dict[str, List[float]] = {
        "tempo_proxy": [],
        "pause_ratio": [],
        "f0_median": [],
        "f0_variability": [],
        "loudness_median": [],
        "loudness_variability": [],
        "voiced_ratio": [],
        "spectral_flux_median": [],
        "spectral_flux_variability": [],
        "hnr_median": [],
        "alpha_ratio_median": [],
        "hammarberg_index_median": [],
    }
    if lld_df.empty:
        return results

    start_index = lld_df.index.get_level_values("start")
    if isinstance(start_index, pd.TimedeltaIndex):
        starts_sec = start_index.total_seconds().to_numpy(dtype=np.float64)
    else:
        starts_sec = pd.to_timedelta(start_index).total_seconds().to_numpy(dtype=np.float64)

    f0_col = _find_column(lld_df, ["F0semitoneFrom27.5Hz_sma3nz"])
    loud_col = _find_column(lld_df, ["Loudness_sma3", "loudness_sma3"])
    spectral_flux_col = _find_column(lld_df, ["spectralFlux_sma3"])
    hnr_col = _find_column(lld_df, ["HNRdBACF_sma3nz"])
    alpha_ratio_col = _find_column(lld_df, ["alphaRatio_sma3"])
    hammarberg_col = _find_column(lld_df, ["hammarbergIndex_sma3"])

    if loud_col is None:
        return results

    loud = lld_df[loud_col].to_numpy(dtype=np.float64)
    spectral_flux = (
        lld_df[spectral_flux_col].to_numpy(dtype=np.float64)
        if spectral_flux_col
        else np.full(loud.shape, np.nan, dtype=np.float64)
    )
    hnr = (
        lld_df[hnr_col].to_numpy(dtype=np.float64)
        if hnr_col
        else np.full(loud.shape, np.nan, dtype=np.float64)
    )
    alpha_ratio = (
        lld_df[alpha_ratio_col].to_numpy(dtype=np.float64)
        if alpha_ratio_col
        else np.full(loud.shape, np.nan, dtype=np.float64)
    )
    hammarberg = (
        lld_df[hammarberg_col].to_numpy(dtype=np.float64)
        if hammarberg_col
        else np.full(loud.shape, np.nan, dtype=np.float64)
    )
    if f0_col is None:
        f0 = np.full(loud.shape, np.nan, dtype=np.float64)
        voiced = loud > np.percentile(loud[np.isfinite(loud)], 55)
    else:
        f0 = lld_df[f0_col].to_numpy(dtype=np.float64)
        voiced = np.isfinite(f0) & (f0 > 0.0)
        if float(np.mean(voiced)) < 0.02:
            voiced = loud > np.percentile(loud[np.isfinite(loud)], 60)

    window_ids = np.floor(starts_sec / max(window_sec, 0.1)).astype(int)
    step_sec = float(np.nanmedian(np.diff(starts_sec))) if starts_sec.size > 1 else 0.01
    if not np.isfinite(step_sec) or step_sec <= 0:
        step_sec = 0.01

    for w_id in np.unique(window_ids):
        sel = window_ids == w_id
        if not np.any(sel):
            continue
        voiced_w = voiced[sel]
        loud_w = loud[sel]
        f0_w = f0[sel]
        spectral_flux_w = spectral_flux[sel]
        hnr_w = hnr[sel]
        alpha_ratio_w = alpha_ratio[sel]
        hammarberg_w = hammarberg[sel]

        voiced_ratio = float(np.mean(voiced_w))
        pause_ratio = float(1.0 - voiced_ratio)

        duration = max(step_sec * np.sum(sel), 0.05)
        tempo_proxy = _count_segments(voiced_w) / duration

        f0_valid = f0_w[np.isfinite(f0_w) & (f0_w > 0.0)]
        loud_valid = loud_w[np.isfinite(loud_w)]
        spectral_flux_valid = spectral_flux_w[np.isfinite(spectral_flux_w)]
        hnr_valid = hnr_w[np.isfinite(hnr_w)]
        alpha_ratio_valid = alpha_ratio_w[np.isfinite(alpha_ratio_w)]
        hammarberg_valid = hammarberg_w[np.isfinite(hammarberg_w)]

        results["tempo_proxy"].append(float(tempo_proxy))
        results["pause_ratio"].append(float(pause_ratio))
        results["f0_median"].append(_median(f0_valid))
        results["f0_variability"].append(_iqr(f0_valid))
        results["loudness_median"].append(_median(loud_valid))
        results["loudness_variability"].append(_iqr(loud_valid))
        results["voiced_ratio"].append(float(voiced_ratio))
        results["spectral_flux_median"].append(_median(spectral_flux_valid))
        results["spectral_flux_variability"].append(_iqr(spectral_flux_valid))
        results["hnr_median"].append(_median(hnr_valid))
        results["alpha_ratio_median"].append(_median(alpha_ratio_valid))
        results["hammarberg_index_median"].append(_median(hammarberg_valid))

    return results


def _aggregate_day_features(
    lld_df: pd.DataFrame,
    fun_row: pd.Series,
    quality_voiced_fraction: float,
) -> Dict[str, float]:
    win_lists = _window_feature_lists(lld_df)

    out: Dict[str, float] = {}
    for feature_name in ROBUST_FEATURES:
        arr = np.asarray([v for v in win_lists.get(feature_name, []) if np.isfinite(v)], dtype=np.float64)
        out[feature_name] = _median(arr)
        out[f"{feature_name}_window_iqr"] = _iqr(arr)

    # Functional fallbacks keep eGeMAPS compatibility when window-level estimates are sparse.
    if np.isnan(out["tempo_proxy"]):
        out["tempo_proxy"] = _safe_float(fun_row.get("VoicedSegmentsPerSec"))

    if np.isnan(out["f0_median"]):
        out["f0_median"] = _safe_float(fun_row.get("F0semitoneFrom27.5Hz_sma3nz_percentile50.0"))

    if np.isnan(out["f0_variability"]):
        out["f0_variability"] = _safe_float(fun_row.get("F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2"))

    if np.isnan(out["loudness_median"]):
        out["loudness_median"] = _safe_float(fun_row.get("loudness_sma3_percentile50.0"))

    if np.isnan(out["loudness_variability"]):
        out["loudness_variability"] = _safe_float(fun_row.get("loudness_sma3_pctlrange0-2"))

    if np.isnan(out["spectral_flux_median"]):
        out["spectral_flux_median"] = _safe_float(fun_row.get("spectralFlux_sma3_amean"))

    if np.isnan(out["spectral_flux_variability"]):
        out["spectral_flux_variability"] = _safe_float(fun_row.get("spectralFlux_sma3_stddevNorm"))

    if np.isnan(out["hnr_median"]):
        out["hnr_median"] = _safe_float(fun_row.get("HNRdBACF_sma3nz_amean"))

    if np.isnan(out["alpha_ratio_median"]):
        out["alpha_ratio_median"] = _safe_float(fun_row.get("alphaRatioV_sma3nz_amean"))

    if np.isnan(out["hammarberg_index_median"]):
        out["hammarberg_index_median"] = _safe_float(fun_row.get("hammarbergIndexV_sma3nz_amean"))

    if np.isnan(out["pause_ratio"]):
        uv = _safe_float(fun_row.get("MeanUnvoicedSegmentLength"))
        vv = _safe_float(fun_row.get("MeanVoicedSegmentLengthSec"))
        if np.isfinite(uv) and np.isfinite(vv) and (uv + vv) > 0:
            out["pause_ratio"] = float(np.clip(uv / (uv + vv), 0.0, 1.0))

    if np.isnan(out["voiced_ratio"]):
        if np.isfinite(out.get("pause_ratio", np.nan)):
            out["voiced_ratio"] = float(np.clip(1.0 - out["pause_ratio"], 0.0, 1.0))
        else:
            out["voiced_ratio"] = float(np.clip(quality_voiced_fraction, 0.0, 1.0))

    if np.isnan(out.get("pause_ratio", np.nan)) and np.isfinite(out.get("voiced_ratio", np.nan)):
        out["pause_ratio"] = float(np.clip(1.0 - out["voiced_ratio"], 0.0, 1.0))

    out["pause_ratio"] = float(np.clip(out.get("pause_ratio", np.nan), 0.0, 1.0)) if np.isfinite(out.get("pause_ratio", np.nan)) else float("nan")
    out["voiced_ratio"] = float(np.clip(out.get("voiced_ratio", np.nan), 0.0, 1.0)) if np.isfinite(out.get("voiced_ratio", np.nan)) else float("nan")

    return out


def _analyze_clip(wav_path: Path, extractor: EGeMAPSExtractor, dataset_root: Path) -> Dict[str, object]:
    patient_id = wav_path.parent.name
    day = _parse_day(wav_path.stem)
    if day is None:
        raise ValueError(f"Could not parse day index from file name: {wav_path.name}")

    record: Dict[str, object] = {
        "patient_id": patient_id,
        "day": day,
        "audio_path": str(wav_path.resolve().relative_to(dataset_root.resolve())),
        "analysis_error": "",
    }

    try:
        audio, sr = _load_audio_mono_resampled(wav_path)
        duration_sec = float(audio.size / max(sr, 1))
        lld_df, fun_row = extractor.extract(wav_path)
        quality = _compute_quality_metrics(audio, sr, lld_df)
        features = _aggregate_day_features(lld_df, fun_row, quality.voiced_fraction)

        record.update(
            {
                "duration_sec": duration_sec,
                "quality_status": quality.status,
                "quality_reliability": quality.reliability,
                "quality_score": quality.quality_score,
                "quality_voiced_fraction": quality.voiced_fraction,
                "quality_snr_proxy_db": quality.snr_proxy_db,
                "quality_clipping_ratio": quality.clipping_ratio,
                "quality_pitch_success_ratio": quality.pitch_success_ratio,
                "quality_spectral_flatness": quality.spectral_flatness,
            }
        )
        record.update(features)
    except Exception as exc:
        record.update(
            {
                "duration_sec": 0.0,
                "quality_status": "reject",
                "quality_reliability": 0.0,
                "quality_score": 0.0,
                "quality_voiced_fraction": 0.0,
                "quality_snr_proxy_db": 0.0,
                "quality_clipping_ratio": 1.0,
                "quality_pitch_success_ratio": 0.0,
                "quality_spectral_flatness": 1.0,
                "analysis_error": str(exc),
            }
        )
        for feat in ROBUST_FEATURES:
            record[feat] = float("nan")
            record[f"{feat}_window_iqr"] = float("nan")

    return record


def _compute_pooled_floors(day_df: pd.DataFrame) -> Dict[str, float]:
    baseline_pool = day_df[(day_df["day"].between(1, 3)) & (day_df["quality_status"] != "reject")]
    floors: Dict[str, float] = {}
    for feat in ROBUST_FEATURES:
        vals = baseline_pool[feat].dropna().to_numpy(dtype=np.float64)
        if vals.size >= 4:
            scale = 1.4826 * _mad(vals)
            floor = max(scale * 0.20, FEATURE_FLOOR_DEFAULTS[feat])
        else:
            floor = FEATURE_FLOOR_DEFAULTS[feat]
        floors[feat] = float(max(floor, EPS))
    return floors


def _feature_reliability(feature: str, quality_status: str, quality_reliability: float) -> float:
    if quality_status == "reject":
        return 0.0
    if quality_status == "degraded":
        return float(np.clip(quality_reliability * DEGRADED_MULTIPLIERS.get(feature, 0.65), 0.0, 1.0))
    return float(np.clip(quality_reliability, 0.0, 1.0))


def _apply_patient_direction_consistency(scored_df: pd.DataFrame) -> pd.DataFrame:
    df = scored_df.sort_values("day").copy()
    dep_count = int((df["final_status"] == "depression-like shift").sum())
    man_count = int((df["final_status"] == "mania-like shift").sum())

    chosen_direction: str | None
    if dep_count > 0 and man_count > 0:
        chosen_direction = None
    elif dep_count >= MIN_DIRECTION_DAYS and man_count == 0:
        chosen_direction = "depression-like shift"
    elif man_count >= MIN_DIRECTION_DAYS and dep_count == 0:
        chosen_direction = "mania-like shift"
    else:
        chosen_direction = None

    if chosen_direction is None:
        for idx, row in df.iterrows():
            if int(row.get("day", 0)) < 4:
                continue
            if str(row.get("final_status", "")) in {"abstain_due_to_quality", "abstain_insufficient_baseline"}:
                continue
            if str(row.get("quality_status", "")) == "degraded":
                a_t = _safe_float(row.get("A_t"))
                if np.isfinite(a_t) and a_t <= 1.60:
                    df.at[idx, "final_status"] = "stable"
                else:
                    df.at[idx, "final_status"] = "changed but unclear"
            else:
                df.at[idx, "final_status"] = "stable"
        return df

    opposite = "mania-like shift" if chosen_direction == "depression-like shift" else "depression-like shift"
    df.loc[df["final_status"] == opposite, "final_status"] = "changed but unclear"
    return df


def _score_patient(patient_df: pd.DataFrame, pooled_floors: Dict[str, float]) -> pd.DataFrame:
    df = patient_df.sort_values("day").copy()

    baseline_rows = df[(df["day"].isin([1, 2, 3])) & (df["quality_status"] != "reject")]
    baseline_days_present = set(int(d) for d in baseline_rows["day"].tolist())
    baseline_ready = baseline_days_present == {1, 2, 3}

    baseline_center: Dict[str, float] = {f: float("nan") for f in ROBUST_FEATURES}
    baseline_scale: Dict[str, float] = {f: float("nan") for f in ROBUST_FEATURES}

    if baseline_ready:
        for feat in ROBUST_FEATURES:
            vals = baseline_rows[feat].dropna().to_numpy(dtype=np.float64)
            if vals.size < 2:
                baseline_ready = False
                break
            center = float(np.median(vals))
            scale = float(max(1.4826 * _mad(vals), pooled_floors[feat], EPS))
            baseline_center[feat] = center
            baseline_scale[feat] = scale

    c_dep = 0.0
    c_man = 0.0
    dep_consecutive = 0
    man_consecutive = 0
    dep_trigger_streak = 0
    man_trigger_streak = 0
    trend_offset: Dict[str, float] = {f: 0.0 for f in ROBUST_FEATURES}

    scored_rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        day = int(row["day"])
        quality_status = str(row["quality_status"])
        quality_reliability = float(row.get("quality_reliability", 0.0) or 0.0)

        out = row.to_dict()
        out["baseline_ready"] = bool(baseline_ready)
        out["baseline_days_used"] = "1,2,3" if baseline_ready else "insufficient"

        for feat in ROBUST_FEATURES:
            out[f"baseline_{feat}"] = baseline_center.get(feat, float("nan"))
            out[f"baseline_scale_{feat}"] = baseline_scale.get(feat, float("nan"))

        if not baseline_ready:
            out["A_t"] = float("nan")
            out["D_t"] = float("nan")
            out["M_t"] = float("nan")
            out["C_D_t"] = float("nan")
            out["C_M_t"] = float("nan")
            out["final_status"] = "abstain_insufficient_baseline"
            scored_rows.append(out)
            continue

        if quality_status == "reject":
            out["A_t"] = float("nan")
            out["D_t"] = float("nan")
            out["M_t"] = float("nan")
            out["C_D_t"] = c_dep
            out["C_M_t"] = c_man
            out["final_status"] = "abstain_due_to_quality"
            scored_rows.append(out)
            continue

        z_map: Dict[str, float] = {}
        z_anchor_map: Dict[str, float] = {}
        wrz_values: List[float] = []
        d_sum = 0.0
        m_sum = 0.0

        for feat in ROBUST_FEATURES:
            x_val = _safe_float(row.get(feat))
            if not np.isfinite(x_val):
                z_val = 0.0
                z_anchor = 0.0
                rel = 0.0
            else:
                if day < 4:
                    z_anchor = 0.0
                    z_val = 0.0
                else:
                    z_raw = (x_val - baseline_center[feat]) / (baseline_scale[feat] + EPS)
                    z_anchor = float(np.clip(z_raw, -3.0, 3.0))
                    z_val = float(np.clip(z_anchor - trend_offset[feat], -3.0, 3.0))
                rel = _feature_reliability(feat, quality_status, quality_reliability)

            z_map[feat] = z_val
            z_anchor_map[feat] = z_anchor
            wrz = GLOBAL_WEIGHTS[feat] * rel * z_val
            wrz_values.append(wrz)
            d_sum += GLOBAL_WEIGHTS[feat] * rel * DEP_SIGNS.get(feat, 0.0) * z_val
            m_sum += GLOBAL_WEIGHTS[feat] * rel * MAN_SIGNS.get(feat, 0.0) * z_val
            out[f"z_{feat}"] = z_val
            out[f"z_anchor_{feat}"] = z_anchor
            out[f"r_{feat}"] = rel

        a_t = float(math.sqrt(np.sum(np.square(np.asarray(wrz_values, dtype=np.float64)))))
        d_t = float(d_sum)
        m_t = float(m_sum)

        if day >= 4:
            # Decay prevents stale directional evidence from dominating later recovery days.
            c_dep = max(0.0, (0.88 * c_dep) + d_t - CUSUM_K_DEP)
            c_man = max(0.0, (0.88 * c_man) + m_t - CUSUM_K_MAN)
            dep_consecutive = dep_consecutive + 1 if d_t > DIRECTION_DAY_TRIGGER else 0
            man_consecutive = man_consecutive + 1 if m_t > DIRECTION_DAY_TRIGGER else 0
        else:
            dep_consecutive = 0
            man_consecutive = 0

        high_confidence_quality = quality_status == "valid" and quality_reliability >= 0.80

        dep_support = 0
        dep_oppose = 0
        man_support = 0
        man_oppose = 0
        for feat, sign in DEP_SIGNS.items():
            rel = float(out.get(f"r_{feat}", 0.0) or 0.0)
            if rel < 0.25:
                continue
            signed = sign * float(z_map.get(feat, 0.0))
            if signed > 0.45:
                dep_support += 1
            elif signed < -0.45:
                dep_oppose += 1
        for feat, sign in MAN_SIGNS.items():
            rel = float(out.get(f"r_{feat}", 0.0) or 0.0)
            if rel < 0.25:
                continue
            signed = sign * float(z_map.get(feat, 0.0))
            if signed > 0.45:
                man_support += 1
            elif signed < -0.45:
                man_oppose += 1

        dep_var_support = 0
        man_var_support = 0
        for feat in ("f0_variability", "loudness_variability", "spectral_flux_variability"):
            rel = float(out.get(f"r_{feat}", 0.0) or 0.0)
            if rel < 0.25:
                continue
            z_val = float(z_map.get(feat, 0.0))
            if z_val <= -0.10:
                dep_var_support += 1
            if z_val >= 0.10:
                man_var_support += 1

        dep_trigger = (
            day >= 4
            and high_confidence_quality
            and dep_consecutive >= 1
            and c_dep >= CUSUM_TRIGGER
            and (c_dep - c_man) >= SEPARATION_MARGIN
            and d_t > DIRECTION_DAY_TRIGGER
            and dep_support >= 2
            and dep_oppose <= 1
            and dep_var_support >= 1
            and float(z_map.get("f0_variability", 0.0)) < -0.10
            and float(z_map.get("pause_ratio", 0.0)) > -0.35
        )
        man_trigger = (
            day >= 4
            and high_confidence_quality
            and man_consecutive >= 1
            and c_man >= CUSUM_TRIGGER
            and (c_man - c_dep) >= SEPARATION_MARGIN
            and m_t > DIRECTION_DAY_TRIGGER
            and man_support >= 2
            and man_oppose <= 1
            and man_var_support >= 1
            and float(z_map.get("f0_variability", 0.0)) > 0.10
        )

        dep_trigger_streak = dep_trigger_streak + 1 if dep_trigger else 0
        man_trigger_streak = man_trigger_streak + 1 if man_trigger else 0

        if dep_trigger_streak >= 1 and man_trigger_streak == 0:
            final_status = "depression-like shift"
        elif man_trigger_streak >= 1 and dep_trigger_streak == 0:
            final_status = "mania-like shift"
        elif day < 4:
            final_status = "stable"
        elif quality_status == "degraded":
            final_status = "stable" if a_t <= STABLE_MAGNITUDE_MAX else "changed but unclear"
        elif day >= 6 and a_t < 1.60 and max(abs(d_t), abs(m_t)) < 0.85:
            final_status = "stable"
        elif a_t <= STABLE_MAGNITUDE_MAX and max(abs(d_t), abs(m_t)) < 0.80:
            final_status = "stable"
        elif a_t <= CHANGE_MAGNITUDE_TRIGGER and abs(d_t - m_t) < 0.40:
            final_status = "stable"
        elif high_confidence_quality and day >= 4 and a_t >= 0.95:
            if (
                (m_t - d_t) >= 0.35
                and m_t > DIRECTION_DAY_TRIGGER
                and man_var_support >= 1
                and float(z_map.get("f0_variability", 0.0)) > 0.10
            ):
                final_status = "mania-like shift"
            elif (
                (d_t - m_t) >= 0.35
                and d_t > DIRECTION_DAY_TRIGGER
                and dep_var_support >= 1
                and float(z_map.get("f0_variability", 0.0)) < -0.10
                and float(z_map.get("pause_ratio", 0.0)) > -0.35
            ):
                final_status = "depression-like shift"
            else:
                final_status = "changed but unclear"
        elif a_t >= CHANGE_MAGNITUDE_TRIGGER:
            final_status = "changed but unclear"
        else:
            final_status = "changed but unclear"

        out["A_t"] = a_t
        out["D_t"] = d_t
        out["M_t"] = m_t
        out["C_D_t"] = float(c_dep)
        out["C_M_t"] = float(c_man)
        out["final_status"] = final_status

        if day >= 4 and quality_status != "reject":
            if final_status in {"stable", "changed but unclear"}:
                for feat in ROBUST_FEATURES:
                    rel = float(out.get(f"r_{feat}", 0.0) or 0.0)
                    if rel <= 0.0:
                        continue
                    anchor = float(z_anchor_map.get(feat, 0.0))
                    trend_offset[feat] = float(
                        np.clip(
                            ((1.0 - TREND_ALPHA) * trend_offset[feat]) + (TREND_ALPHA * anchor),
                            -TREND_MAX_ABS,
                            TREND_MAX_ABS,
                        )
                    )
            else:
                for feat in ROBUST_FEATURES:
                    trend_offset[feat] = float(0.95 * trend_offset[feat])

        scored_rows.append(out)

    return _apply_patient_direction_consistency(pd.DataFrame(scored_rows))


def _build_patient_summary(scored_df: pd.DataFrame) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    for patient_id, group in scored_df.groupby("patient_id"):
        g = group.sort_values("day")
        quality_counts = g["quality_status"].value_counts().to_dict()
        final_counts = g["final_status"].value_counts().to_dict()

        dep_days = g[g["final_status"] == "depression-like shift"]["day"]
        man_days = g[g["final_status"] == "mania-like shift"]["day"]
        unclear_days = g[g["final_status"] == "changed but unclear"]["day"]

        summary_rows.append(
            {
                "patient_id": patient_id,
                "days": int(g.shape[0]),
                "baseline_ready": bool(g["baseline_ready"].iloc[0]),
                "quality_valid_days": int(quality_counts.get("valid", 0)),
                "quality_degraded_days": int(quality_counts.get("degraded", 0)),
                "quality_reject_days": int(quality_counts.get("reject", 0)),
                "stable_days": int(final_counts.get("stable", 0)),
                "depression_like_days": int(final_counts.get("depression-like shift", 0)),
                "mania_like_days": int(final_counts.get("mania-like shift", 0)),
                "changed_unclear_days": int(final_counts.get("changed but unclear", 0)),
                "abstain_quality_days": int(final_counts.get("abstain_due_to_quality", 0)),
                "abstain_baseline_days": int(final_counts.get("abstain_insufficient_baseline", 0)),
                "first_depression_like_day": int(dep_days.min()) if not dep_days.empty else None,
                "first_mania_like_day": int(man_days.min()) if not man_days.empty else None,
                "first_changed_unclear_day": int(unclear_days.min()) if not unclear_days.empty else None,
            }
        )
    return summary_rows


def _write_outputs(
    scored_df: pd.DataFrame,
    summary_rows: List[Dict[str, object]],
    output_dir: Path,
    dataset_root: Path,
    pooled_floors: Dict[str, float],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    scored_csv = output_dir / "day_level_scores.csv"
    scored_json = output_dir / "day_level_scores.json"
    summary_json = output_dir / "patient_summary.json"
    summary_txt = output_dir / "patient_summary.txt"
    config_json = output_dir / "pipeline_config.json"

    ordered = scored_df.sort_values(["patient_id", "day"]).reset_index(drop=True)
    ordered.to_csv(scored_csv, index=False)
    ordered.to_json(scored_json, orient="records", indent=2)

    with summary_json.open("w", encoding="utf-8") as fh:
        json.dump(summary_rows, fh, indent=2)

    lines: List[str] = []
    for row in summary_rows:
        lines.append(
            f"{row['patient_id']}: baseline_ready={row['baseline_ready']}; "
            f"quality(valid/degraded/reject)={row['quality_valid_days']}/"
            f"{row['quality_degraded_days']}/{row['quality_reject_days']}; "
            f"status(stable/dep/mania/unclear/abstain_q/abstain_b)="
            f"{row['stable_days']}/{row['depression_like_days']}/{row['mania_like_days']}/"
            f"{row['changed_unclear_days']}/{row['abstain_quality_days']}/{row['abstain_baseline_days']}; "
            f"first_dep={row['first_depression_like_day']}; "
            f"first_mania={row['first_mania_like_day']}; "
            f"first_unclear={row['first_changed_unclear_day']}"
        )
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    config_payload = {
        "dataset_root": str(dataset_root.resolve()),
        "target_sr": TARGET_SR,
        "window_sec": WINDOW_SEC,
        "robust_features": list(ROBUST_FEATURES),
        "global_weights": GLOBAL_WEIGHTS,
        "degraded_multipliers": DEGRADED_MULTIPLIERS,
        "depression_signs": DEP_SIGNS,
        "mania_signs": MAN_SIGNS,
        "feature_floor_defaults": FEATURE_FLOOR_DEFAULTS,
        "pooled_feature_floors": pooled_floors,
        "cusum": {
            "k_dep": CUSUM_K_DEP,
            "k_man": CUSUM_K_MAN,
            "trigger": CUSUM_TRIGGER,
            "direction_day_trigger": DIRECTION_DAY_TRIGGER,
            "separation_margin": SEPARATION_MARGIN,
            "change_magnitude_trigger": CHANGE_MAGNITUDE_TRIGGER,
            "stable_magnitude_max": STABLE_MAGNITUDE_MAX,
            "min_direction_days_for_patient": MIN_DIRECTION_DAYS,
        },
        "trend_channel": {
            "alpha": TREND_ALPHA,
            "max_abs_offset": TREND_MAX_ABS,
            "applies_when": ["stable", "changed but unclear"],
        },
        "quality_logic": {
            "labels": ["valid", "degraded", "reject"],
            "indicators": [
                "voiced_fraction",
                "snr_proxy_db",
                "clipping_ratio",
                "pitch_success_ratio",
                "spectral_flatness",
            ],
        },
    }
    with config_json.open("w", encoding="utf-8") as fh:
        json.dump(config_payload, fh, indent=2)

    return {
        "day_level_csv": scored_csv,
        "day_level_json": scored_json,
        "summary_json": summary_json,
        "summary_txt": summary_txt,
        "config_json": config_json,
    }


def run_longitudinal_pipeline(dataset: str | None = None, output_dir: str | None = None) -> Dict[str, object]:
    dataset_root = _resolve_dataset_root(dataset)
    out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "results"

    files = _collect_audio_files(dataset_root)
    if not files:
        raise RuntimeError(f"No .wav files found under dataset root: {dataset_root}")

    extractor = EGeMAPSExtractor()

    records: List[Dict[str, object]] = []
    for patient_id, day, wav_path in files:
        _ = patient_id, day
        records.append(_analyze_clip(wav_path, extractor, dataset_root))

    day_df = pd.DataFrame(records).sort_values(["patient_id", "day"]).reset_index(drop=True)
    pooled_floors = _compute_pooled_floors(day_df)

    scored_parts: List[pd.DataFrame] = []
    for _, patient_group in day_df.groupby("patient_id"):
        scored_parts.append(_score_patient(patient_group, pooled_floors=pooled_floors))

    scored_df = pd.concat(scored_parts, axis=0).sort_values(["patient_id", "day"]).reset_index(drop=True)
    summary_rows = _build_patient_summary(scored_df)
    output_paths = _write_outputs(scored_df, summary_rows, out_dir, dataset_root, pooled_floors)

    return {
        "dataset_root": str(dataset_root.resolve()),
        "num_files": int(len(files)),
        "num_patients": int(scored_df["patient_id"].nunique()),
        "output_paths": {k: str(v.resolve()) for k, v in output_paths.items()},
        "summary": summary_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic personalized longitudinal audio change detector (non-ML)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset folder path (defaults to auto-detected hackathon dataset directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for CSV/JSON/TXT outputs (default: classifier/results).",
    )
    args = parser.parse_args()

    result = run_longitudinal_pipeline(dataset=args.dataset, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
