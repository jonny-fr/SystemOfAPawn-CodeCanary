"""
classifier/analyzer.py
======================
Single-day analyzer that wraps the batch pipeline for incremental use.

Flow
----
Day 1-3  → preprocess + extract features → store as baseline (is_baseline=True, score=None)
Day 4+   → preprocess + extract features → rebuild full time-series from DB features
           → compute personal mu/sigma from days 1-3 → run classify_patient_timeseries
           → map result to DailyScore via scoring.py

Personal baseline
-----------------
Replaces compute_neutral_reference() (which needs Patient D/E data) by computing
mu / sigma directly from the user's own days 1-3 features using median + MAD,
the same robust estimator used in the original neutral reference function.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from classifier.pipeline import (
    FEATURE_NAMES,
    FEATURE_WEIGHTS,
    SIGN_DEP,
    SIGN_MAN,
    classify_patient_timeseries,
    compute_state_score,
    compute_z_scores,
    extract_features,
    preprocess_audio,
)
from classifier.scoring import DailyScore, compute_daily_score


# ---------------------------------------------------------------------------
# Internal: personal baseline helpers
# ---------------------------------------------------------------------------

def _compute_personal_mu_sigma(
    baseline_features: list[dict],
) -> tuple[dict, dict]:
    """
    Compute mu / sigma from the user's own baseline days (1-3).

    Uses median + MAD (robust, identical to compute_neutral_reference step 3).
    Handles 1, 2, or 3 valid days gracefully.
    """
    mu: dict[str, float] = {}
    sigma: dict[str, float] = {}

    valid = [d for d in baseline_features if d is not None]

    for feature in FEATURE_NAMES:
        values = np.array([
            d[feature] for d in valid
            if feature in d and d[feature] is not None and math.isfinite(d[feature])
        ], dtype=float)

        if len(values) == 0:
            mu[feature] = 0.0
            sigma[feature] = 1.0
        elif len(values) == 1:
            mu[feature] = float(values[0])
            # Small sigma so any deviation shows up; avoids division by ~0
            sigma[feature] = max(abs(float(values[0])) * 0.1, 1e-6)
        else:
            med = float(np.median(values))
            mad = float(np.median(np.abs(values - med)))
            mu[feature] = med
            sigma[feature] = float(1.4826 * mad + 1e-8)

    return mu, sigma


def _compute_personal_thresholds(
    baseline_features: list[dict],
    mu: dict,
    sigma: dict,
) -> tuple[float, float]:
    """
    Thresholds for dep / man state scores derived from the personal baseline.

    With personal mu/sigma, baseline z-scores centre around 0 so scores are
    near 0. We use max(baseline scores) * 1.5 with a generous floor so a
    day must deviate meaningfully from the user's own baseline to trigger.
    """
    dep_scores = []
    man_scores = []
    for f in baseline_features:
        if f is None:
            continue
        z = compute_z_scores(f, mu, sigma)
        dep_scores.append(compute_state_score(z, SIGN_DEP, FEATURE_WEIGHTS))
        man_scores.append(compute_state_score(z, SIGN_MAN, FEATURE_WEIGHTS))

    floor = 0.5
    threshold_dep = max(max(dep_scores, default=floor) * 1.5, floor)
    threshold_man = max(max(man_scores, default=floor) * 1.5, floor)
    return threshold_dep, threshold_man


# ---------------------------------------------------------------------------
# Label mapping: pipeline label_smoothed → scoring.py final_status
# ---------------------------------------------------------------------------

_LABEL_TO_STATUS: dict[str, str] = {
    'normal':            'stable',
    'depression-onset':  'depression-like shift',
    'depression-like':   'depression-like shift',
    'mania-onset':       'mania-like shift',
    'mania-like':        'mania-like shift',
    'unclear':           'changed but unclear',
    'reject':            'abstain_due_to_quality',
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_single_day(
    audio_path: str,
    day_number: int,
    previous_features: list[Optional[dict]],
    previous_qualities: list[str],
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> dict:
    """
    Analyze one audio file in the context of all previous recordings.

    Parameters
    ----------
    audio_path : str
        Path to the audio file for today.
    day_number : int
        1-based day number.  Days 1-3 are baseline (no score returned).
    previous_features : list[dict | None]
        Feature dicts for days 1..(day_number-1), ordered by day.
        None entries represent rejected recordings.
    previous_qualities : list[str]
        Quality strings ('clean'/'degraded'/'reject') parallel to previous_features.
    progress_callback : callable(int, str) or None
        Called with (percent 0-100, german status message) during processing.

    Returns
    -------
    dict
        features        : dict of 15 raw acoustic features (+ 'reliability')
        quality         : str  ('clean' | 'degraded' | 'reject')
        is_baseline     : bool
        score           : float or None
        state           : str  (pipeline label, e.g. 'normal', 'depression-like')
        confidence      : float
        dep_score       : float or None   (raw D_t from pipeline)
        man_score       : float or None   (raw M_t from pipeline)
        daily_score_obj : DailyScore
    """
    def _prog(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    # ------------------------------------------------------------------
    # Step 1: Preprocess audio
    # ------------------------------------------------------------------
    _prog(10, 'Audio wird geladen...')
    preprocessed = preprocess_audio(audio_path)
    quality: str = preprocessed['quality']

    # ------------------------------------------------------------------
    # Step 2: Extract features
    # ------------------------------------------------------------------
    _prog(30, 'Merkmale werden extrahiert...')
    features: Optional[dict] = extract_features(preprocessed)

    # Rejected audio or extraction failure
    if features is None or quality == 'reject':
        _prog(100, 'Analyse abgeschlossen.')
        daily_score_obj = compute_daily_score(
            day=day_number,
            final_status='abstain_due_to_quality',
            quality_status='reject',
            D_t=0.0, M_t=0.0, A_t=0.0,
            quality_reliability=0.0,
        )
        return {
            'features': {},
            'quality': 'reject',
            'is_baseline': day_number <= 3,
            'score': None,
            'state': 'reject',
            'confidence': 0.0,
            'dep_score': None,
            'man_score': None,
            'daily_score_obj': daily_score_obj,
        }

    reliability = features.get('reliability', 1.0)

    # ------------------------------------------------------------------
    # Step 3: Baseline days (1-3) – store features, no score
    # ------------------------------------------------------------------
    if day_number <= 3:
        _prog(100, 'Analyse abgeschlossen.')
        daily_score_obj = compute_daily_score(
            day=day_number,
            final_status='stable',
            quality_status=quality,
            D_t=0.0, M_t=0.0, A_t=0.0,
            quality_reliability=reliability,
        )
        return {
            'features': features,
            'quality': quality,
            'is_baseline': True,
            'score': None,
            'state': 'normal',
            'confidence': float(reliability),
            'dep_score': None,
            'man_score': None,
            'daily_score_obj': daily_score_obj,
        }

    # ------------------------------------------------------------------
    # Step 4: Build full feature + quality lists (history + today)
    # ------------------------------------------------------------------
    _prog(50, 'Baseline wird berechnet...')

    all_features = list(previous_features) + [features]
    all_qualities = list(previous_qualities) + [quality]

    # ------------------------------------------------------------------
    # Step 5: Compute personal mu / sigma from days 1-3
    # ------------------------------------------------------------------
    # Collect the first 3 non-rejected feature dicts as personal baseline
    baseline_feats: list[dict] = []
    for f in all_features:
        if f is not None:
            baseline_feats.append(f)
        if len(baseline_feats) == 3:
            break

    if len(baseline_feats) == 0:
        # No valid baseline at all – abstain
        _prog(100, 'Analyse abgeschlossen.')
        daily_score_obj = compute_daily_score(
            day=day_number,
            final_status='abstain_insufficient_baseline',
            quality_status=quality,
            D_t=0.0, M_t=0.0, A_t=0.0,
            quality_reliability=reliability,
        )
        return {
            'features': features,
            'quality': quality,
            'is_baseline': False,
            'score': None,
            'state': 'reject',
            'confidence': 0.0,
            'dep_score': None,
            'man_score': None,
            'daily_score_obj': daily_score_obj,
        }

    mu, sigma = _compute_personal_mu_sigma(baseline_feats)
    threshold_dep, threshold_man = _compute_personal_thresholds(baseline_feats, mu, sigma)

    reference = {
        'mu': mu,
        'sigma': sigma,
        'threshold_dep': threshold_dep,
        'threshold_man': threshold_man,
    }

    # ------------------------------------------------------------------
    # Step 6: Run full time-series classification
    # ------------------------------------------------------------------
    _prog(70, 'Klassifizierung läuft...')

    daily_results = classify_patient_timeseries(all_features, reference, all_qualities)

    # The last entry is today
    today = daily_results[-1]

    label_smoothed: str = today.get('label_smoothed', today.get('label', 'unclear'))
    dep_smooth: float = today.get('dep_score_smooth', 0.0)
    man_smooth: float = today.get('man_score_smooth', 0.0)
    confidence: float = float(today.get('confidence', 0.0))

    # Magnitude of change vector (L2 norm of z-score deviations)
    z_today = compute_z_scores(features, mu, sigma)
    A_t = float(np.sqrt(sum(v ** 2 for v in z_today.values())))

    # Map pipeline label to scoring.py final_status
    final_status = _LABEL_TO_STATUS.get(label_smoothed, 'changed but unclear')

    # ------------------------------------------------------------------
    # Step 7: Compute mood score (-100..+100)
    # ------------------------------------------------------------------
    _prog(90, 'Score wird berechnet...')

    daily_score_obj = compute_daily_score(
        day=day_number,
        final_status=final_status,
        quality_status=quality,
        D_t=dep_smooth,
        M_t=man_smooth,
        A_t=A_t,
        quality_reliability=reliability,
    )

    _prog(100, 'Analyse abgeschlossen.')

    score_value = daily_score_obj.score  # float or None

    return {
        'features': features,
        'quality': quality,
        'is_baseline': False,
        'score': score_value,
        'state': label_smoothed,
        'confidence': confidence,
        'dep_score': dep_smooth,
        'man_score': man_smooth,
        'daily_score_obj': daily_score_obj,
    }
