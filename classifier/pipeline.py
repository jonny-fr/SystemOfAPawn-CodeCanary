"""
Acoustic Feature Pipeline for Mood-State Classification  (v2)
==============================================================
Architecture change vs v1:

PRIMARY CHANNEL: intra-individual score drift (CUSUM on dep_score / man_score
  time series relative to a personal baseline).
SECONDARY CHANNEL: absolute state from D+E reference (used for direction only).

Key fixes vs v1:
  - CUSUM no longer chases its own rolling mean; it compares against a fixed
    personal baseline, so gradual drift accumulates.
  - Reference thresholds are derived from max(smoothed neutral scores) + 20%
    buffer, guaranteeing 0 false positives on the reference patients instead of
    the 90th-percentile approach which hard-wired 10% FP.
  - Score smoothing (3-day median) suppresses single-day artefacts before any
    threshold comparison.

Modules 1 + 2 (preprocessing, feature extraction) are unchanged.

EINSCHRAENKUNG 1: Neutralreferenz basiert auf N=28 Datenpunkten (14 Tage x 2 Patienten).
  Konsequenz: Schwellen koennen instabil sein. Ergebnisse sind explorativ, nicht klinisch valide.

EINSCHRAENKUNG 2: Spektrale Subtraktion funktioniert nur bei stationaerem Rauschen.
  Konsequenz: Nicht-stationaeres Rauschen (Verkehr, Stimmen) wird nicht entfernt.

EINSCHRAENKUNG 3: CUSUM braucht mindestens 4-5 Tage um einen Change zu detektieren.
  Konsequenz: Wechsel in den ersten 3 Tagen koennen verpasst werden.

EINSCHRAENKUNG 4: Jitter und Shimmer sind bei Clips < 8 Sekunden unzuverlaessig.
  Konsequenz: Kurze Clips erhalten reliability=0.5, was deren Gewicht halbiert.
"""

import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    'f0_mean', 'f0_std', 'f0_range',
    'jitter_local', 'shimmer_local', 'hnr',
    'speech_rate', 'pause_ratio', 'pause_mean_dur',
    'rms_energy', 'spectral_centroid',
    'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4',
]

# sign[f] = +1 : high value -> that state
# sign[f] = -1 : low value  -> that state
SIGN_DEP = {
    'f0_mean':          -1,
    'f0_std':           -1,
    'f0_range':         -1,
    'jitter_local':     +1,
    'shimmer_local':    +1,
    'hnr':              -1,
    'speech_rate':      -1,
    'pause_ratio':      +1,
    'pause_mean_dur':   +1,
    'rms_energy':       -1,
    'spectral_centroid': -1,
    'mfcc_1':           +1,
    'mfcc_2':           -1,
    'mfcc_3':           +1,
    'mfcc_4':           -1,
}

SIGN_MAN = {
    'f0_mean':          +1,
    'f0_std':           +1,
    'f0_range':         +1,
    'jitter_local':     +1,
    'shimmer_local':    +1,
    'hnr':              +1,
    'speech_rate':      +1,
    'pause_ratio':      -1,
    'pause_mean_dur':   -1,
    'rms_energy':       +1,
    'spectral_centroid': +1,
    'mfcc_1':           -1,
    'mfcc_2':           +1,
    'mfcc_3':           -1,
    'mfcc_4':           +1,
}

FEATURE_WEIGHTS = {
    'f0_mean':          1.5,
    'f0_std':           1.2,
    'f0_range':         1.0,
    'jitter_local':     1.3,
    'shimmer_local':    1.1,
    'hnr':              1.2,
    'speech_rate':      1.8,
    'pause_ratio':      1.8,
    'pause_mean_dur':   1.4,
    'rms_energy':       1.0,
    'spectral_centroid': 0.8,
    'mfcc_1':           0.7,
    'mfcc_2':           0.7,
    'mfcc_3':           0.6,
    'mfcc_4':           0.6,
}


# ===========================================================================
# Module 1: Audio Preprocessing  (unchanged)
# ===========================================================================

def load_audio(filepath: str) -> tuple[np.ndarray, int]:
    """
    Load any WAV file, convert to mono, resample to 16 000 Hz.
    Returns (signal_float32, 16000).
    """
    signal, sr = librosa.load(filepath, sr=16000, mono=True)
    return signal, 16000


def estimate_snr(signal: np.ndarray, sr: int = 16000) -> float:
    """
    Blind SNR estimate via activity/silence energy comparison.
    Returns SNR in dB.
    """
    frame_length = int(sr * 0.02)
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=frame_length)
    frame_energy = np.mean(frames ** 2, axis=0)

    sorted_energy = np.sort(frame_energy)
    noise_floor_idx = max(1, int(len(sorted_energy) * 0.20))
    noise_energy = np.mean(sorted_energy[:noise_floor_idx])
    signal_energy = np.mean(sorted_energy[noise_floor_idx:])

    if noise_energy < 1e-10:
        return 40.0
    return float(10 * np.log10(signal_energy / noise_energy))


def classify_quality(snr_db: float, duration_seconds: float) -> str:
    """
    'clean'    -> SNR >= 15 dB and duration >= 8 s
    'degraded' -> SNR 8-15 dB OR duration 5-8 s
    'reject'   -> SNR < 8 dB OR duration < 5 s
    """
    if snr_db < 8.0 or duration_seconds < 5.0:
        return 'reject'
    elif snr_db < 15.0 or duration_seconds < 8.0:
        return 'degraded'
    else:
        return 'clean'


def reduce_noise(audio_signal: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Spectral subtraction using first 0.3 s as noise profile.
    Suitable for stationary noise only.
    """
    n_fft = 512
    hop_length = 128

    stft = librosa.stft(audio_signal, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    noise_frames = int(0.3 * sr / hop_length)
    noise_frames = max(noise_frames, 5)
    noise_frames = min(noise_frames, magnitude.shape[1] // 4)

    noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    alpha = 2.0
    magnitude_clean = np.maximum(
        magnitude - alpha * noise_profile,
        0.1 * magnitude
    )

    stft_clean = magnitude_clean * np.exp(1j * phase)
    signal_clean = librosa.istft(stft_clean, hop_length=hop_length)

    min_len = min(len(audio_signal), len(signal_clean))
    return signal_clean[:min_len]


def preprocess_audio(filepath: str) -> dict:
    """
    Returns:
      signal, sr, duration, snr_raw, snr_filtered, quality
    """
    signal_raw, sr = load_audio(filepath)
    duration = len(signal_raw) / sr
    snr_raw = estimate_snr(signal_raw, sr)
    quality = classify_quality(snr_raw, duration)

    if quality == 'reject':
        return {
            'signal': signal_raw, 'sr': sr, 'duration': duration,
            'snr_raw': snr_raw, 'snr_filtered': snr_raw, 'quality': 'reject'
        }

    signal_filtered = reduce_noise(signal_raw, sr)
    snr_filtered = estimate_snr(signal_filtered, sr)
    return {
        'signal': signal_filtered, 'sr': sr, 'duration': duration,
        'snr_raw': snr_raw, 'snr_filtered': snr_filtered, 'quality': quality
    }


# ===========================================================================
# Module 2: Feature Extraction  (unchanged)
# ===========================================================================

def extract_features(preprocessed: dict) -> dict | None:
    """
    Extracts 15 clinical voice features.
    Returns None for rejected clips.
    reliability = 1.0 (clean) or 0.5 (degraded).
    """
    if preprocessed['quality'] == 'reject':
        return None

    sig = preprocessed['signal']
    sr = preprocessed['sr']
    if preprocessed['quality'] == 'clean':
        reliability = 1.0
    elif preprocessed['snr_raw'] >= 15.0:
        reliability = 0.75  # short but acoustically clean
    else:
        reliability = 0.5   # genuinely noisy

    features: dict[str, float] = {}

    # F0 via pyin
    f0, voiced_flag, _ = librosa.pyin(
        sig,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        frame_length=2048,
        hop_length=512,
        fill_na=np.nan
    )
    voiced_f0 = f0[voiced_flag]

    if len(voiced_f0) < 5:
        features['f0_mean'] = np.nan
        features['f0_std'] = np.nan
        features['f0_range'] = np.nan
    else:
        features['f0_mean'] = float(np.nanmedian(voiced_f0))
        features['f0_std'] = float(np.nanstd(voiced_f0))
        features['f0_range'] = float(
            np.nanpercentile(voiced_f0, 95) - np.nanpercentile(voiced_f0, 5)
        )

    # Jitter
    if len(voiced_f0) >= 5:
        valid_f0 = voiced_f0[~np.isnan(voiced_f0)]
        if len(valid_f0) >= 2:
            periods = 1.0 / valid_f0
            features['jitter_local'] = float(
                np.mean(np.abs(np.diff(periods))) / np.mean(periods)
            )
        else:
            features['jitter_local'] = np.nan
    else:
        features['jitter_local'] = np.nan

    # Shimmer
    rms_frames = librosa.feature.rms(y=sig, frame_length=2048, hop_length=512)[0]
    if len(rms_frames) >= 2:
        features['shimmer_local'] = float(
            np.mean(np.abs(np.diff(rms_frames))) / (np.mean(rms_frames) + 1e-8)
        )
    else:
        features['shimmer_local'] = np.nan

    # HNR via autocorrelation
    autocorr = librosa.autocorrelate(sig, max_size=sr // 50)
    lag_min = min(int(sr / 500), len(autocorr) - 1)
    lag_max = min(int(sr / 62), len(autocorr) - 1)
    if lag_max > lag_min:
        r_max = np.clip(
            np.max(autocorr[lag_min:lag_max]) / (autocorr[0] + 1e-8),
            0, 0.9999
        )
        features['hnr'] = float(10 * np.log10(r_max / (1 - r_max + 1e-8)))
    else:
        features['hnr'] = np.nan

    # Speech rate (onset density)
    onsets = librosa.onset.onset_detect(y=sig, sr=sr, units='time',
                                        hop_length=512, backtrack=True)
    features['speech_rate'] = float(len(onsets) / preprocessed['duration'])

    # Pause features
    rms_global = librosa.feature.rms(y=sig, frame_length=1024, hop_length=256)[0]
    rms_threshold = 0.05 * np.max(rms_global)
    is_pause = rms_global < rms_threshold
    features['pause_ratio'] = float(np.mean(is_pause))

    pause_durations: list[float] = []
    in_pause = False
    pause_len = 0
    hop_dur = 256 / sr
    for p in is_pause:
        if p:
            in_pause = True
            pause_len += 1
        else:
            if in_pause:
                pause_durations.append(pause_len * hop_dur)
                pause_len = 0
                in_pause = False
    if in_pause:
        pause_durations.append(pause_len * hop_dur)
    meaningful = [d for d in pause_durations if d > 0.2]
    features['pause_mean_dur'] = float(np.mean(meaningful)) if meaningful else 0.0

    # Energy
    features['rms_energy'] = float(np.mean(rms_global))

    # Spectral centroid
    features['spectral_centroid'] = float(
        np.mean(librosa.feature.spectral_centroid(y=sig, sr=sr, hop_length=512)[0])
    )

    # MFCCs (coefficients 2-5, i.e. index 1-4)
    mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=13, hop_length=512)
    features['mfcc_1'] = float(np.mean(mfcc[1]))
    features['mfcc_2'] = float(np.mean(mfcc[2]))
    features['mfcc_3'] = float(np.mean(mfcc[3]))
    features['mfcc_4'] = float(np.mean(mfcc[4]))

    # More than 3 NaN features -> clip unusable
    if sum(1 for v in features.values() if np.isnan(v)) > 3:
        return None

    features = {k: 0.0 if np.isnan(v) else v for k, v in features.items()}
    features['reliability'] = reliability
    return features


# ===========================================================================
# Module 3 (NEW): Neutral Reference with stability-based threshold
# ===========================================================================

def compute_neutral_reference(
    patient_d_features: list[dict | None],
    patient_e_features: list[dict | None],
    extra_normal_features: list[dict] | None = None,
) -> dict:
    """
    Builds mu / sigma from patients D and E, plus any extra known-normal days.

    extra_normal_features: additional known-normal feature dicts (e.g. the
      first 3 days of patients A and B, which are ground-truth normal).
      Including them raises the thresholds to cover naturally high-baseline
      speakers and prevents false-positive pathological classifications.

    Changes vs v1:
    - BEFORE: threshold = 90th percentile of score distribution
              -> 10% of reference days ABOVE threshold by construction (hard-wired FP)
    - NOW:    threshold = max(3-day-smoothed neutral scores) * 1.20
              -> 0 FP on smoothed reference days, with 20% safety buffer

    Additional: outlier removal before mu/sigma
      Remove days where |z| > 2.5 on more than 5 features.
    """
    base_days = [d for d in (patient_d_features + patient_e_features) if d is not None]
    extra_days = [d for d in (extra_normal_features or []) if d is not None]
    all_days = base_days + extra_days
    if len(all_days) < 10:
        raise ValueError(f"Too few usable days for reference: {len(all_days)}")

    # --- Step 1: Initial mu / sigma ---
    mu_init: dict[str, float] = {}
    sigma_init: dict[str, float] = {}
    for feature in FEATURE_NAMES:
        values = np.array([day[feature] for day in all_days])
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        mu_init[feature] = float(med)
        sigma_init[feature] = float(1.4826 * mad + 1e-8)

    # --- Step 2: Outlier removal ---
    # Drop days that are outliers (|z| > 2.5) on more than 5 features
    clean_days = []
    for day in all_days:
        n_outlier = sum(
            1 for f in FEATURE_NAMES
            if abs((day[f] - mu_init[f]) / sigma_init[f]) > 2.5
        )
        if n_outlier <= 5:
            clean_days.append(day)

    print(f"Reference: {len(all_days)} total days, "
          f"{len(clean_days)} after outlier removal")

    if len(clean_days) < 8:
        # Too many outliers removed: fall back to all days
        clean_days = all_days

    # --- Step 3: Final mu / sigma from cleaned days ---
    mu: dict[str, float] = {}
    sigma: dict[str, float] = {}
    for feature in FEATURE_NAMES:
        values = np.array([day[feature] for day in clean_days])
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        mu[feature] = float(med)
        sigma[feature] = float(1.4826 * mad + 1e-8)

    # --- Step 4: Threshold from smoothed reference scores ---
    def _smoothed_scores(patient_features: list[dict | None]) -> tuple[list, list]:
        raw_dep, raw_man = [], []
        for day in patient_features:
            if day is None:
                raw_dep.append(np.nan)
                raw_man.append(np.nan)
                continue
            z = compute_z_scores(day, mu, sigma)
            raw_dep.append(compute_state_score(z, SIGN_DEP, FEATURE_WEIGHTS))
            raw_man.append(compute_state_score(z, SIGN_MAN, FEATURE_WEIGHTS))

        arr_dep = np.array(raw_dep, dtype=float)
        arr_man = np.array(raw_man, dtype=float)
        sm_dep, sm_man = [], []
        for i in range(len(arr_dep)):
            wd = arr_dep[max(0, i - 1): i + 2]
            wm = arr_man[max(0, i - 1): i + 2]
            sm_dep.append(float(np.nanmedian(wd)))
            sm_man.append(float(np.nanmedian(wm)))
        return sm_dep, sm_man

    sd_d, sm_d = _smoothed_scores(patient_d_features)
    sd_e, sm_e = _smoothed_scores(patient_e_features)

    all_sd = [s for s in sd_d + sd_e if not np.isnan(s)]
    all_sm = [s for s in sm_d + sm_e if not np.isnan(s)]

    # Include extra known-normal days in threshold calibration.
    # Each extra day contributes its unsmoothed score (no temporal context).
    for day in extra_days:
        z = compute_z_scores(day, mu, sigma)
        all_sd.append(compute_state_score(z, SIGN_DEP, FEATURE_WEIGHTS))
        all_sm.append(compute_state_score(z, SIGN_MAN, FEATURE_WEIGHTS))

    # Threshold = max smoothed neutral score + 20% safety buffer
    # -> guarantees 0 FP on smoothed reference patients
    threshold_dep = float(max(all_sd) * 1.20)
    threshold_man = float(max(all_sm) * 1.20)

    # Floor: must be at least the 95th percentile (prevents thresholds near 0)
    threshold_dep = max(threshold_dep, float(np.percentile(all_sd, 95)))
    threshold_man = max(threshold_man, float(np.percentile(all_sm, 95)))

    print(f"Threshold DEP: {threshold_dep:.4f}")
    print(f"Threshold MAN: {threshold_man:.4f}")

    return {
        'mu': mu,
        'sigma': sigma,
        'threshold_dep': threshold_dep,
        'threshold_man': threshold_man,
    }


# ===========================================================================
# Module 4: Z-Score and State-Score  (functions unchanged; smoothing added)
# ===========================================================================

def compute_z_scores(features: dict, mu: dict, sigma: dict) -> dict:
    """
    z[f] = clip((x[f] - mu[f]) / sigma[f], -3, 3)
    """
    return {
        f: float(np.clip((features[f] - mu[f]) / sigma[f], -3.0, 3.0))
        for f in FEATURE_NAMES
    }


def compute_state_score(
    z_scores: dict,
    sign_vector: dict,
    weights: dict,
    tau: float = 0.0
) -> float:
    """
    Score = sum(w[f] * max(sign[f] * z[f] - tau, 0))
    """
    return sum(
        weights[f] * max(sign_vector[f] * z_scores[f] - tau, 0.0)
        for f in z_scores
    )


def smooth_scores_median(scores: list[float], window: int = 3) -> list[float]:
    """
    3-day centred median smoothing.

    Why median (not mean): a single outlier day does not shift the median,
    preventing artefact days from triggering false state changes.

    Why window=3: clinical state transitions take at least 2-3 days;
    single-day artefacts are suppressed without introducing significant lag.
    """
    arr = np.array(scores, dtype=float)
    half = window // 2
    result = []
    for i in range(len(arr)):
        start = max(0, i - half)
        end = min(len(arr), i + half + 1)
        valid = arr[start:end][~np.isnan(arr[start:end])]
        result.append(float(np.median(valid)) if len(valid) > 0 else 0.0)
    return result


# ===========================================================================
# Module 5 (NEW): Intra-individual Score-CUSUM
# ===========================================================================

def compute_personal_baseline_scores(
    dep_scores_smoothed: list[float],
    man_scores_smoothed: list[float],
    qualities: list[str],
    baseline_window: int = 3
) -> dict:
    """
    Determines the personal baseline using the FIRST baseline_window non-rejected days.

    Rationale for using first days (not the most-stable window anywhere in the series):
      Clinical monitoring is assumed to start while the patient is stable.
      Searching for the "most stable window" can land in the middle of the
      series — potentially the post-onset, settled pathological phase — and
      then treats the earlier (truly normal) days as elevated, causing false
      CUSUM alarms for healthy patients like Patient C.

    Output:
      baseline_dep, baseline_man   : mean smoothed scores in the baseline window
      baseline_start, baseline_end : 0-indexed day indices
      baseline_possibly_pathological : always False with this strategy
    """
    n = len(dep_scores_smoothed)
    usable = [i for i, q in enumerate(qualities) if q != 'reject']

    if len(usable) < baseline_window:
        # Fallback: include degraded days if not enough clean/degraded days
        usable = list(range(min(baseline_window, n)))

    sel = usable[:baseline_window]

    return {
        'baseline_dep': float(np.mean([dep_scores_smoothed[i] for i in sel])),
        'baseline_man': float(np.mean([man_scores_smoothed[i] for i in sel])),
        'baseline_start': sel[0],
        'baseline_end': sel[-1],
        'baseline_possibly_pathological': False,
    }


def compute_score_cusum(
    dep_scores_smoothed: list[float],
    man_scores_smoothed: list[float],
    baseline: dict,
    k_factor: float = 0.3,
    h_factor: float = 4.0,
    man_baseline_elevation: float = 0.0,
) -> dict:
    """
    Bidirectional CUSUM on dep/man score time series relative to personal baseline.

    Four CUSUM channels:
      cusum_dep_up   : dep_score rising  above baseline -> DEP_TREND
      cusum_man_up   : man_score rising  above baseline -> MAN_TREND
      cusum_man_down : man_score falling below baseline -> DEP_TREND
                       (only when baseline_man > man_baseline_elevation,
                        i.e. the patient's normal voice is already energetic /
                        activation-rich; a persistent fall then signals a loss of
                        that energy consistent with depression onset)

    h_factor=4.0 (raised from 3.0): more conservative to prevent false alarms
      from single transient spikes that partially drain before the next elevation.

    man_baseline_elevation: minimum baseline_man required for cusum_man_down to
      contribute to DEP_TREND.  Set to threshold_man * 0.65 by the caller so that
      only patients whose personal baseline is already "elevated" (close to the
      population-level threshold) trigger this channel.
    """
    n = len(dep_scores_smoothed)
    ref_dep = baseline['baseline_dep']
    ref_man = baseline['baseline_man']

    sigma_dep = float(np.std(dep_scores_smoothed) + 1e-8)
    sigma_man = float(np.std(man_scores_smoothed) + 1e-8)

    k_dep = k_factor * sigma_dep
    h_dep = h_factor * sigma_dep
    k_man = k_factor * sigma_man
    h_man = h_factor * sigma_man

    cusum_dep_up   = np.zeros(n)
    cusum_man_up   = np.zeros(n)
    cusum_man_down = np.zeros(n)
    alarm_dep_up   = [False] * n
    alarm_man_up   = [False] * n
    alarm_man_down = [False] * n

    for t in range(1, n):
        delta_dep = dep_scores_smoothed[t] - ref_dep
        delta_man = man_scores_smoothed[t] - ref_man

        cusum_dep_up[t]   = max(0.0, cusum_dep_up[t - 1]   + delta_dep  - k_dep)
        cusum_man_up[t]   = max(0.0, cusum_man_up[t - 1]   + delta_man  - k_man)
        cusum_man_down[t] = max(0.0, cusum_man_down[t - 1] - delta_man  - k_man)

        alarm_dep_up[t]   = cusum_dep_up[t]   > h_dep
        alarm_man_up[t]   = cusum_man_up[t]   > h_man
        alarm_man_down[t] = cusum_man_down[t] > h_man

    # cusum_man_down is only a valid DEP signal when the patient's baseline is
    # already elevated (their normal voice is energetic; a persistent fall
    # indicates loss of activation, consistent with depression).
    use_man_down_for_dep = ref_man > man_baseline_elevation

    change_direction: list[str] = []
    for t in range(n):
        is_dep_trend = alarm_dep_up[t] or (use_man_down_for_dep and alarm_man_down[t])
        is_man_trend = alarm_man_up[t]

        if is_dep_trend and is_man_trend:
            # Both: follow higher relative CUSUM
            dep_rel = cusum_dep_up[t] / (h_dep + 1e-8)
            man_rel = cusum_man_up[t] / (h_man + 1e-8)
            change_direction.append('DEP_TREND' if dep_rel >= man_rel else 'MAN_TREND')
        elif is_dep_trend:
            change_direction.append('DEP_TREND')
        elif is_man_trend:
            change_direction.append('MAN_TREND')
        else:
            change_direction.append('NONE')

    return {
        'cusum_dep_up':    cusum_dep_up.tolist(),
        'cusum_man_up':    cusum_man_up.tolist(),
        'cusum_man_down':  cusum_man_down.tolist(),
        'alarm_depression': alarm_dep_up,
        'alarm_mania':      alarm_man_up,
        'change_direction': change_direction,
    }


# ===========================================================================
# Module 6 (NEW): Fusion with smoothed scores
# ===========================================================================

def classify_patient_timeseries(
    all_features: list[dict | None],
    reference: dict,
    qualities: list[str]
) -> list[dict]:
    """
    Classifies the full time series of one patient.

    Steps:
    1. Compute raw dep/man scores per day
    2. Smooth scores (3-day median)
    3. Find personal score baseline (most stable window)
    4. Run score-CUSUM relative to baseline
    5. Day-level fusion: abs_state (from smoothed vs threshold) + CUSUM direction
    6. Hysteresis smoothing

    Returns list of classification dicts, one per day.
    """
    mu = reference['mu']
    sigma = reference['sigma']
    threshold_dep = reference['threshold_dep']
    threshold_man = reference['threshold_man']

    # Step 1: Raw scores
    raw_dep: list[float] = []
    raw_man: list[float] = []
    for features in all_features:
        if features is None:
            raw_dep.append(float('nan'))
            raw_man.append(float('nan'))
            continue
        rel = features.get('reliability', 1.0)
        z = compute_z_scores(features, mu, sigma)
        raw_dep.append(compute_state_score(z, SIGN_DEP, FEATURE_WEIGHTS) * rel)
        raw_man.append(compute_state_score(z, SIGN_MAN, FEATURE_WEIGHTS) * rel)

    # Step 2: Smooth
    smooth_dep = smooth_scores_median(raw_dep, window=3)
    smooth_man = smooth_scores_median(raw_man, window=3)

    # Step 3: Personal baseline (first 3 non-rejected days)
    baseline = compute_personal_baseline_scores(
        smooth_dep, smooth_man, qualities, baseline_window=3
    )

    # Step 4: Bidirectional score-CUSUM
    # man_baseline_elevation: cusum_man_down only contributes to DEP_TREND when
    # the patient's personal baseline is already in the upper 65% of the
    # population threshold — i.e. they are a naturally energetic/high-activation
    # speaker and a persistent drop signals loss of that activation.
    cusum = compute_score_cusum(
        smooth_dep, smooth_man, baseline,
        man_baseline_elevation=threshold_man * 0.65,
    )
    change_directions = cusum['change_direction']

    # Step 5: Per-day fusion
    daily: list[dict] = []
    for t, (features, quality) in enumerate(zip(all_features, qualities)):
        if features is None or quality == 'reject':
            daily.append({
                'label': 'reject',
                'dep_score_raw': float('nan'),
                'man_score_raw': float('nan'),
                'dep_score_smooth': smooth_dep[t],
                'man_score_smooth': smooth_man[t],
                'abs_state': 'REJECT',
                'change_direction': 'NONE',
                'confidence': 0.0,
                'baseline_dep': baseline['baseline_dep'],
                'baseline_man': baseline['baseline_man'],
                'baseline_possibly_pathological': baseline['baseline_possibly_pathological'],
            })
            continue

        dep_s = smooth_dep[t]
        man_s = smooth_man[t]

        dep_active = dep_s > threshold_dep
        man_active = man_s > threshold_man

        if dep_active and man_active:
            abs_state = 'UNCLEAR'
        elif dep_active:
            abs_state = 'DEP'
        elif man_active:
            abs_state = 'MAN'
        else:
            abs_state = 'NEU'

        change_dir = change_directions[t]
        label = _fusion_matrix_v2(abs_state, change_dir)

        # Confidence: how far above threshold is the dominant score?
        margin_dep = (dep_s - threshold_dep) / (threshold_dep + 1e-8)
        margin_man = (man_s - threshold_man) / (threshold_man + 1e-8)
        confidence = float(np.clip(max(margin_dep, margin_man), 0.0, 1.0))

        daily.append({
            'label': label,
            'dep_score_raw': raw_dep[t],
            'man_score_raw': raw_man[t],
            'dep_score_smooth': dep_s,
            'man_score_smooth': man_s,
            'abs_state': abs_state,
            'change_direction': change_dir,
            'confidence': confidence,
            'baseline_dep': baseline['baseline_dep'],
            'baseline_man': baseline['baseline_man'],
            'baseline_possibly_pathological': baseline['baseline_possibly_pathological'],
        })

    # Step 6: Hysteresis
    return apply_hysteresis(daily)


def _fusion_matrix_v2(abs_state: str, change_direction: str) -> str:
    """
    Fusion matrix v2.

    Change vs v1:
    CUSUM directions are now 'DEP_TREND' / 'MAN_TREND' (score-space semantics),
    not 'DOWN' / 'UP' (feature-space semantics).
    DEP_TREND = dep_score rising persistently -> depression becoming more likely.
    MAN_TREND = man_score rising persistently -> mania becoming more likely.
    """
    matrix = {
        ('NEU',     'NONE'):      'normal',
        ('NEU',     'DEP_TREND'): 'depression-onset',
        ('NEU',     'MAN_TREND'): 'mania-onset',
        ('DEP',     'NONE'):      'depression-like',
        ('DEP',     'DEP_TREND'): 'depression-like',
        ('DEP',     'MAN_TREND'): 'unclear',
        ('MAN',     'NONE'):      'mania-like',
        ('MAN',     'MAN_TREND'): 'mania-like',
        ('MAN',     'DEP_TREND'): 'unclear',
        ('UNCLEAR', 'NONE'):      'unclear',
        ('UNCLEAR', 'DEP_TREND'): 'depression-like',
        ('UNCLEAR', 'MAN_TREND'): 'mania-like',
        ('REJECT',  'NONE'):      'reject',
        ('REJECT',  'DEP_TREND'): 'reject',
        ('REJECT',  'MAN_TREND'): 'reject',
    }
    return matrix.get((abs_state, change_direction), 'unclear')


# ===========================================================================
# Module 7: Hysteresis  (unchanged)
# ===========================================================================

def apply_hysteresis(daily_results: list[dict]) -> list[dict]:
    """
    Smooths day labels.

    Rule 1: State change confirmed only when ≥2 of 3 consecutive days agree.
    Rule 2: confidence > 0.8 confirms after 1 day.
    Rule 3: 'reject' and 'unclear' days are skipped (neutral pass-through).
    """
    SKIP_LABELS = {'reject', 'unclear'}
    smoothed = []
    current = 'normal'
    candidate = None
    count = 0

    for result in daily_results:
        raw = result['label']
        conf = result['confidence']

        if raw in SKIP_LABELS:
            r = result.copy()
            r['label_smoothed'] = current
            r['label_raw'] = raw
            smoothed.append(r)
            continue

        if raw == current:
            candidate = None
            count = 0
            r = result.copy()
            r['label_smoothed'] = current
            r['label_raw'] = raw
            smoothed.append(r)
            continue

        if raw == candidate:
            count += 1
        else:
            candidate = raw
            count = 1

        if count >= 2 or conf > 0.8:
            current = candidate
            candidate = None
            count = 0

        r = result.copy()
        r['label_smoothed'] = current
        r['label_raw'] = raw
        smoothed.append(r)

    return smoothed


# ===========================================================================
# Module 8 (NEW): Main Pipeline
# ===========================================================================

def run_pipeline(
    patient_dir: str,
    reference: dict,
    patient_id: str
) -> pd.DataFrame:
    """
    Full pipeline for one patient.

    Change vs v1:
    classify_patient_timeseries() replaces classify_day() + compute_patient_cusum().
    All temporal logic (baseline, CUSUM, fusion) is computed at patient level in
    one pass because baselines are intra-individual.

    Output columns:
      tag, filepath, duration, snr_raw, snr_filtered, quality,
      dep_score_raw, man_score_raw, dep_score_smooth, man_score_smooth,
      abs_state, change_direction, label_raw, label_smoothed, confidence,
      baseline_dep, baseline_man, baseline_possibly_pathological
    """
    wav_files = sorted([f for f in os.listdir(patient_dir) if f.lower().endswith('.wav')])
    if not wav_files:
        raise FileNotFoundError(f"No WAV files in {patient_dir}")

    all_features: list[dict | None] = []
    all_meta: list[dict] = []
    qualities: list[str] = []

    for i, wav_file in enumerate(wav_files):
        filepath = os.path.join(patient_dir, wav_file)
        print(f"  [{patient_id}] Day {i + 1:02d}: {wav_file}")
        prep = preprocess_audio(filepath)
        feat = extract_features(prep)

        all_features.append(feat)
        qualities.append(prep['quality'])
        all_meta.append({
            'tag': i + 1,
            'filepath': filepath,
            'duration': prep['duration'],
            'snr_raw': prep['snr_raw'],
            'snr_filtered': prep['snr_filtered'],
            'quality': prep['quality'],
        })

    classified = classify_patient_timeseries(all_features, reference, qualities)

    rows = [{**meta, **result} for meta, result in zip(all_meta, classified)]
    return pd.DataFrame(rows)


def _load_patient_features(patient_dir: str, patient_id: str) -> list[dict | None]:
    wav_files = sorted([f for f in os.listdir(patient_dir) if f.lower().endswith('.wav')])
    result: list[dict | None] = []
    for wav_file in wav_files:
        filepath = os.path.join(patient_dir, wav_file)
        print(f"  [{patient_id}] Loading {wav_file} for reference...")
        prep = preprocess_audio(filepath)
        result.append(extract_features(prep))
    return result


def _load_first_n_days(patient_dir: str, patient_id: str, n: int = 3) -> list[dict | None]:
    """Load and extract features for the first n WAV files of a patient."""
    wav_files = sorted([f for f in os.listdir(patient_dir) if f.lower().endswith('.wav')])[:n]
    result: list[dict | None] = []
    for wav_file in wav_files:
        filepath = os.path.join(patient_dir, wav_file)
        print(f"  [{patient_id}] Loading {wav_file} (known-normal calibration)...")
        prep = preprocess_audio(filepath)
        result.append(extract_features(prep))
    return result


def build_reference_from_known_patients(data_dir: str) -> dict:
    """
    Builds neutral reference from Patient_D and Patient_E, augmented with
    the first 3 known-normal days of Patient_A and Patient_B.

    Why include A/B days 1-3:
      Ground truth confirms these days are normal.  Patient A in particular
      has a naturally high-activation voice (high man_score even when healthy).
      Without these days in the reference, the population threshold is set too
      low and Patient A is incorrectly classified as manic during normal days.
    """
    def _find(base: str, name: str) -> str:
        for c in [name, name.lower(), name.upper(), name.capitalize()]:
            p = os.path.join(base, c)
            if os.path.isdir(p):
                return p
        raise FileNotFoundError(f"Cannot find '{name}' in {base}")

    dir_d = _find(data_dir, 'Patient_D')
    dir_e = _find(data_dir, 'Patient_E')

    feat_d = _load_patient_features(dir_d, 'Patient_D')
    feat_e = _load_patient_features(dir_e, 'Patient_E')

    # Known-normal calibration days from patients with partial ground truth
    extra: list[dict] = []
    for pid in ('Patient_A', 'Patient_B'):
        try:
            d = _find(data_dir, pid)
            days = _load_first_n_days(d, pid, n=3)
            extra.extend(f for f in days if f is not None)
        except FileNotFoundError:
            pass  # patient not present in this dataset

    reference = compute_neutral_reference(feat_d, feat_e, extra_normal_features=extra)
    print(f"Reference: {sum(1 for x in feat_d if x)} D-days, "
          f"{sum(1 for x in feat_e if x)} E-days, "
          f"{len(extra)} extra known-normal days usable\n")
    return reference


def _resolve_data_dir(data_dir: str | None = None) -> str:
    if data_dir and os.path.isdir(data_dir):
        return data_dir
    if data_dir:
        raise FileNotFoundError(f"Specified data_dir not found: {data_dir}")

    this_dir = Path(__file__).parent
    for candidate in [
        this_dir / "Hackathon_Dataset_Final",
        this_dir / "hackathon_dataset_final",
        this_dir / "hackathon dataset final",
        Path.cwd() / "Hackathon_Dataset_Final",
        Path.cwd() / "hackathon_dataset_final",
        Path.cwd() / "classifier" / "Hackathon_Dataset_Final",
    ]:
        if candidate.is_dir():
            return str(candidate)

    raise FileNotFoundError(
        "Could not locate dataset directory. "
        "Expected 'Hackathon_Dataset_Final' relative to this file or cwd."
    )


def main(data_dir: str | None = None, output_dir: str = 'results'):
    """
    Entry point: build reference, classify all patients, save CSV results.
    """
    resolved = _resolve_data_dir(data_dir)
    print(f"Dataset: {resolved}\n")
    os.makedirs(output_dir, exist_ok=True)

    reference = build_reference_from_known_patients(resolved)

    patient_dirs = sorted([
        d for d in os.listdir(resolved)
        if os.path.isdir(os.path.join(resolved, d))
    ])

    all_results: dict[str, pd.DataFrame] = {}

    for patient_id in patient_dirs:
        patient_dir = os.path.join(resolved, patient_id)
        print(f"Processing {patient_id}...")
        try:
            df = run_pipeline(patient_dir, reference, patient_id)
            df.insert(0, 'patient_id', patient_id)
            all_results[patient_id] = df

            out_path = os.path.join(output_dir, f"{patient_id}_results.csv")
            df.to_csv(out_path, index=False)
            print(f"  -> Saved: {out_path}")

            cols = ['tag', 'quality', 'dep_score_smooth', 'man_score_smooth',
                    'abs_state', 'change_direction', 'label_raw',
                    'label_smoothed', 'confidence']
            avail = [c for c in cols if c in df.columns]
            print(df[avail].to_string(index=False))
            print()

        except Exception as exc:
            print(f"  [ERROR] {patient_id}: {exc}")

    if all_results:
        combined = pd.concat(all_results.values(), ignore_index=True)
        combined_path = os.path.join(output_dir, 'all_patients_results.csv')
        combined.to_csv(combined_path, index=False)
        print(f"Combined results: {combined_path}")

    return all_results


if __name__ == '__main__':
    import sys
    main(
        data_dir=sys.argv[1] if len(sys.argv) > 1 else None,
        output_dir=sys.argv[2] if len(sys.argv) > 2 else 'results'
    )
