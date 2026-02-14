"""
feature_extraction.py -- Extract numerical features from raw ECG signals.

Why do we need feature extraction?
  Machine learning models like SVM and Random Forest need numerical input
  in a fixed-size format: one row per sample, one column per feature.
  But our ECG data is a 2D array per sample (1000 timesteps x 12 leads).
  We need to summarize each signal into a small set of meaningful numbers.

What features do we extract?

  A. PER-LEAD SIGNAL FEATURES (10 features x 12 leads = 120):
    TIME-DOMAIN:
      1. Mean           -- average voltage level
      2. Std deviation  -- how much the voltage varies
      3. Max            -- highest voltage peak
      4. Min            -- lowest voltage dip
      5. RMS            -- root mean square, measures signal energy
      6. Zero crossings -- how often the signal crosses the baseline
    STATISTICAL:
      7. Skewness       -- asymmetry of the value distribution
      8. Kurtosis       -- "peakedness" (high = sharp peaks)
    FREQUENCY-DOMAIN:
      9. Dominant freq   -- the strongest rhythm frequency (Hz)
     10. Spectral energy -- total power across all frequencies

  B. HRV FEATURES (9 features from Lead II only):
    TIME-DOMAIN HRV:
      11. Mean RR  -- average time between heartbeats (seconds)
      12. SDNN     -- std deviation of RR intervals (overall variability)
      13. RMSSD    -- root mean square of successive RR differences
      14. pNN50    -- % of successive RR intervals differing by >50ms
      15. HR mean  -- average heart rate (bpm)
      16. HR std   -- std deviation of heart rate
    FREQUENCY-DOMAIN HRV:
      17. LF power   -- power in 0.04-0.15 Hz band (sympathetic activity)
      18. HF power   -- power in 0.15-0.4 Hz band (parasympathetic activity)
      19. LF/HF ratio -- autonomic balance indicator

  C. MORPHOLOGICAL FEATURES (6 features x 12 leads = 72):
    Computed from averaged heartbeat templates per lead:
      20. R-peak amplitude -- height of the R-wave
      21. QRS duration     -- width of the main spike (ms)
      22. R/S ratio        -- R-peak height / S-trough depth
      23. ST deviation     -- voltage shift after QRS (heart attack indicator)
      24. T-wave amplitude -- height of the T-wave
      25. Beat shape std   -- consistency of individual beats vs template

  D. WAVELET FEATURES (6 features x 12 leads = 72):
    Discrete Wavelet Transform (DWT) with Daubechies-4 wavelet, 4 levels:
      26. Energy d3        -- signal power in detail level 3 (~6-12 Hz, QRS range)
      27. Energy d4        -- signal power in detail level 4 (~3-6 Hz, T-wave range)
      28. Energy approx    -- signal power in approximation (~0-3 Hz, baseline wander)
      29. Entropy d3       -- disorder/complexity at detail level 3
      30. Entropy d4       -- disorder/complexity at detail level 4
      31. Detail ratio     -- high-freq energy / low-freq energy balance

  E. FREQUENCY BAND ENERGY FEATURES (3 features x 12 leads = 36):
    FFT spectrum split into clinically meaningful bands:
      32. Low band energy  -- 0.5-5 Hz (P-wave, T-wave, heart rate rhythm)
      33. Mid band energy  -- 5-15 Hz (QRS complex range)
      34. High band energy -- 15-40 Hz (sharp transitions, fine details)

  Total: 120 + 9 + 72 + 72 + 36 = 309 features per ECG record.

Uses scipy for:
  - scipy.stats.skew and scipy.stats.kurtosis (statistical features)
  - scipy.fft.rfft and scipy.fft.rfftfreq (frequency-domain features)
  - scipy.signal.find_peaks (R-peak detection)
  - scipy.interpolate.interp1d (HRV frequency analysis)
Uses PyWavelets (pywt) for:
  - pywt.wavedec (Discrete Wavelet Transform decomposition)
"""

import numpy as np
import pywt
from scipy import stats
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


# =============================================================================
# CONSTANTS
# =============================================================================

# The 100Hz records have 100 samples per second
SAMPLING_RATE = 100  # Hz

# Names of the 12 standard ECG leads
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# Names of the 10 features we extract per lead
STAT_NAMES = ["mean", "std", "max", "min", "skew", "kurtosis",
              "rms", "zero_crossings", "dominant_freq", "spectral_energy"]

# Names of the 9 HRV features we extract from Lead II
HRV_NAMES = ["mean_rr", "sdnn", "rmssd", "pnn50",
             "hr_mean", "hr_std", "lf_power", "hf_power", "lf_hf_ratio"]

# Names of the 6 morphological features we extract per lead
MORPH_NAMES = ["r_amplitude", "qrs_duration", "rs_ratio",
               "st_deviation", "t_amplitude", "beat_shape_std"]

# Names of the 6 wavelet features we extract per lead
WAVELET_NAMES = ["energy_d3", "energy_d4", "energy_approx",
                 "entropy_d3", "entropy_d4", "detail_ratio"]

# Names of the 3 frequency band energy features we extract per lead
FREQ_BAND_NAMES = ["low_band_energy", "mid_band_energy", "high_band_energy"]

# Full list of all feature names (useful for plots and analysis)
# 120 signal + 9 HRV + 72 morphological + 72 wavelet + 36 freq band = 309 total
FEATURE_NAMES = ([f"{lead}_{stat}" for lead in LEAD_NAMES for stat in STAT_NAMES]
                 + [f"HRV_{name}" for name in HRV_NAMES]
                 + [f"{lead}_{morph}" for lead in LEAD_NAMES for morph in MORPH_NAMES]
                 + [f"{lead}_{wav}" for lead in LEAD_NAMES for wav in WAVELET_NAMES]
                 + [f"{lead}_{fb}" for lead in LEAD_NAMES for fb in FREQ_BAND_NAMES])


# =============================================================================
# SINGLE-LEAD FEATURE EXTRACTION
# =============================================================================

def _extract_single_lead_features(signal_1d):
    """
    Extract 10 features from a single ECG lead.

    Parameters
    ----------
    signal_1d : np.ndarray of shape (1000,)
        The voltage values for one lead over 10 seconds at 100Hz.

    Returns
    -------
    features : list of 10 float values
        The computed features in the order listed in STAT_NAMES.
    """
    # --- Time-domain features ---

    # Mean: the average voltage. A normal ECG baseline should be near zero.
    mean_val = np.mean(signal_1d)

    # Standard deviation: how spread out the values are.
    # Higher std = more variable signal (could indicate irregular rhythm).
    std_val = np.std(signal_1d)

    # Max and Min: the peak and trough voltages.
    # Very high peaks might indicate hypertrophy (thickened heart muscle).
    max_val = np.max(signal_1d)
    min_val = np.min(signal_1d)

    # RMS (Root Mean Square): a measure of signal energy/magnitude.
    # Calculated as: sqrt(mean(signal^2))
    # Unlike mean, RMS is always positive and captures the overall "power".
    rms_val = np.sqrt(np.mean(signal_1d ** 2))

    # Zero crossings: count how many times the signal crosses zero.
    # np.sign() converts values to -1, 0, or +1.
    # np.diff() finds the change between consecutive signs.
    # A change > 0 means the signal crossed zero.
    # Abnormal hearts may have more or fewer crossings than normal.
    zero_crossings = np.sum(np.abs(np.diff(np.sign(signal_1d))) > 0)

    # --- Statistical features (using scipy.stats) ---

    # Skewness: measures asymmetry of the signal's value distribution.
    # Skew = 0 means symmetric (like a normal distribution).
    # Positive skew = longer tail to the right.
    # Negative skew = longer tail to the left.
    # Heart conditions can shift the distribution asymmetry.
    # Note: scipy returns NaN for constant signals (zero variance),
    # so we default to 0.0 in that case (no asymmetry if signal is flat).
    skew_val = stats.skew(signal_1d)
    if np.isnan(skew_val):
        skew_val = 0.0

    # Kurtosis: measures "tailedness" or how sharp the peaks are.
    # Kurtosis = 0 for a normal distribution.
    # High kurtosis = sharp, tall peaks (could indicate specific waveform changes).
    # Low kurtosis = flat, spread out signal.
    # Same NaN guard as skewness above.
    kurt_val = stats.kurtosis(signal_1d)
    if np.isnan(kurt_val):
        kurt_val = 0.0

    # --- Frequency-domain features (using scipy.fft) ---
    # The FFT (Fast Fourier Transform) decomposes the signal into its
    # component frequencies. Think of it like splitting white light into
    # a rainbow -- we see which "colors" (frequencies) are present.

    # rfft = "real FFT" -- optimized for real-valued signals (not complex).
    # Returns complex numbers whose magnitude = strength of each frequency.
    fft_values = rfft(signal_1d)
    fft_magnitudes = np.abs(fft_values)  # Convert complex -> magnitude

    # rfftfreq gives the actual frequency (in Hz) for each FFT output.
    # d=1/SAMPLING_RATE tells it the time between samples.
    fft_freqs = rfftfreq(len(signal_1d), d=1.0 / SAMPLING_RATE)

    # Dominant frequency: the frequency with the highest magnitude.
    # We skip index 0 because that's the "DC component" (just the mean).
    # A normal heart rate of 60-100 bpm = 1.0-1.67 Hz.
    # Arrhythmias would show different dominant frequencies.
    dominant_freq = fft_freqs[1:][np.argmax(fft_magnitudes[1:])]

    # Spectral energy: the total power across all frequencies.
    # Calculated as sum of squared magnitudes.
    # Higher energy = stronger overall signal.
    spectral_energy = np.sum(fft_magnitudes ** 2)

    return [
        mean_val, std_val, max_val, min_val,
        skew_val, kurt_val, rms_val, zero_crossings,
        dominant_freq, spectral_energy,
    ]


# =============================================================================
# HRV (HEART RATE VARIABILITY) FEATURES — computed from Lead II only
# =============================================================================

def _extract_hrv_features(peaks):
    """
    Extract 9 Heart Rate Variability features from R-peak locations.

    How it works:
      1. Use pre-detected R-peaks (from _find_r_peaks_leadII).
      2. Compute the time between consecutive R-peaks (RR intervals).
      3. Calculate time-domain HRV stats from those intervals.
      4. For frequency-domain HRV, interpolate the RR intervals to a
         uniform 4 Hz time series, then use FFT to measure power in the
         LF (0.04-0.15 Hz) and HF (0.15-0.4 Hz) frequency bands.

    Why Lead II only?
      Lead II has the tallest, cleanest R-peaks by design. HRV measures
      the timing between heartbeats, which is the same regardless of
      which lead you look at — so one lead is enough.

    Parameters
    ----------
    peaks : np.ndarray of int
        R-peak sample indices detected from Lead II.

    Returns
    -------
    features : list of 9 float values
        [mean_rr, sdnn, rmssd, pnn50, hr_mean, hr_std,
         lf_power, hf_power, lf_hf_ratio]
    """
    # Need at least 3 R-peaks to compute meaningful HRV features.
    # If we can't find enough peaks, return zeros (some signals are noisy).
    if len(peaks) < 3:
        return [0.0] * 9

    # --- Step 2: Compute RR intervals ---
    # RR interval = time (in seconds) between consecutive R-peaks.
    # We divide by SAMPLING_RATE to convert from samples to seconds.
    rr_intervals = np.diff(peaks) / SAMPLING_RATE  # in seconds

    # Filter out physiologically impossible RR intervals:
    # < 0.3s would mean heart rate > 200 bpm (too fast)
    # > 2.0s would mean heart rate < 30 bpm (too slow)
    valid_mask = (rr_intervals > 0.3) & (rr_intervals < 2.0)
    rr_intervals = rr_intervals[valid_mask]

    if len(rr_intervals) < 2:
        return [0.0] * 9

    # --- Step 3: Time-domain HRV features ---

    # Mean RR: average time between beats (seconds).
    # Normal resting: ~0.6-1.0s (60-100 bpm).
    mean_rr = np.mean(rr_intervals)

    # SDNN: standard deviation of RR intervals.
    # Measures overall heart rate variability.
    # Higher SDNN = more variability = generally healthier heart.
    sdnn = np.std(rr_intervals)

    # RMSSD: root mean square of successive RR differences.
    # Measures short-term (beat-to-beat) variability.
    # Sensitive to parasympathetic (rest/digest) nervous system activity.
    rr_diffs = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(rr_diffs ** 2))

    # pNN50: percentage of successive RR intervals that differ by > 50ms.
    # Another parasympathetic indicator. Healthy hearts: pNN50 > 3%.
    pnn50 = np.sum(np.abs(rr_diffs) > 0.05) / len(rr_diffs) * 100

    # Heart rate stats (convert RR intervals to beats per minute).
    heart_rates = 60.0 / rr_intervals  # bpm
    hr_mean = np.mean(heart_rates)
    hr_std = np.std(heart_rates)

    # --- Step 4: Frequency-domain HRV features ---
    # The RR intervals are unevenly spaced in time (because heart rate varies).
    # To use FFT, we need evenly spaced data. So we interpolate to 4 Hz.

    # Time points of each RR interval (cumulative sum of intervals)
    rr_times = np.cumsum(rr_intervals)
    rr_times = rr_times - rr_times[0]  # start from 0

    # Only compute frequency features if we have enough data (>2 seconds)
    if rr_times[-1] < 2.0 or len(rr_intervals) < 4:
        return [mean_rr, sdnn, rmssd, pnn50, hr_mean, hr_std,
                0.0, 0.0, 0.0]

    # Interpolate to uniform 4 Hz sampling.
    # rr_times and rr_intervals are the same length (both from np.diff(peaks)).
    # We interpolate RR interval values at evenly spaced time points.
    interp_rate = 4.0  # Hz
    uniform_times = np.arange(rr_times[0], rr_times[-1], 1.0 / interp_rate)

    if len(uniform_times) < 2:
        return [mean_rr, sdnn, rmssd, pnn50, hr_mean, hr_std,
                0.0, 0.0, 0.0]

    interpolator = interp1d(rr_times, rr_intervals,
                            kind='linear', fill_value='extrapolate')
    rr_interp = interpolator(uniform_times)

    # FFT on the interpolated RR series
    fft_vals = np.abs(rfft(rr_interp - np.mean(rr_interp))) ** 2
    fft_freqs = rfftfreq(len(rr_interp), d=1.0 / interp_rate)

    # LF power (0.04-0.15 Hz): reflects both sympathetic + parasympathetic activity.
    # Increased during stress, mental effort, or standing.
    lf_mask = (fft_freqs >= 0.04) & (fft_freqs < 0.15)
    lf_power = np.sum(fft_vals[lf_mask]) if np.any(lf_mask) else 0.0

    # HF power (0.15-0.4 Hz): reflects parasympathetic (vagal) activity.
    # Increased during relaxation and slow breathing.
    hf_mask = (fft_freqs >= 0.15) & (fft_freqs < 0.4)
    hf_power = np.sum(fft_vals[hf_mask]) if np.any(hf_mask) else 0.0

    # LF/HF ratio: indicator of autonomic balance.
    # High ratio = sympathetic dominance. Low ratio = parasympathetic dominance.
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0

    return [mean_rr, sdnn, rmssd, pnn50, hr_mean, hr_std,
            lf_power, hf_power, lf_hf_ratio]


# =============================================================================
# MORPHOLOGICAL FEATURES — beat shape analysis per lead
# =============================================================================

def _find_r_peaks_leadII(lead_ii_signal):
    """
    Detect R-peaks from Lead II (reusable by both HRV and morphological steps).

    Returns the array of peak indices, or an empty array if too few found.
    """
    signal_std = np.std(lead_ii_signal)
    signal_mean = np.mean(lead_ii_signal)
    height_threshold = signal_mean + 0.5 * signal_std

    peaks, _ = find_peaks(
        lead_ii_signal,
        distance=40,
        height=height_threshold,
    )
    return peaks


def _extract_morphological_features(lead_signal, r_peak_indices):
    """
    Extract 6 morphological features from one ECG lead using R-peak locations.

    How it works:
      1. Use R-peak positions (detected from Lead II) to cut out individual
         heartbeats from this lead. Each beat window spans from 25 samples
         before the R-peak to 40 samples after (−250ms to +400ms at 100Hz).
      2. Average all beats together to create a "template" beat.
      3. Measure clinically meaningful features from the template.

    Why use Lead II's R-peaks for all leads?
      The R-peak (tallest spike) happens at the same instant across all 12
      leads — the heart depolarizes at one moment in time. Lead II just gives
      the cleanest detection. Once we know WHEN each beat occurs, we can cut
      windows from ANY lead at those same time points.

    Parameters
    ----------
    lead_signal : np.ndarray of shape (1000,)
        The voltage values for one lead over 10 seconds at 100Hz.
    r_peak_indices : np.ndarray of int
        Sample indices of R-peaks detected from Lead II.

    Returns
    -------
    features : list of 6 float values
        [r_amplitude, qrs_duration, rs_ratio, st_deviation,
         t_amplitude, beat_shape_std]
    """
    # --- Step 1: Segment individual beats ---
    # Window: 25 samples before R-peak to 40 samples after.
    # At 100Hz: 250ms before, 400ms after. This captures P-QRS-T complex.
    pre_samples = 25   # 250ms before R-peak
    post_samples = 40  # 400ms after R-peak
    beat_length = pre_samples + post_samples  # 65 samples per beat

    beats = []
    for peak_idx in r_peak_indices:
        start = peak_idx - pre_samples
        end = peak_idx + post_samples
        # Only keep beats that fit entirely within the signal
        if start >= 0 and end <= len(lead_signal):
            beat = lead_signal[start:end]
            beats.append(beat)

    # Need at least 2 beats to compute a meaningful template
    if len(beats) < 2:
        return [0.0] * 6

    beats = np.array(beats)  # shape: (n_beats, 65)

    # --- Step 2: Create average beat template ---
    # Averaging smooths out noise while preserving the consistent waveform shape.
    template = np.mean(beats, axis=0)  # shape: (65,)

    # The R-peak is at index 25 (pre_samples) in the template
    r_idx = pre_samples  # index 25

    # --- Step 3: Measure features from the template ---

    # 1. R-peak amplitude: height of the R-wave in this lead.
    #    In leads facing the heart's electrical axis, this is tall and positive.
    #    Abnormally tall R-waves can indicate ventricular hypertrophy.
    r_amplitude = template[r_idx]

    # 2. QRS duration: width of the main spike (in milliseconds).
    #    The QRS complex is the sharp spike around the R-peak.
    #    We measure its width at 50% of the R-peak height above baseline.
    #    Normal: 80-120ms. Wide QRS (>120ms) = conduction problems.
    baseline = np.mean(template[:5])  # first 5 samples as baseline estimate
    half_height = baseline + (r_amplitude - baseline) * 0.5

    # Find where the signal crosses the half-height level around the R-peak
    # Search left from R-peak for QRS start
    qrs_start = r_idx
    for j in range(r_idx, max(r_idx - 15, 0), -1):  # search up to 150ms left
        if template[j] < half_height:
            qrs_start = j
            break

    # Search right from R-peak for QRS end
    qrs_end = r_idx
    for j in range(r_idx, min(r_idx + 15, beat_length)):  # search up to 150ms right
        if template[j] < half_height:
            qrs_end = j
            break

    # Convert from samples to milliseconds (each sample = 10ms at 100Hz)
    qrs_duration = (qrs_end - qrs_start) * (1000.0 / SAMPLING_RATE)

    # 3. R/S ratio: R-peak height divided by S-trough depth.
    #    The S-trough is the dip right after the R-peak (within ~80ms).
    #    This ratio changes across leads and helps detect chamber enlargement.
    s_region = template[r_idx:min(r_idx + 8, beat_length)]  # 80ms after R
    s_trough = np.min(s_region)
    s_depth = abs(s_trough - baseline)
    rs_ratio = abs(r_amplitude - baseline) / s_depth if s_depth > 1e-6 else 0.0

    # 4. ST deviation: voltage level in the ST segment relative to baseline.
    #    The ST segment is ~80-120ms after the R-peak (samples 33-37).
    #    ST elevation = acute myocardial infarction (heart attack).
    #    ST depression = ischemia (reduced blood flow to the heart).
    st_start = r_idx + 8   # ~80ms after R-peak
    st_end = r_idx + 12    # ~120ms after R-peak
    if st_end <= beat_length:
        st_segment = template[st_start:st_end]
        st_deviation = np.mean(st_segment) - baseline
    else:
        st_deviation = 0.0

    # 5. T-wave amplitude: height of the T-wave relative to baseline.
    #    The T-wave is the bump ~200-350ms after the R-peak (samples 45-60).
    #    T-wave inversion (negative amplitude) suggests ischemia.
    #    Tall peaked T-waves can indicate high potassium levels.
    t_start = r_idx + 20   # ~200ms after R-peak
    t_end = min(r_idx + 35, beat_length)  # ~350ms after R-peak
    if t_end > t_start:
        t_region = template[t_start:t_end]
        t_amplitude = np.max(t_region) - baseline
    else:
        t_amplitude = 0.0

    # 6. Beat shape std: how consistent individual beats are vs the template.
    #    Low std = regular, consistent heartbeats (normal sinus rhythm).
    #    High std = irregular beat shapes (arrhythmia, ectopic beats).
    #    Computed as the average standard deviation across all time points.
    beat_shape_std = np.mean(np.std(beats, axis=0))

    return [r_amplitude, qrs_duration, rs_ratio, st_deviation,
            t_amplitude, beat_shape_std]


# =============================================================================
# WAVELET FEATURES — time-frequency decomposition per lead
# =============================================================================

def _extract_wavelet_features(signal_1d):
    """
    Extract 6 wavelet features from a single ECG lead.

    How it works:
      1. Decompose the signal using the Discrete Wavelet Transform (DWT)
         with a Daubechies-4 (db4) wavelet at 4 levels.
      2. This produces 5 arrays of coefficients:
         - d1 (detail level 1): ~25-50 Hz — mostly noise, we skip this
         - d2 (detail level 2): ~12-25 Hz — high-frequency edges, we skip this
         - d3 (detail level 3): ~6-12 Hz  — QRS complex range (the main spike)
         - d4 (detail level 4): ~3-6 Hz   — T-wave and heart rate range
         - a4 (approximation):  ~0-3 Hz   — baseline wander and slow trends
      3. From d3, d4, and a4, we measure energy (total power) and entropy
         (how disordered/complex the coefficients are).

    Why wavelets instead of just FFT?
      FFT averages over the entire 10 seconds — it tells you WHAT frequencies
      exist but not WHEN they occur. Wavelets capture both time and frequency,
      so they can see the sharp QRS spike (brief, high-frequency) separately
      from the slow T-wave (long, low-frequency). This is critical because
      abnormal hearts often have normal frequencies but at wrong times.

    Why Daubechies-4?
      The db4 wavelet shape resembles a QRS complex — it has a sharp peak
      with small oscillations. This makes it naturally good at detecting and
      measuring heartbeat features. It's the most commonly used wavelet for
      ECG analysis in research literature.

    Parameters
    ----------
    signal_1d : np.ndarray of shape (1000,)
        The voltage values for one lead over 10 seconds at 100Hz.

    Returns
    -------
    features : list of 6 float values
        [energy_d3, energy_d4, energy_approx, entropy_d3, entropy_d4,
         detail_ratio]
    """
    # Decompose the signal into wavelet coefficients at 4 levels.
    # pywt.wavedec returns [a4, d4, d3, d2, d1] (approximation first, then
    # details from coarsest to finest).
    coeffs = pywt.wavedec(signal_1d, wavelet='db4', level=4)
    # coeffs[0] = a4 (approximation, ~0-3 Hz)
    # coeffs[1] = d4 (detail level 4, ~3-6 Hz)
    # coeffs[2] = d3 (detail level 3, ~6-12 Hz)
    # coeffs[3] = d2 (detail level 2, ~12-25 Hz) — skipped (mostly noise)
    # coeffs[4] = d1 (detail level 1, ~25-50 Hz) — skipped (mostly noise)

    a4 = coeffs[0]
    d4 = coeffs[1]
    d3 = coeffs[2]

    # --- Energy features ---
    # Energy = sum of squared coefficients. Measures how much signal power
    # lives in each frequency band. Higher energy = more activity at that scale.
    energy_d3 = np.sum(d3 ** 2)
    energy_d4 = np.sum(d4 ** 2)
    energy_approx = np.sum(a4 ** 2)

    # --- Entropy features ---
    # Shannon entropy measures how "disordered" or "complex" the coefficients are.
    # Low entropy = organized, predictable pattern (like a clean normal heartbeat).
    # High entropy = chaotic, unpredictable (like atrial fibrillation).
    # We normalize coefficients to a probability-like distribution first.
    def _wavelet_entropy(c):
        """Compute Shannon entropy of wavelet coefficients."""
        c_squared = c ** 2
        total = np.sum(c_squared)
        if total < 1e-10:
            return 0.0
        # Normalize to probability distribution (values sum to 1)
        p = c_squared / total
        # Remove zeros to avoid log(0) which is undefined
        p = p[p > 0]
        # Shannon entropy: -sum(p * log(p))
        return -np.sum(p * np.log(p))

    entropy_d3 = _wavelet_entropy(d3)
    entropy_d4 = _wavelet_entropy(d4)

    # --- Detail ratio ---
    # Ratio of high-frequency energy (d3 = QRS range) to low-frequency
    # energy (a4 = baseline). A high ratio means the signal has strong,
    # sharp features relative to the baseline — typical of clear heartbeats.
    # A low ratio means the signal is dominated by slow drift (possible
    # baseline wander or low-amplitude ECG).
    detail_ratio = energy_d3 / energy_approx if energy_approx > 1e-10 else 0.0

    return [energy_d3, energy_d4, energy_approx,
            entropy_d3, entropy_d4, detail_ratio]


# =============================================================================
# FREQUENCY BAND ENERGY FEATURES — clinically meaningful spectral bands
# =============================================================================

def _extract_freq_band_features(signal_1d):
    """
    Extract 3 frequency band energy features from a single ECG lead.

    How it works:
      1. Compute the FFT of the signal (same as _extract_single_lead_features).
      2. Split the frequency spectrum into 3 clinically meaningful bands.
      3. Sum the squared magnitudes in each band to get the energy.

    Why split into bands?
      Our existing spectral_energy feature sums ALL frequencies together.
      But different heart conditions affect different frequency ranges:
        - P-wave and T-wave abnormalities show up in low frequencies
        - QRS problems (bundle branch block) show up in mid frequencies
        - Fine fragmentation (scarring) shows up in high frequencies
      Splitting lets the classifier see WHERE the energy is concentrated.

    Band definitions (based on ECG physiology literature):
      - Low  (0.5-5 Hz):  P-wave, T-wave, and heart rate rhythm.
      - Mid  (5-15 Hz):   QRS complex (the main heartbeat spike).
      - High (15-40 Hz):  Sharp edges, high-frequency notches, and noise.

    We start at 0.5 Hz (not 0 Hz) to exclude the DC component (just the
    mean voltage level, which carries no rhythm information).

    Parameters
    ----------
    signal_1d : np.ndarray of shape (1000,)
        The voltage values for one lead over 10 seconds at 100Hz.

    Returns
    -------
    features : list of 3 float values
        [low_band_energy, mid_band_energy, high_band_energy]
    """
    # Compute FFT magnitudes and frequencies
    fft_values = rfft(signal_1d)
    fft_magnitudes = np.abs(fft_values)
    fft_freqs = rfftfreq(len(signal_1d), d=1.0 / SAMPLING_RATE)

    # Squared magnitudes = power at each frequency
    power = fft_magnitudes ** 2

    # Low band (0.5-5 Hz): P-wave, T-wave, heart rate
    low_mask = (fft_freqs >= 0.5) & (fft_freqs < 5.0)
    low_band_energy = np.sum(power[low_mask])

    # Mid band (5-15 Hz): QRS complex
    mid_mask = (fft_freqs >= 5.0) & (fft_freqs < 15.0)
    mid_band_energy = np.sum(power[mid_mask])

    # High band (15-40 Hz): sharp transitions, fine detail
    high_mask = (fft_freqs >= 15.0) & (fft_freqs < 40.0)
    high_band_energy = np.sum(power[high_mask])

    return [low_band_energy, mid_band_energy, high_band_energy]


# =============================================================================
# BATCH FEATURE EXTRACTION
# =============================================================================

def extract_features(X_signals):
    """
    Extract features from a batch of ECG recordings.

    This function loops through every record and every lead, extracting
    10 features from each lead. The result is a 2D array where each row
    is one ECG record and each column is one feature.

    Parameters
    ----------
    X_signals : np.ndarray of shape (n_records, 1000, 12)
        Raw ECG signal data.
        - n_records: number of ECG recordings
        - 1000: time steps (10 seconds at 100Hz)
        - 12: ECG leads

    Returns
    -------
    features : np.ndarray of shape (n_records, 309)
        The extracted feature vectors.
        120 per-lead signal features (12 leads x 10)
        + 9 HRV features (from Lead II R-peaks)
        + 72 morphological features (12 leads x 6)
        + 72 wavelet features (12 leads x 6)
        + 36 frequency band energy features (12 leads x 3)
    """
    n_records = X_signals.shape[0]
    all_features = []

    for i in range(n_records):
        record_features = []

        # Loop through all 12 leads for per-lead signal features
        for lead_idx in range(12):
            lead_signal = X_signals[i, :, lead_idx]  # shape: (1000,)
            lead_features = _extract_single_lead_features(lead_signal)
            record_features.extend(lead_features)  # Add 10 features to the list

        # Detect R-peaks once from Lead II (index 1), reuse for HRV + morphology
        lead_ii = X_signals[i, :, 1]
        r_peaks = _find_r_peaks_leadII(lead_ii)

        # Extract 9 HRV features from the R-peak timing
        hrv_features = _extract_hrv_features(r_peaks)
        record_features.extend(hrv_features)

        # Extract 6 morphological features from each lead using the R-peak positions
        for lead_idx in range(12):
            lead_signal = X_signals[i, :, lead_idx]
            morph_features = _extract_morphological_features(lead_signal, r_peaks)
            record_features.extend(morph_features)

        # Extract 6 wavelet features from each lead
        for lead_idx in range(12):
            lead_signal = X_signals[i, :, lead_idx]
            wav_features = _extract_wavelet_features(lead_signal)
            record_features.extend(wav_features)

        # Extract 3 frequency band energy features from each lead
        for lead_idx in range(12):
            lead_signal = X_signals[i, :, lead_idx]
            fb_features = _extract_freq_band_features(lead_signal)
            record_features.extend(fb_features)

        all_features.append(record_features)

        # Print progress every 2000 records
        if (i + 1) % 2000 == 0:
            print(f"    Extracted features for {i + 1}/{n_records} records...")

    print(f"    Extracted features for {n_records}/{n_records} records.")
    features = np.array(all_features)

    # Safety net: replace any remaining NaN or infinite values with 0.
    # This can happen if an entire lead is all zeros (e.g., std=0 causes issues).
    nan_count = np.sum(np.isnan(features))
    if nan_count > 0:
        print(f"    Note: Replaced {nan_count} NaN values in features with 0.")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


# =============================================================================
# FEATURE SELECTION — remove redundant/noisy features
# =============================================================================

def select_features(X_train, y_train, X_val, X_test, max_features=150):
    """
    Select the most useful features and discard the rest.

    Why feature selection?
      With 309 features, many are redundant (e.g., spectral_energy ≈ sum of
      band energies) or noisy (carrying no useful signal). Redundant features:
        - Add noise to SVM's distance calculations
        - Confuse K-Means (it treats every dimension equally)
        - Can cause overfitting (model memorizes noise instead of patterns)
      Removing them often IMPROVES accuracy, especially for K-Means.

    How it works (two independent rankings, then combine):

      1. RANDOM FOREST IMPORTANCE:
         Train a quick Random Forest and ask it: "Which features helped you
         the most?" Each tree tracks which features reduced prediction errors.
         Features that rarely help get low importance scores.

      2. MUTUAL INFORMATION:
         A statistics-based measure (no model needed). For each feature, it
         asks: "How much does knowing this feature's value reduce my uncertainty
         about the label?" High mutual info = the feature carries useful signal.
         This catches features the RF might miss (e.g., features that are useful
         in combination but weak alone).

      3. COMBINE:
         We rank features by both methods, take the top N from each, and keep
         the union. This gives us a robust set that both methods agree on.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_samples, 309)
        Training features (unscaled).
    y_train : np.ndarray of shape (n_samples,)
        Training labels (0=Normal, 1=Abnormal).
    X_val : np.ndarray of shape (n_val, 309)
        Validation features.
    X_test : np.ndarray of shape (n_test, 309)
        Test features.
    max_features : int
        Maximum number of features to keep (default: 150).

    Returns
    -------
    X_train_sel, X_val_sel, X_test_sel : np.ndarray
        Feature arrays with only the selected columns.
    selected_names : list of str
        Names of the kept features.
    selected_indices : np.ndarray of int
        Column indices of the kept features (useful for later analysis).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif

    n_features = X_train.shape[1]
    print(f"    Starting with {n_features} features, selecting top {max_features}...")

    # --- Method 1: Random Forest feature importance ---
    # Train a small, fast RF just for ranking features (not for classification).
    # 50 trees is enough for stable importance scores.
    print("    Computing Random Forest feature importances...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=10,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_importances = rf.feature_importances_

    # --- Method 2: Mutual Information ---
    # mutual_info_classif estimates how much each feature tells us about the label.
    # random_state for reproducibility, n_neighbors=5 is the default.
    print("    Computing mutual information scores...")
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

    # --- Combine both rankings ---
    # Get top max_features from each method
    rf_top = set(np.argsort(rf_importances)[-max_features:])
    mi_top = set(np.argsort(mi_scores)[-max_features:])

    # Union: keep any feature that EITHER method ranked highly
    selected_set = rf_top | mi_top

    # If the union is larger than max_features, trim by average rank
    if len(selected_set) > max_features:
        # Rank each feature (higher rank = more important)
        rf_ranks = np.argsort(np.argsort(rf_importances))  # 0=worst, N-1=best
        mi_ranks = np.argsort(np.argsort(mi_scores))
        avg_ranks = (rf_ranks + mi_ranks) / 2.0

        # Sort by average rank, keep top max_features
        candidates = sorted(selected_set, key=lambda i: avg_ranks[i], reverse=True)
        selected_set = set(candidates[:max_features])

    selected_indices = np.sort(list(selected_set))
    selected_names = [FEATURE_NAMES[i] for i in selected_indices]

    # Apply selection to all splits
    X_train_sel = X_train[:, selected_indices]
    X_val_sel = X_val[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]

    print(f"    Selected {len(selected_indices)} features.")

    return X_train_sel, X_val_sel, X_test_sel, selected_names, selected_indices
