"""
feature_extraction.py -- Extract numerical features from raw ECG signals.

Why do we need feature extraction?
  Machine learning models like SVM and Random Forest need numerical input
  in a fixed-size format: one row per sample, one column per feature.
  But our ECG data is a 2D array per sample (1000 timesteps x 12 leads).
  We need to summarize each signal into a small set of meaningful numbers.

What features do we extract?
  For each of the 12 ECG leads, we compute 10 features:

  TIME-DOMAIN features (describe the shape of the signal):
    1. Mean           -- average voltage level
    2. Std deviation  -- how much the voltage varies
    3. Max            -- highest voltage peak
    4. Min            -- lowest voltage dip
    5. RMS            -- root mean square, measures signal energy
    6. Zero crossings -- how often the signal crosses the baseline (zero)

  STATISTICAL features (describe the distribution of values):
    7. Skewness       -- asymmetry (positive = tail to the right)
    8. Kurtosis       -- "peakedness" (high = sharp peaks)

  FREQUENCY-DOMAIN features (describe the rhythmic content):
    9. Dominant freq   -- the strongest rhythm frequency (in Hz)
   10. Spectral energy -- total power across all frequencies

  Total: 10 features x 12 leads = 120 features per ECG record.

Why these features?
  Heart conditions change the ECG in characteristic ways:
  - Arrhythmias change the heart rate -> affects dominant frequency
  - Myocardial infarction changes the ST segment -> affects skewness, zero crossings
  - Hypertrophy increases voltage -> affects max, min, RMS
  These simple features capture enough information for basic classification.

Uses scipy for:
  - scipy.stats.skew and scipy.stats.kurtosis (statistical features)
  - scipy.fft.rfft and scipy.fft.rfftfreq (frequency-domain features)
"""

import numpy as np
from scipy import stats
from scipy.fft import rfft, rfftfreq


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

# Full list of all 120 feature names (useful for plots and analysis)
# Format: "LeadName_FeatureName", e.g., "II_mean", "V1_kurtosis"
FEATURE_NAMES = [f"{lead}_{stat}" for lead in LEAD_NAMES for stat in STAT_NAMES]


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
    features : np.ndarray of shape (n_records, 120)
        The extracted feature vectors.
        120 = 12 leads x 10 features per lead.
    """
    n_records = X_signals.shape[0]
    all_features = []

    for i in range(n_records):
        record_features = []

        # Loop through all 12 leads for this record
        for lead_idx in range(12):
            lead_signal = X_signals[i, :, lead_idx]  # shape: (1000,)
            lead_features = _extract_single_lead_features(lead_signal)
            record_features.extend(lead_features)  # Add 10 features to the list

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
