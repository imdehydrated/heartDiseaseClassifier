# Heart Disease Classifier

Classify heart conditions from ECG (electrocardiogram) signals using machine learning and deep learning. This project uses the **PTB-XL** dataset and compares four approaches: **SVM**, **Random Forest**, **K-Means**, and a **1D CNN**.

## What This Project Does

1. Downloads 21,799 real clinical ECG recordings from the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/)
2. Extracts 309 numerical features from each recording using signal processing (scipy, PyWavelets)
3. Trains three ML models on the extracted features to classify ECGs as **Normal** or **Abnormal**
4. Trains a 1D CNN directly on the raw ECG signals (no manual feature extraction needed)
5. Compares all four models and generates plots showing the results

## Quick Start

```powershell
# 1. Clone the repository
git clone https://github.com/imdehydrated/heartDiseaseClassifier.git
cd heartDiseaseClassifier

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the classifier (downloads dataset on first run — ~1.7 GB)
python main.py
```

## Project Structure

```
heartDiseaseClassifier/
├── src/                        # All source code
│   ├── __init__.py             # Makes src a Python package
│   ├── data_loader.py          # Downloads PTB-XL dataset, loads signals & labels
│   ├── feature_extraction.py   # Extracts numerical features using scipy
│   ├── classifiers.py          # SVM, Random Forest, K-Means training & evaluation
│   ├── cnn_model.py            # 1D CNN: learns features from raw ECG signals
│   └── visualization.py        # Plots: confusion matrices, comparison charts
├── data/                       # Downloaded dataset (auto-created, not in git)
├── results/                    # Generated plots and reports (auto-created)
├── main.py                     # Entry point — run this file
├── requirements.txt            # Python dependencies
└── .gitignore                  # Files git should ignore
```

## How It Works

### The Pipeline

When you run `python main.py`, seven steps happen in order:

| Step | What Happens | Module |
|------|-------------|--------|
| 1 | Download and load 21,799 ECG recordings | `data_loader.py` |
| 2 | Extract 309 features per recording (signal, HRV, morphological, wavelet, frequency band) | `feature_extraction.py` |
| 3 | Select best features using RF importance + mutual information (removes redundant/noisy ones) | `feature_extraction.py` |
| 4 | Scale features to a standard range | `classifiers.py` |
| 5 | Train SVM, Random Forest, and K-Means on extracted features; evaluate on test set | `classifiers.py` |
| 6 | Train 1D CNN on raw ECG signals (skips feature extraction entirely); evaluate on test set | `cnn_model.py` |
| 7 | Generate comparison plots and text reports | `visualization.py` |

### The Four Classifiers

| Classifier | Type | How It Works |
|-----------|------|-------------|
| **SVM** | Supervised | Finds the best curved boundary between Normal and Abnormal (with PCA + GridSearchCV tuning) |
| **Random Forest** | Supervised | 500 decision trees vote on each prediction (with balanced class weights) |
| **K-Means** | Unsupervised | Groups ECGs into 5 clusters by similarity (with PCA to 20 dimensions), then maps clusters to labels |
| **1D CNN** | Deep Learning | Learns features directly from raw ECG signals using a multi-scale entry block (kernel=7/15/31), LeakyReLU activations, per-lead normalization, data augmentation (noise, scaling, shift, lead dropout), optional label smoothing, deterministic TTA, and validation-calibrated decision thresholding |

**Why a CNN?** The three ML models above rely on 309 features that we designed by hand (mean, std, wavelet energy, etc.). The CNN takes a completely different approach: it receives the raw ECG waveform and learns its own features through convolutional filters. This is the key advantage of deep learning -- it replaces hundreds of lines of manual feature engineering with automatic feature learning, and often discovers patterns that humans wouldn't think to look for.

**CNN architecture (Session 2):** The first convolutional block is replaced by a multi-scale entry block that runs three parallel convolutions (kernel=7/15/31, covering 70ms/150ms/310ms of ECG signal) and concatenates their outputs into 48 channels. This lets the model detect QRS spikes, full QRS complex width, and P-wave/T-wave features simultaneously. All conv blocks use LeakyReLU instead of ReLU to prevent dead neurons. Total parameters: ~59k (up from ~46k).

**CNN preprocessing:** Each ECG lead is normalized to zero mean and unit standard deviation before entering the network (removes voltage scale differences between patients). The current default uses train-set per-lead statistics for all splits (`normalization_mode='train_stats_per_lead'`), with per-record normalization still available as an option. During training, four augmentations are applied: Gaussian noise, amplitude scaling (0.8-1.2x), time shifts (up to 50 samples), and lead dropout (each lead has a 15% chance of being silenced to simulate poor electrode contact). Label smoothing is available as an optional regularizer, and is currently OFF by default.

**CNN clinical threshold policy:** The final CNN prediction is not forced to use a fixed 0.50 cutoff. Instead, the model chooses a validation-based threshold that maximizes **Abnormal recall** while enforcing a minimum **Normal specificity** floor (default 0.80). If no threshold satisfies the floor, the floor is relaxed in 0.02 steps. This makes the classifier safer for screening-style use while keeping weighted F1 reporting for comparability.

**CNN robust inference (TTA):** At validation and test time, the model uses deterministic test-time augmentation (identity, small time shifts, small amplitude scales). Logits from these views are averaged before sigmoid conversion. This usually improves stability without changing training.

**CNN diagnosis knobs:** `train_cnn(...)` also supports optional tuning kwargs (`label_smoothing`, `lead_dropout_p`, `early_stop_patience`, `scheduler_patience`, `max_epochs`, `early_stop_mode`, `normalization_mode`, `random_seed`) so controlled ablations can be run without changing the `main.py` call signature. `load_dataset(...)` also supports optional `paper_style_filter=True` for paper-like metadata filtering.

### What Changed and Why (Simple Terms)

- We tested 9 controlled CNN runs (A0/A1/A2/A3/B1/B2/C1/S1/S2) to find out why results dropped.
- Turning off label smoothing helped the most in the main run (A1), so label smoothing is now OFF by default.
- Turning off lead dropout did not help clinical specificity, so lead dropout stays ON.
- Longer training and longer patience increased recall, but pushed specificity below the clinical target on test data.
- We also tested paper-like preprocessing options. Best result came from train-set per-lead normalization without paper-style record filtering.
- Bottom line: we kept lead dropout + clinical threshold + deterministic TTA, kept label smoothing OFF, switched default normalization to train-set per-lead stats, and left paper-style filtering as optional.

### Features Extracted (309 total)

**A. Per-Lead Signal Features** (10 x 12 leads = 120)

| Feature | What It Measures |
|---------|-----------------|
| Mean | Average voltage level |
| Std deviation | Signal variability |
| Max / Min | Peak and trough voltages |
| RMS | Overall signal energy |
| Zero crossings | How often signal crosses the baseline |
| Skewness | Asymmetry of the signal |
| Kurtosis | Sharpness of peaks |
| Dominant frequency | Strongest rhythm (in Hz) |
| Spectral energy | Total power across frequencies |

**B. Heart Rate Variability (HRV) Features** (9 from Lead II)

| Feature | What It Measures |
|---------|-----------------|
| Mean RR / SDNN | Average and variability of time between beats |
| RMSSD / pNN50 | Short-term beat-to-beat variability |
| HR mean / HR std | Heart rate average and variability |
| LF power / HF power | Sympathetic and parasympathetic nervous activity |
| LF/HF ratio | Autonomic balance indicator |

**C. Morphological Features** (6 x 12 leads = 72)

| Feature | What It Measures |
|---------|-----------------|
| R-peak amplitude | Height of the R-wave |
| QRS duration | Width of the main spike (ms) |
| R/S ratio | R-peak height vs S-trough depth |
| ST deviation | Voltage shift after QRS (heart attack indicator) |
| T-wave amplitude | Height of the T-wave |
| Beat shape std | Consistency of individual beats |

**D. Wavelet Features** (6 x 12 leads = 72)

| Feature | What It Measures |
|---------|-----------------|
| Energy d3 / d4 / approx | Signal power at QRS, T-wave, and baseline scales |
| Entropy d3 / d4 | Complexity/disorder at each scale |
| Detail ratio | High-frequency vs low-frequency energy balance |

**E. Frequency Band Energy Features** (3 x 12 leads = 36)

| Feature | What It Measures |
|---------|-----------------|
| Low band (0.5–5 Hz) | P-wave, T-wave, heart rate rhythm |
| Mid band (5–15 Hz) | QRS complex energy |
| High band (15–40 Hz) | Sharp transitions and fine details |

## Dataset

- **Source:** [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) on PhysioNet
- **Size:** 21,799 twelve-lead ECG recordings from 18,869 patients
- **Duration:** 10 seconds each, sampled at 100 Hz
- **Labels:** 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP) — simplified to Normal vs Abnormal
- **Split:** Folds 1-8 for training, fold 9 for validation, fold 10 for testing

## Dependencies

- `wfdb` — Read ECG data files
- `pandas` — CSV and tabular data
- `numpy` — Numerical operations
- `scikit-learn` — Machine learning models and metrics
- `scipy` — Signal processing, statistics, and FFT
- `matplotlib` — Charts and plots
- `PyWavelets` — Wavelet decomposition for time-frequency features
- `torch` — PyTorch deep learning framework (1D CNN model)

## References

- Wagner, P., et al. "PTB-XL, a large publicly available electrocardiography dataset." *Scientific Data* 7, 154 (2020). [https://doi.org/10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6)
- [PhysioNet PTB-XL page](https://physionet.org/content/ptb-xl/1.0.3/)
