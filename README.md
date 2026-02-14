# Heart Disease Classifier

Classify heart conditions from ECG (electrocardiogram) signals using machine learning. This project uses the **PTB-XL** dataset and compares three approaches: **SVM**, **Random Forest**, and **K-Means**.

## What This Project Does

1. Downloads 21,799 real clinical ECG recordings from the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/)
2. Extracts 120 numerical features from each recording using signal processing (scipy)
3. Trains three machine learning models to classify ECGs as **Normal** or **Abnormal**
4. Compares the models and generates plots showing the results

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
│   └── visualization.py        # Plots: confusion matrices, comparison charts
├── data/                       # Downloaded dataset (auto-created, not in git)
├── results/                    # Generated plots and reports (auto-created)
├── main.py                     # Entry point — run this file
├── requirements.txt            # Python dependencies
└── .gitignore                  # Files git should ignore
```

## How It Works

### The Pipeline

When you run `python main.py`, five steps happen in order:

| Step | What Happens | Module |
|------|-------------|--------|
| 1 | Download and load 21,799 ECG recordings | `data_loader.py` |
| 2 | Extract 120 features per recording (10 per lead x 12 leads) | `feature_extraction.py` |
| 3 | Scale features to a standard range | `classifiers.py` |
| 4 | Train SVM, Random Forest, and K-Means; evaluate on test set | `classifiers.py` |
| 5 | Generate comparison plots and text reports | `visualization.py` |

### The Three Classifiers

| Classifier | Type | How It Works |
|-----------|------|-------------|
| **SVM** | Supervised | Finds the best boundary that separates Normal from Abnormal ECGs |
| **Random Forest** | Supervised | 100 decision trees vote on each prediction |
| **K-Means** | Unsupervised | Groups ECGs by similarity without using labels, then maps groups to labels |

### Features Extracted (per lead)

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
- `scipy` — Signal processing and statistics
- `matplotlib` — Charts and plots

## References

- Wagner, P., et al. "PTB-XL, a large publicly available electrocardiography dataset." *Scientific Data* 7, 154 (2020). [https://doi.org/10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6)
- [PhysioNet PTB-XL page](https://physionet.org/content/ptb-xl/1.0.3/)
