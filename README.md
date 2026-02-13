# Heart Disease Classifier

Classify heart conditions from ECG (electrocardiogram) signals using machine learning. This project uses the **PTB-XL** dataset and compares three approaches: **SVM**, **Random Forest**, and **K-Means**.

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

_Full documentation will be added after implementation._
