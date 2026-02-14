"""
data_loader.py -- Download and load the PTB-XL ECG dataset.

This module handles the entire data pipeline:
  1. Downloading the dataset from PhysioNet (if not already on disk).
  2. Reading the metadata CSV files.
  3. Loading the 100Hz ECG signal data using the wfdb library.
  4. Parsing the scp_codes column to assign diagnostic labels.
  5. Splitting into train / validation / test sets.

What is the PTB-XL dataset?
  - 21,799 clinical 12-lead ECG recordings from 18,869 patients.
  - Each recording is 10 seconds long.
  - A 12-lead ECG records electrical activity of the heart from 12 angles
    (called "leads"): I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.
  - Each record has diagnostic labels (e.g., Normal, Myocardial Infarction).
  - Available at: https://physionet.org/content/ptb-xl/1.0.3/
"""

import os
import ast
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import wfdb


# =============================================================================
# CONSTANTS
# =============================================================================

# URL for the PTB-XL ZIP file on PhysioNet (version 1.0.3)
DATASET_URL = (
    "https://physionet.org/static/published-projects/ptb-xl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
)

# Where the data folder lives, relative to the project root.
# os.path.dirname(__file__) gives us the "src/" folder.
# Going one level up (dirname again) gives us the project root.
# We then join "data" to get the path: <project_root>/data/
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)

# After unzipping, the dataset lives inside this subfolder.
DATASET_SUBDIR = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"


# =============================================================================
# DOWNLOAD FUNCTION
# =============================================================================

def download_dataset():
    """
    Download and extract the PTB-XL dataset if it is not already present.

    The ZIP file is approximately 1.7 GB. Extraction produces ~2.5 GB of files.
    This function prints progress messages so you know it's working.

    Returns
    -------
    dataset_path : str
        The full path to the extracted dataset folder.
    """
    dataset_path = os.path.join(DATA_DIR, DATASET_SUBDIR)

    # If the folder already exists, skip downloading
    if os.path.isdir(dataset_path):
        print(f"  Dataset already exists at: {dataset_path}")
        return dataset_path

    # Create the data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    zip_path = os.path.join(DATA_DIR, "ptb-xl.zip")

    # Download the ZIP file with a progress indicator
    print("  Downloading PTB-XL dataset (~1.7 GB)...")
    print("  This may take several minutes depending on your internet speed.")

    def _download_progress(block_count, block_size, total_size):
        """Print download progress so the user knows it's not frozen."""
        downloaded = block_count * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent:5.1f}%  ({mb_done:.0f} / {mb_total:.0f} MB)", end="", flush=True)

    urllib.request.urlretrieve(DATASET_URL, zip_path, reporthook=_download_progress)
    print("\n  Download complete!")

    # Extract the ZIP file
    print("  Extracting files...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    # Remove the zip file to save disk space
    os.remove(zip_path)
    print(f"  Dataset extracted to: {dataset_path}")

    return dataset_path


# =============================================================================
# LABEL PARSING FUNCTIONS
# =============================================================================

def _parse_scp_codes(scp_codes_str):
    """
    Convert the scp_codes string from the CSV into a Python dictionary.

    In the CSV file, the scp_codes column stores dictionaries as strings,
    for example: "{'NORM': 100.0, 'SR': 0.0}"

    ast.literal_eval() safely converts this string into an actual Python
    dictionary. We use this instead of eval() because eval() can execute
    arbitrary code (security risk), while literal_eval() only parses
    simple data structures like dicts, lists, and numbers.

    Parameters
    ----------
    scp_codes_str : str
        The raw string from the CSV column.

    Returns
    -------
    dict
        A dictionary mapping SCP code names to likelihood scores.
    """
    return ast.literal_eval(scp_codes_str)


def _assign_superclass(scp_dict, code_to_superclass):
    """
    Map SCP diagnostic codes to their superclass categories.

    The PTB-XL dataset uses 71 specific diagnostic codes (like "NORM",
    "IMI", "AFIB"). These belong to 5 broader superclasses:
      - NORM: Normal ECG
      - MI:   Myocardial Infarction (heart attack)
      - STTC: ST/T Change (changes in specific parts of the heartbeat)
      - CD:   Conduction Disturbance (electrical signal problems)
      - HYP:  Hypertrophy (thickened heart muscle)

    Each code has a "likelihood" score (0-100). We only consider codes
    with likelihood > 50, which is the dataset's convention for a
    confident diagnosis.

    Parameters
    ----------
    scp_dict : dict
        Maps SCP code names to likelihood scores, e.g. {'NORM': 100.0}.
    code_to_superclass : dict
        Maps each SCP code to its superclass, e.g. {'NORM': 'NORM', 'IMI': 'MI'}.

    Returns
    -------
    list of str
        The superclass labels that apply to this record, e.g. ['NORM'] or ['MI', 'STTC'].
    """
    labels = set()
    for code, likelihood in scp_dict.items():
        if likelihood > 50.0 and code in code_to_superclass:
            superclass = code_to_superclass[code]
            if pd.notna(superclass):
                labels.add(superclass)
    return list(labels)


def _create_binary_label(superclass_list):
    """
    Convert superclass labels into a simple binary label.

    This is the simplest classification task:
      0 = Normal   (the record has ONLY the 'NORM' superclass)
      1 = Abnormal (the record has any other superclass, or no valid label)

    Why binary? Multi-label classification (where one ECG can have multiple
    conditions) is complex. Starting with Normal vs Abnormal is the easiest
    way to learn classification.

    Parameters
    ----------
    superclass_list : list of str
        The superclass labels for one ECG record.

    Returns
    -------
    int
        0 for Normal, 1 for Abnormal.
    """
    if len(superclass_list) == 1 and superclass_list[0] == "NORM":
        return 0  # Normal
    return 1      # Abnormal


# =============================================================================
# SIGNAL LOADING FUNCTION
# =============================================================================

def _load_ecg_signals(dataset_path, filenames):
    """
    Load ECG signal arrays from WFDB files.

    WFDB (WaveForm DataBase) is the standard format for storing physiological
    signals like ECGs. Each record has two files:
      - .hea (header): metadata about the recording
      - .dat (data): the actual signal values in binary format

    The wfdb.rdrecord() function reads both files and returns a Record object.
    The .p_signal attribute contains the signal as a NumPy array.

    We use the 100Hz version (filename_lr = "low resolution") instead of
    the 500Hz version. This means:
      - Each record has 1,000 time steps (10 seconds x 100 samples/second)
      - Each time step has 12 values (one per ECG lead)
      - Shape per record: (1000, 12)

    Parameters
    ----------
    dataset_path : str
        Root folder of the extracted PTB-XL dataset.
    filenames : list of str
        The 'filename_lr' column values, e.g. 'records100/00000/00001_lr'.

    Returns
    -------
    signals : np.ndarray of shape (n_records, 1000, 12)
    """
    signals = []
    total = len(filenames)

    for i, fname in enumerate(filenames):
        full_path = os.path.join(dataset_path, fname)
        record = wfdb.rdrecord(full_path)
        signals.append(record.p_signal)  # shape: (1000, 12)

        # Print progress every 2000 records so you know it's working
        if (i + 1) % 2000 == 0:
            print(f"    Loaded {i + 1}/{total} ECG records...")

    print(f"    Loaded {total}/{total} ECG records.")
    return np.array(signals)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def load_dataset():
    """
    Download (if needed), load, and prepare the entire PTB-XL dataset.

    This is the only function you need to call from outside this module.
    It handles everything and returns a dictionary with all the data
    you need for training and evaluation.

    The dataset is split using the built-in 'strat_fold' column:
      - Folds 1-8: Training set   (~17,000 records)
      - Fold 9:    Validation set  (~2,100 records)
      - Fold 10:   Test set        (~2,100 records, highest quality labels)

    This stratified split ensures similar class distributions across sets
    and is the recommended split from the dataset authors.

    Returns
    -------
    dict with keys:
        'X_train', 'X_val', 'X_test'
            NumPy arrays of shape (n_samples, 1000, 12) -- the raw ECG signals.
        'y_train', 'y_val', 'y_test'
            NumPy arrays of integers (0=Normal, 1=Abnormal) -- binary labels.
        'metadata'
            The full pandas DataFrame with all metadata for reference.
    """
    # Step 1: Download the dataset if it's not already on disk
    print("  Checking for dataset...")
    dataset_path = download_dataset()

    # Step 2: Load the metadata CSV files
    print("  Loading metadata...")
    db = pd.read_csv(
        os.path.join(dataset_path, "ptbxl_database.csv"),
        index_col="ecg_id"  # Use the ECG ID as the row index
    )
    scp_statements = pd.read_csv(
        os.path.join(dataset_path, "scp_statements.csv"),
        index_col=0  # Use the SCP code as the row index
    )

    # Step 3: Build a mapping from SCP code -> superclass
    # Example: {'NORM': 'NORM', 'IMI': 'MI', 'AFIB': 'CD', ...}
    # We only keep codes that have a diagnostic_class defined.
    scp_with_class = scp_statements[scp_statements.diagnostic_class.notna()]
    code_to_superclass = scp_with_class.diagnostic_class.to_dict()

    # Step 4: Parse labels for every record
    # Convert the string "{'NORM': 100.0}" into a real dictionary
    db["scp_codes_dict"] = db.scp_codes.apply(_parse_scp_codes)
    # Map to superclass labels like ['NORM'] or ['MI', 'STTC']
    db["superclass_list"] = db.scp_codes_dict.apply(
        lambda d: _assign_superclass(d, code_to_superclass)
    )
    # Create binary label: 0=Normal, 1=Abnormal
    db["binary_label"] = db.superclass_list.apply(_create_binary_label)

    # Step 5: Split using the strat_fold column
    train_df = db[db.strat_fold.isin(range(1, 9))]  # Folds 1-8
    val_df = db[db.strat_fold == 9]                   # Fold 9
    test_df = db[db.strat_fold == 10]                  # Fold 10

    print(f"  Split sizes -- Train: {len(train_df)}, "
          f"Val: {len(val_df)}, Test: {len(test_df)}")

    # Step 6: Load the actual ECG signal data for each split
    print("  Loading training ECG signals...")
    X_train = _load_ecg_signals(dataset_path, train_df.filename_lr.tolist())

    print("  Loading validation ECG signals...")
    X_val = _load_ecg_signals(dataset_path, val_df.filename_lr.tolist())

    print("  Loading test ECG signals...")
    X_test = _load_ecg_signals(dataset_path, test_df.filename_lr.tolist())

    # Return everything in a dictionary
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": train_df.binary_label.values,
        "y_val": val_df.binary_label.values,
        "y_test": test_df.binary_label.values,
        "metadata": db,
    }
