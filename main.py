"""
main.py -- Heart Disease Classifier Pipeline

This is the main entry point for the project. When you run this file,
it executes the entire machine learning pipeline from start to finish:

  Step 1: Download the PTB-XL ECG dataset (skipped if already downloaded)
  Step 2: Extract 309 features from the raw ECG signals
  Step 3: Select best features (remove redundant/noisy ones)
  Step 4: Scale the features so all values are in a similar range
  Step 5: Train three classifiers (SVM, Random Forest, K-Means) and evaluate them
  Step 6: Generate comparison plots and save reports to the results/ folder

HOW TO RUN:
  Make sure your virtual environment is active, then run:

    python main.py

  The first run will download the dataset (~1.7 GB), which takes a few minutes.
  Subsequent runs skip the download and go straight to processing.

WHAT YOU'LL SEE:
  - Progress messages as each step completes
  - A summary table at the end comparing all three classifiers
  - Plots and reports saved in the results/ folder
"""

import time

import numpy as np

# Import our custom modules from the src/ package.
# Each module handles one part of the pipeline.
from src.data_loader import load_dataset
from src.feature_extraction import extract_features, select_features
from src.classifiers import scale_features, train_svm, train_random_forest, train_kmeans
from src.visualization import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_feature_distributions,
    plot_feature_importance,
    plot_kmeans_clusters,
    save_classification_reports,
)


def main():
    """
    Run the complete heart disease classification pipeline.

    This function orchestrates all the steps and prints progress
    along the way so you know what's happening.
    """
    print("=" * 60)
    print("  Heart Disease Classifier")
    print("  Dataset: PTB-XL (12-lead ECG recordings)")
    print("  Task: Normal vs Abnormal classification")
    print("=" * 60)

    # Record the start time so we can report total duration at the end
    overall_start = time.time()

    # ==================================================================
    # STEP 1: Load the dataset
    # ==================================================================
    # This downloads the PTB-XL dataset if it's not already on disk,
    # reads all 21,799 ECG recordings, parses their diagnostic labels,
    # and splits them into training, validation, and test sets.
    # ==================================================================
    print("\n[Step 1/5] Loading dataset...")
    step_start = time.time()

    data = load_dataset()

    elapsed = time.time() - step_start
    print(f"  Completed in {elapsed:.1f} seconds.\n")

    # Print a quick summary of what was loaded
    print(f"  Training samples:   {data['X_train'].shape[0]}")
    print(f"  Validation samples: {data['X_val'].shape[0]}")
    print(f"  Test samples:       {data['X_test'].shape[0]}")
    print(f"  Signal shape:       {data['X_train'].shape[1:]} "
          f"(time_steps x leads)")

    # Show the class balance (how many Normal vs Abnormal)
    n_normal = np.sum(data["y_train"] == 0)
    n_abnormal = np.sum(data["y_train"] == 1)
    print(f"  Training labels:    {n_normal} Normal, {n_abnormal} Abnormal")

    # ==================================================================
    # STEP 2: Extract features
    # ==================================================================
    # The raw ECG signals are 2D arrays (1000 timesteps x 12 leads).
    # Machine learning models need fixed-size 1D input, so we compute
    # 309 features per ECG record: signal stats, HRV, morphological,
    # wavelet, and frequency band energy features.
    # ==================================================================
    print("\n[Step 2/6] Extracting features from ECG signals...")
    step_start = time.time()

    X_train_feat = extract_features(data["X_train"])
    X_val_feat = extract_features(data["X_val"])
    X_test_feat = extract_features(data["X_test"])

    elapsed = time.time() - step_start
    print(f"  Feature vector shape: {X_train_feat.shape} (samples x features)")
    print(f"  Completed in {elapsed:.1f} seconds.")

    # ==================================================================
    # STEP 3: Feature selection
    # ==================================================================
    # With 309 features, many are redundant or noisy. We use Random
    # Forest importance + mutual information to keep only the most
    # useful features. This reduces noise and improves accuracy,
    # especially for K-Means.
    # ==================================================================
    print("\n[Step 3/6] Selecting best features...")
    step_start = time.time()

    X_train_feat, X_val_feat, X_test_feat, selected_names, selected_idx = \
        select_features(X_train_feat, data["y_train"], X_val_feat, X_test_feat)

    elapsed = time.time() - step_start
    print(f"  Reduced to {X_train_feat.shape[1]} features.")
    print(f"  Completed in {elapsed:.1f} seconds.")

    # ==================================================================
    # STEP 4: Scale features
    # ==================================================================
    # Standardize all features to have mean=0 and std=1.
    # This is important because some features (like spectral energy)
    # have much larger numbers than others (like mean voltage).
    # Without scaling, SVM would focus only on the large-numbered features.
    # ==================================================================
    print("\n[Step 4/6] Scaling features...")

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train_feat, X_val_feat, X_test_feat
    )
    print("  Done (fit on training data, applied to all splits).")

    # ==================================================================
    # STEP 5: Train and evaluate classifiers
    # ==================================================================
    # We train three different models and test them all on the same
    # test set so the comparison is fair.
    #
    # SVM and Random Forest are "supervised" -- they learn from labeled
    # examples. K-Means is "unsupervised" -- it groups data by similarity
    # without using labels.
    # ==================================================================
    print("\n[Step 5/6] Training and evaluating classifiers...")

    y_train = data["y_train"]
    y_test = data["y_test"]

    # --- 5a: Support Vector Machine ---
    step_start = time.time()
    svm_model, svm_results = train_svm(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    elapsed = time.time() - step_start
    print(f"    SVM accuracy: {svm_results['accuracy']:.4f} "
          f"(took {elapsed:.1f}s)")

    # --- 5b: Random Forest ---
    step_start = time.time()
    rf_model, rf_results = train_random_forest(
        X_train_scaled, y_train, X_test_scaled, y_test,
        feature_names=selected_names,
    )
    elapsed = time.time() - step_start
    print(f"    Random Forest accuracy: {rf_results['accuracy']:.4f} "
          f"(took {elapsed:.1f}s)")

    # --- 5c: K-Means Clustering ---
    step_start = time.time()
    km_model, km_results = train_kmeans(
        X_train_scaled, y_train, X_test_scaled, y_test, n_clusters=5
    )
    elapsed = time.time() - step_start
    print(f"    K-Means accuracy: {km_results['accuracy']:.4f} "
          f"(took {elapsed:.1f}s)")

    # Collect all results for comparison
    all_results = [svm_results, rf_results, km_results]

    # ==================================================================
    # STEP 5: Generate visualizations and reports
    # ==================================================================
    # Create plots that help you understand and present the results.
    # All files are saved to the results/ folder.
    # ==================================================================
    print("\n[Step 6/6] Generating visualizations and reports...")

    # Confusion matrices for the two supervised models
    plot_confusion_matrix(svm_results, "confusion_matrix_svm.png")
    plot_confusion_matrix(rf_results, "confusion_matrix_rf.png")

    # Side-by-side bar chart comparing all three models
    plot_model_comparison(all_results)

    # Histograms showing how features differ between Normal and Abnormal
    plot_feature_distributions(X_train_feat, y_train, selected_names)

    # Feature importance chart from Random Forest (top 20 features)
    plot_feature_importance(rf_results)

    # K-Means cluster distribution chart
    plot_kmeans_clusters(km_results)

    # Detailed text reports
    save_classification_reports(all_results)

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    total_time = time.time() - overall_start

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<18} {'Type':<15} {'Accuracy':>9} {'F1':>9}")
    print("  " + "-" * 51)

    for r in all_results:
        model_type = "unsupervised" if r["model_name"] == "K-Means" else "supervised"
        print(f"  {r['model_name']:<18} {model_type:<15} "
              f"{r['accuracy']:>8.4f} {r['f1']:>9.4f}")

    print(f"\n  Total time: {total_time:.1f} seconds")
    print(f"  Results saved to: results/")
    print("=" * 60)


# This block runs only when you execute "python main.py" directly.
# It does NOT run if another file imports something from this file.
if __name__ == "__main__":
    main()
