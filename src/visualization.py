"""
visualization.py -- Generate plots and save them to the results/ folder.

This module creates visual charts that help you understand how well
each classifier performed. All charts are saved as PNG image files
in the results/ folder.

Charts created:
  1. Confusion matrix heatmaps -- shows what each model got right/wrong
  2. Model comparison bar chart -- side-by-side accuracy/precision/recall/F1
  3. Feature distribution histograms -- how features differ between classes
  4. K-Means cluster distribution -- how K-Means grouped the data
  5. Text reports -- detailed classification reports saved as .txt files
"""

import os

import numpy as np
import matplotlib
# Use a non-interactive backend so plots are saved to files
# without needing a display window (works in any environment).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# =============================================================================
# CONSTANTS
# =============================================================================

# Where to save all output figures and reports.
# Same logic as data_loader.py: go up from src/ to project root, then add results/
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results"
)


def _ensure_results_dir():
    """Create the results/ folder if it doesn't exist yet."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(results, filename):
    """
    Plot a confusion matrix as a color-coded heatmap and save it.

    WHAT IS A CONFUSION MATRIX?
    It's a 2x2 table that shows exactly where the model got things
    right and wrong:

                        Predicted Normal    Predicted Abnormal
    Actually Normal     TRUE NEGATIVE       FALSE POSITIVE
    Actually Abnormal   FALSE NEGATIVE      TRUE POSITIVE

    - True Negative (top-left):  Normal ECG correctly predicted as Normal
    - False Positive (top-right): Normal ECG WRONGLY predicted as Abnormal
      (a "false alarm" -- the patient is fine but the model flagged them)
    - False Negative (bottom-left): Abnormal ECG WRONGLY predicted as Normal
      (this is dangerous -- the model missed a real problem!)
    - True Positive (bottom-right): Abnormal ECG correctly predicted as Abnormal

    The darker the blue, the higher the number in that cell.

    Parameters
    ----------
    results : dict
        The results dictionary from classifiers.py (must have 'confusion_matrix').
    filename : str
        Name of the output file, e.g., "confusion_matrix_svm.png".
    """
    _ensure_results_dir()

    # Create a figure (the canvas) and an axis (the plot area)
    fig, ax = plt.subplots(figsize=(6, 5))

    # Use sklearn's built-in confusion matrix display
    ConfusionMatrixDisplay(
        confusion_matrix=results["confusion_matrix"],
        display_labels=["Normal", "Abnormal"],
    ).plot(ax=ax, cmap="Blues")  # cmap="Blues" = use blue color gradient

    ax.set_title(f'{results["model_name"]} -- Confusion Matrix')
    plt.tight_layout()  # Prevent labels from being cut off
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close()  # Close the figure to free memory
    print(f"    Saved: {filename}")


# =============================================================================
# MODEL COMPARISON BAR CHART
# =============================================================================

def plot_model_comparison(all_results):
    """
    Create a grouped bar chart comparing all models on 4 metrics.

    This makes it easy to see at a glance which model performed best.
    Each model gets a different color, and the four metrics (accuracy,
    precision, recall, F1) are shown side by side.

    Parameters
    ----------
    all_results : list of dict
        A list of results dictionaries, one per model.
    """
    _ensure_results_dir()

    model_names = [r["model_name"] for r in all_results]
    metrics = ["accuracy", "precision", "recall", "f1"]

    # x positions for the groups of bars
    x = np.arange(len(metrics))
    width = 0.25  # Width of each bar

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, result in enumerate(all_results):
        values = [result[m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=result["model_name"])

        # Add the actual number on top of each bar
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # center horizontally
                bar.get_height() + 0.01,              # slightly above the bar
                f"{val:.3f}",                          # 3 decimal places
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Model Comparison: Normal vs Abnormal ECG Classification")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1.15)  # Scores range from 0 to 1; leave room for labels

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print("    Saved: model_comparison.png")


# =============================================================================
# FEATURE DISTRIBUTION HISTOGRAMS
# =============================================================================

def plot_feature_distributions(X_features, y_labels, feature_names):
    """
    Plot histograms showing how key features differ between classes.

    A histogram shows how often each value appears. If Normal and Abnormal
    ECGs have different distributions for a feature, that feature is useful
    for classification.

    We pick 4 features from Lead II (the most commonly used clinical lead)
    to keep things simple.

    Parameters
    ----------
    X_features : np.ndarray of shape (n_samples, 120)
        The extracted feature vectors.
    y_labels : np.ndarray of shape (n_samples,)
        Binary labels (0=Normal, 1=Abnormal).
    feature_names : list of str
        Names for all 120 features.
    """
    _ensure_results_dir()

    # Lead II is the second lead (index 1). Each lead has 10 features.
    # So Lead II features start at index 1 * 10 = 10.
    lead_ii_start = 10

    # Pick 4 interesting features from Lead II
    feature_indices = [
        lead_ii_start + 0,   # Mean (average voltage)
        lead_ii_start + 1,   # Std (signal variability)
        lead_ii_start + 4,   # Skewness (signal asymmetry)
        lead_ii_start + 8,   # Dominant frequency (main rhythm)
    ]

    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()  # Convert 2D grid to 1D list for easier looping

    for ax, feat_idx in zip(axes, feature_indices):
        # Separate the feature values by class
        normal_vals = X_features[y_labels == 0, feat_idx]
        abnormal_vals = X_features[y_labels == 1, feat_idx]

        # Plot overlapping histograms
        # alpha=0.6 makes them semi-transparent so you can see both
        ax.hist(normal_vals, bins=50, alpha=0.6, label="Normal", color="steelblue")
        ax.hist(abnormal_vals, bins=50, alpha=0.6, label="Abnormal", color="salmon")
        ax.set_title(feature_names[feat_idx])
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend()

    plt.suptitle(
        "Feature Distributions: Normal vs Abnormal (Lead II)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_distributions.png"), dpi=150)
    plt.close()
    print("    Saved: feature_distributions.png")


# =============================================================================
# K-MEANS CLUSTER VISUALIZATION
# =============================================================================

def plot_kmeans_clusters(results):
    """
    Show what's actually inside each K-Means cluster using a stacked bar chart.

    Each bar represents one cluster. The bar is split into two colors showing
    how many Normal vs Abnormal ECGs ended up in that cluster. This reveals
    whether K-Means found any meaningful grouping, even if the final accuracy
    is low.

    Parameters
    ----------
    results : dict
        K-Means results dictionary (must have 'raw_cluster_labels' and 'y_true').
    """
    _ensure_results_dir()

    cluster_labels = results["raw_cluster_labels"]  # Raw cluster IDs (0, 1, ...)
    y_true = results["y_true"]                      # Actual labels
    mapping = results.get("cluster_to_label_mapping", {})
    label_names = {0: "Normal", 1: "Abnormal"}

    unique_clusters = np.unique(cluster_labels)

    # Count Normal and Abnormal samples in each cluster
    normal_counts = []
    abnormal_counts = []
    bar_labels = []

    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        n_normal = np.sum(y_true[mask] == 0)
        n_abnormal = np.sum(y_true[mask] == 1)
        normal_counts.append(n_normal)
        abnormal_counts.append(n_abnormal)

        mapped = label_names.get(mapping.get(cluster_id, -1), "?")
        bar_labels.append(f"Cluster {cluster_id}\n(mapped to {mapped})")

    x = np.arange(len(unique_clusters))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))

    # Stacked bars: Normal on bottom, Abnormal on top
    ax.bar(x, normal_counts, width, label="Actually Normal",
           color="steelblue")
    ax.bar(x, abnormal_counts, width, bottom=normal_counts,
           label="Actually Abnormal", color="salmon")

    # Add count labels inside each bar section
    for i in range(len(unique_clusters)):
        if normal_counts[i] > 0:
            ax.text(i, normal_counts[i] / 2, str(normal_counts[i]),
                    ha="center", va="center", fontsize=10, fontweight="bold")
        if abnormal_counts[i] > 0:
            ax.text(i, normal_counts[i] + abnormal_counts[i] / 2,
                    str(abnormal_counts[i]),
                    ha="center", va="center", fontsize=10, fontweight="bold")

    ax.set_title("K-Means: What's Actually Inside Each Cluster")
    ax.set_ylabel("Number of Samples")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "kmeans_cluster_distribution.png"), dpi=150)
    plt.close()
    print("    Saved: kmeans_cluster_distribution.png")


# =============================================================================
# TEXT REPORTS
# =============================================================================

def save_classification_reports(all_results):
    """
    Save detailed text classification reports to files.

    Each report includes per-class precision, recall, F1, and overall metrics.
    These are useful for a deeper look beyond the bar chart.

    Parameters
    ----------
    all_results : list of dict
        A list of results dictionaries, one per model.
    """
    _ensure_results_dir()

    for result in all_results:
        # Create filename like: classification_report_svm.txt
        safe_name = result["model_name"].lower().replace(" ", "_")
        filename = f"classification_report_{safe_name}.txt"
        filepath = os.path.join(RESULTS_DIR, filename)

        with open(filepath, "w") as f:
            f.write(f"Classification Report: {result['model_name']}\n")
            f.write("=" * 50 + "\n\n")
            f.write(result["report"])
            f.write(f"\nOverall Metrics:\n")
            f.write(f"  Accuracy:  {result['accuracy']:.4f}\n")
            f.write(f"  Precision: {result['precision']:.4f}\n")
            f.write(f"  Recall:    {result['recall']:.4f}\n")
            f.write(f"  F1 Score:  {result['f1']:.4f}\n")

        print(f"    Saved: {filename}")
