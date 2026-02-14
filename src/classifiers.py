"""
classifiers.py -- Train and evaluate three machine learning approaches.

This module implements three different ways to classify ECG signals:

1. SVM (Support Vector Machine) -- SUPERVISED
   Think of it like drawing a line (or curve) on a graph that separates
   two groups of dots. SVM finds the BEST line -- the one that has the
   most space between it and the nearest dots on each side.
   "Supervised" means we give it labeled examples to learn from.

2. Random Forest -- SUPERVISED
   Imagine asking 100 people to guess if an ECG is normal or abnormal.
   Each person only sees part of the data and uses simple yes/no questions
   (like "Is the heart rate above 80?"). The final answer is whatever
   the majority votes for. Random Forest does exactly this with 100
   "decision trees" instead of people.

3. K-Means Clustering -- UNSUPERVISED
   Imagine dumping all ECG records on a table and grouping them by
   similarity -- WITHOUT knowing which ones are normal or abnormal.
   K-Means does this automatically: it picks 2 center points, assigns
   each record to the nearest center, then adjusts the centers.
   After clustering, we check if the groups it found match the real labels.
   "Unsupervised" means it does NOT use the labels to learn.

Why compare all three?
   SVM and Random Forest use labels (supervised) and should perform well.
   K-Means doesn't use labels (unsupervised) and will likely perform worse.
   This comparison shows the value of having labeled data.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# =============================================================================
# FEATURE SCALING
# =============================================================================

def scale_features(X_train, X_val, X_test):
    """
    Standardize features so they all have a similar range.

    WHY IS THIS NEEDED?
    Our 120 features have very different ranges. For example:
      - "mean" might range from -0.5 to 0.5
      - "spectral_energy" might range from 0 to 1,000,000

    SVM treats all numbers equally, so it would think spectral_energy
    is way more important just because its numbers are bigger.
    Scaling fixes this by making every feature have:
      - Mean = 0  (centered at zero)
      - Std  = 1  (spread out by the same amount)

    IMPORTANT: We fit the scaler on training data ONLY.
    Think of it this way: the test set is like a final exam. You can't
    look at the exam questions while studying. If we used test data to
    calculate the mean and std, we'd be "cheating" -- this is called
    "data leakage" and gives unrealistically good results.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray of shape (n_samples, 120)
        Feature arrays for each data split.

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled : np.ndarray
        Scaled feature arrays (same shape as input).
    scaler : StandardScaler
        The fitted scaler object (saved in case we need it later).
    """
    scaler = StandardScaler()

    # fit_transform: learns the mean & std from training data, then scales it
    X_train_scaled = scaler.fit_transform(X_train)

    # transform: uses the SAME mean & std from training to scale val/test
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# =============================================================================
# SVM CLASSIFIER
# =============================================================================

def train_svm(X_train, y_train, X_test, y_test):
    """
    Train a Support Vector Machine (SVM) classifier.

    HOW SVM WORKS (simplified):
    Imagine plotting all ECG records on a graph where each axis is a feature.
    Normal hearts cluster in one area, abnormal hearts in another.
    SVM finds the boundary between them that:
      1. Correctly separates the two groups
      2. Maximizes the "margin" (empty space) around the boundary

    We use an "RBF kernel" which allows the boundary to be curved
    (not just a straight line). This is important because real data
    rarely separates perfectly with a straight line.

    NOTE ON SPEED:
    SVM gets very slow with large datasets (training time grows roughly
    as the square of the number of samples). With 17,000+ samples,
    training could take hours. So we limit training to 10,000 samples
    to keep it under a few minutes.

    Parameters
    ----------
    X_train : np.ndarray -- Training features (scaled).
    y_train : np.ndarray -- Training labels (0=Normal, 1=Abnormal).
    X_test  : np.ndarray -- Test features (scaled).
    y_test  : np.ndarray -- Test labels.

    Returns
    -------
    model   : The trained SVM model.
    results : dict with accuracy, precision, recall, F1, etc.
    """
    print("  Training SVM classifier...")

    # Limit training data to 10,000 samples for speed
    max_train_samples = len(X_train)
    if len(X_train) > max_train_samples:
        print(f"    Using {max_train_samples} of {len(X_train)} samples "
              f"(SVM is slow on large datasets).")
        # RandomState(42) ensures we always pick the same random samples
        rng = np.random.RandomState(42)
        indices = rng.choice(len(X_train), max_train_samples, replace=False)
        X_train_sub = X_train[indices]
        y_train_sub = y_train[indices]
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    # Create and train the SVM model
    # kernel='rbf': use a curved boundary (Radial Basis Function)
    # C=1.0: balance between fitting training data and keeping margin wide
    # random_state=42: makes results reproducible (same answer every run)
    svm_model = SVC(kernel="rbf", C=1.0, random_state=42)
    svm_model.fit(X_train_sub, y_train_sub)

    # Make predictions on the test set
    y_pred = svm_model.predict(X_test)

    # Calculate how well it did
    results = _compute_metrics(y_test, y_pred, "SVM")
    return svm_model, results


# =============================================================================
# RANDOM FOREST CLASSIFIER
# =============================================================================

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest classifier.

    HOW RANDOM FOREST WORKS (simplified):
    1. Create 100 "decision trees" (like flowcharts of yes/no questions).
    2. Each tree is trained on a random subset of the data and features.
       This randomness prevents the trees from all making the same mistakes.
    3. To classify a new ECG, every tree votes "Normal" or "Abnormal".
    4. The final answer is the majority vote.

    WHY IT'S GOOD FOR BEGINNERS:
    - Works well "out of the box" without much tuning
    - Handles different feature scales (unlike SVM)
    - Rarely overfits (memorizes training data instead of learning patterns)
    - Can tell you which features are most important

    Parameters
    ----------
    X_train : np.ndarray -- Training features.
    y_train : np.ndarray -- Training labels.
    X_test  : np.ndarray -- Test features.
    y_test  : np.ndarray -- Test labels.

    Returns
    -------
    model   : The trained Random Forest model.
    results : dict with accuracy, precision, recall, F1, etc.
    """
    print("  Training Random Forest classifier...")

    # Create and train the Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,   # Number of trees in the forest
        max_depth=20,       # Max depth of each tree (prevents overfitting)
        random_state=42,    # Reproducible results
        n_jobs=-1,          # Use ALL CPU cores for faster training
    )
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # Calculate how well it did
    results = _compute_metrics(y_test, y_pred, "Random Forest")
    return rf_model, results


# =============================================================================
# K-MEANS CLUSTERING
# =============================================================================

def train_kmeans(X_train, y_train, X_test, y_test, n_clusters=2):
    """
    Perform K-Means clustering (unsupervised).

    HOW K-MEANS WORKS (simplified):
    1. Pick 2 random points as initial "cluster centers".
    2. Assign every ECG record to the nearest center.
    3. Move each center to the average position of its assigned records.
    4. Repeat steps 2-3 until the centers stop moving.

    IMPORTANT DIFFERENCE FROM SVM AND RANDOM FOREST:
    K-Means does NOT look at the labels during training. It groups data
    purely by how similar the features are. After it finds 2 groups,
    we check: does Group 1 mostly contain Normal ECGs? If so, we label
    that group as "Normal". This lets us compare its accuracy with the
    supervised methods.

    K-Means will likely perform worse than SVM and Random Forest.

    Parameters
    ----------
    X_train  : np.ndarray -- Training features (labels NOT used for training).
    y_train  : np.ndarray -- Training labels (used only for cluster mapping).
    X_test   : np.ndarray -- Test features.
    y_test   : np.ndarray -- Test labels (for evaluation).
    n_clusters : int -- Number of clusters (2 for Normal vs Abnormal).

    Returns
    -------
    model   : The trained K-Means model.
    results : dict with accuracy, precision, recall, F1, etc.
    """
    print("  Running K-Means clustering...")

    # Create and train K-Means
    # n_init=10: run the algorithm 10 times with different starting points
    #            and keep the best result (K-Means is sensitive to starting points)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_train)

    # Assign each test record to a cluster (0 or 1)
    cluster_labels = kmeans.predict(X_test)

    # NOW we need to figure out which cluster corresponds to which label.
    # We do this by looking at the training data:
    # For each cluster, find which real label (Normal or Abnormal) appears most.
    train_clusters = kmeans.predict(X_train)
    cluster_to_label = {}

    for cluster_id in range(n_clusters):
        # Find all training samples assigned to this cluster
        mask = train_clusters == cluster_id
        if mask.sum() > 0:
            # What's the most common real label in this cluster?
            labels_in_cluster = y_train[mask]
            values, counts = np.unique(labels_in_cluster, return_counts=True)
            cluster_to_label[cluster_id] = values[np.argmax(counts)]
        else:
            cluster_to_label[cluster_id] = 0  # Default to Normal if empty

    # Convert cluster numbers to predicted labels using our mapping
    y_pred = np.array([cluster_to_label[c] for c in cluster_labels])

    # Calculate how well it did
    results = _compute_metrics(y_test, y_pred, "K-Means")
    results["cluster_to_label_mapping"] = cluster_to_label
    results["raw_cluster_labels"] = cluster_labels  # Raw cluster IDs (0 or 1)
    results["y_true"] = y_test                      # True labels for the test set
    return kmeans, results


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def _compute_metrics(y_true, y_pred, model_name):
    """
    Calculate how well a model performed.

    We compute four standard metrics:

    ACCURACY: What percentage of predictions were correct?
      Example: 85% accuracy = got 85 out of 100 right.

    PRECISION: When the model said "Abnormal", how often was it right?
      High precision = few false alarms.
      Example: 90% precision = 9 out of 10 "Abnormal" predictions were truly abnormal.

    RECALL: Of all the actually Abnormal ECGs, how many did the model catch?
      High recall = few missed cases.
      Example: 80% recall = caught 8 out of 10 abnormal hearts.

    F1 SCORE: A single number that balances precision and recall.
      It's the "harmonic mean" of precision and recall.
      Useful when you want one number to compare models.

    We also compute a CONFUSION MATRIX, which is a 2x2 table showing:
      - True Positives:  correctly identified as Abnormal
      - True Negatives:  correctly identified as Normal
      - False Positives: Normal ECG wrongly called Abnormal (false alarm)
      - False Negatives: Abnormal ECG wrongly called Normal (missed case)

    Parameters
    ----------
    y_true : np.ndarray -- The real labels.
    y_pred : np.ndarray -- The model's predictions.
    model_name : str -- Name of the model (for display purposes).

    Returns
    -------
    dict with all computed metrics.
    """
    return {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "report": classification_report(
            y_true, y_pred,
            target_names=["Normal", "Abnormal"],
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "y_pred": y_pred,
    }
