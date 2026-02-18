"""
classifiers.py -- Train and evaluate three machine learning approaches.

This module implements three different ways to classify ECG signals.
All three work on HANDCRAFTED FEATURES extracted by feature_extraction.py
(309 numbers like mean, std, wavelet energy, etc.).

A fourth approach (1D CNN) lives in cnn_model.py. The CNN is different
because it works on RAW signals and learns its own features automatically.

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

Why compare all four?
   SVM and Random Forest use labels (supervised) and should perform well.
   K-Means doesn't use labels (unsupervised) and will likely perform worse.
   The CNN uses deep learning on raw signals -- a fundamentally different approach.
   This comparison shows the value of labeled data AND automatic feature learning.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
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

    IMPROVEMENTS OVER A BASIC SVM:
    1. PCA: Reduces correlated features to fewer, independent dimensions.
       This removes noise from distance calculations and speeds up training.
    2. GridSearchCV: Tries multiple hyperparameter combinations and picks
       the best one using 3-fold cross-validation on the training set.
    3. class_weight='balanced': Adjusts for class imbalance so the model
       doesn't just predict the majority class.
    4. Training cap: Limits to 10,000 samples for speed (SVM training
       time grows roughly as the square of the number of samples).

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

    # --- Step 1: PCA dimensionality reduction ---
    # Reduce features to the number of components that retain 95% of variance.
    # This removes correlated features and speeds up SVM significantly.
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"    PCA: {X_train.shape[1]} features -> {X_train_pca.shape[1]} components "
          f"(95% variance retained)")

    # --- Step 2: Limit training data for speed ---
    # SVM training time grows as O(n^2), so 17,000 samples would be very slow.
    max_train_samples = 10000
    if len(X_train_pca) > max_train_samples:
        print(f"    Using {max_train_samples} of {len(X_train_pca)} samples "
              f"(SVM is slow on large datasets).")
        rng = np.random.RandomState(42)
        indices = rng.choice(len(X_train_pca), max_train_samples, replace=False)
        X_train_sub = X_train_pca[indices]
        y_train_sub = y_train[indices]
    else:
        X_train_sub = X_train_pca
        y_train_sub = y_train

    # --- Step 3: GridSearchCV for hyperparameter tuning ---
    # Instead of guessing C and gamma, try multiple combinations and pick
    # the best one using cross-validation (splitting training data 3 ways).
    #
    # C: controls the trade-off between fitting training data perfectly
    #    (high C) and keeping the margin wide (low C).
    # gamma: controls how far each training point's influence reaches.
    #    'scale' = 1/(n_features * var), 'auto' = 1/n_features.
    #    Small gamma = wide influence (smoother boundary).
    #    Large gamma = narrow influence (more complex boundary).
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.001],
    }

    # class_weight='balanced' makes the model pay equal attention to both
    # classes, even if one has more samples. It multiplies the penalty for
    # misclassifying the minority class by (n_samples / (2 * n_minority)).
    base_svm = SVC(kernel="rbf", class_weight="balanced", random_state=42)

    print("    Running GridSearchCV (4x4 = 16 combinations, 3-fold CV)...")
    grid_search = GridSearchCV(
        base_svm, param_grid,
        cv=3,              # 3-fold cross-validation
        scoring='f1_weighted',  # Optimize for weighted F1 (handles imbalance)
        n_jobs=-1,          # n_jobs=-1 crashes on Windows Store Python 3.13
        refit=True,        # Retrain best model on full training subset
    )
    grid_search.fit(X_train_sub, y_train_sub)

    best_params = grid_search.best_params_
    print(f"    Best params: C={best_params['C']}, gamma={best_params['gamma']}")
    print(f"    Best CV F1: {grid_search.best_score_:.4f}")

    # The best model is already trained (refit=True)
    svm_model = grid_search.best_estimator_

    # Make predictions on the test set (using PCA-transformed test data)
    y_pred = svm_model.predict(X_test_pca)

    # Calculate how well it did
    results = compute_metrics(y_test, y_pred, "SVM")
    results["best_params"] = best_params
    results["pca_components"] = X_train_pca.shape[1]
    return svm_model, results


# =============================================================================
# RANDOM FOREST CLASSIFIER
# =============================================================================

def train_random_forest(X_train, y_train, X_test, y_test, feature_names=None):
    """
    Train a Random Forest classifier.

    HOW RANDOM FOREST WORKS (simplified):
    1. Create 500 "decision trees" (like flowcharts of yes/no questions).
    2. Each tree is trained on a random subset of the data and features.
       This randomness prevents the trees from all making the same mistakes.
    3. To classify a new ECG, every tree votes "Normal" or "Abnormal".
    4. The final answer is the majority vote.

    IMPROVEMENTS OVER A BASIC RANDOM FOREST:
    1. 500 trees (up from 100): more votes = more stable predictions.
    2. No max_depth cap, but min_samples_leaf=5: trees grow as deep as
       needed but can't memorize individual samples.
    3. class_weight='balanced': adjusts for class imbalance.
    4. Extracts feature importances for visualization.

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
    feature_names : list of str, optional
        Names of each feature (for importance ranking).

    Returns
    -------
    model   : The trained Random Forest model.
    results : dict with accuracy, precision, recall, F1, feature importances, etc.
    """
    print("  Training Random Forest classifier...")

    # Create and train the Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=500,        # 500 trees for stable predictions
        max_depth=None,          # Let trees grow as deep as needed
        min_samples_leaf=5,      # But require at least 5 samples per leaf
        class_weight="balanced", # Handle class imbalance
        random_state=42,         # Reproducible results
        n_jobs=-1,                # n_jobs=-1 crashes on Windows Store Python 3.13
    )
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # Calculate how well it did
    results = compute_metrics(y_test, y_pred, "Random Forest")

    # Store feature importances for visualization
    # Each tree tracks which features reduced prediction errors the most.
    # feature_importances_ averages this across all 500 trees.
    importances = rf_model.feature_importances_
    results["feature_importances"] = importances
    if feature_names is not None:
        results["feature_names"] = feature_names

    return rf_model, results


# =============================================================================
# K-MEANS CLUSTERING
# =============================================================================

def train_kmeans(X_train, y_train, X_test, y_test, n_clusters=5):
    """
    Perform K-Means clustering (unsupervised).

    HOW K-MEANS WORKS (simplified):
    1. Pick N random points as initial "cluster centers".
    2. Assign every ECG record to the nearest center.
    3. Move each center to the average position of its assigned records.
    4. Repeat steps 2-3 until the centers stop moving.

    IMPORTANT DIFFERENCE FROM SVM AND RANDOM FOREST:
    K-Means does NOT look at the labels during training. It groups data
    purely by how similar the features are. After it finds groups,
    we check: does each group mostly contain Normal or Abnormal ECGs?
    This lets us compare its accuracy with the supervised methods.

    IMPROVEMENTS OVER BASIC K-MEANS:
    1. PCA to 20 components: K-Means uses Euclidean distance, which
       breaks down in high dimensions (the "curse of dimensionality").
       Reducing to 20 components concentrates the signal and removes
       noise, giving K-Means a much better chance.
    2. 5 clusters instead of 2: The data may have sub-groups (different
       types of abnormalities). More clusters lets K-Means find finer
       structure, then we map each cluster to Normal or Abnormal.

    K-Means will still likely perform worse than SVM and Random Forest.

    Parameters
    ----------
    X_train  : np.ndarray -- Training features (labels NOT used for training).
    y_train  : np.ndarray -- Training labels (used only for cluster mapping).
    X_test   : np.ndarray -- Test features.
    y_test   : np.ndarray -- Test labels (for evaluation).
    n_clusters : int -- Number of clusters (default: 5).

    Returns
    -------
    model   : The trained K-Means model.
    results : dict with accuracy, precision, recall, F1, etc.
    """
    print("  Running K-Means clustering...")

    # --- Step 1: PCA dimensionality reduction ---
    # Reduce to 20 components so K-Means distance calculations are meaningful.
    n_components = min(20, X_train.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"    PCA: {X_train.shape[1]} features -> {n_components} components")

    # --- Step 2: K-Means clustering ---
    # n_init=10: run the algorithm 10 times with different starting points
    #            and keep the best result (K-Means is sensitive to starting points)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_train_pca)
    print(f"    Using {n_clusters} clusters (mapped to binary labels)")

    # Assign each test record to a cluster
    cluster_labels = kmeans.predict(X_test_pca)

    # --- Step 3: Map clusters to binary labels ---
    # For each cluster, find which real label (Normal or Abnormal) appears most.
    train_clusters = kmeans.predict(X_train_pca)
    cluster_to_label = {}

    for cluster_id in range(n_clusters):
        mask = train_clusters == cluster_id
        if mask.sum() > 0:
            labels_in_cluster = y_train[mask]
            values, counts = np.unique(labels_in_cluster, return_counts=True)
            cluster_to_label[cluster_id] = values[np.argmax(counts)]
        else:
            cluster_to_label[cluster_id] = 0  # Default to Normal if empty

    # Convert cluster numbers to predicted labels using our mapping
    y_pred = np.array([cluster_to_label[c] for c in cluster_labels])

    # Calculate how well it did
    results = compute_metrics(y_test, y_pred, "K-Means")
    results["cluster_to_label_mapping"] = cluster_to_label
    results["raw_cluster_labels"] = cluster_labels
    results["y_true"] = y_test
    results["pca_components"] = n_components
    return kmeans, results


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def compute_metrics(y_true, y_pred, model_name):
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
