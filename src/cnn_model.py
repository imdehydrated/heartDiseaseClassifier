"""
cnn_model.py -- 1D Convolutional Neural Network for ECG Classification

This module takes a fundamentally different approach from the SVM, Random
Forest, and K-Means classifiers in classifiers.py. Instead of using
handcrafted features (mean, std, wavelet energy, etc.), the CNN learns
its own features directly from the raw ECG signals.

HOW IT WORKS:
  1. Raw ECG signals (12 leads x 1000 timesteps) go in
  2. Each lead is normalized to zero mean and unit std (removes voltage
     scale differences between patients/equipment)
  3. During training, random augmentations (noise, scaling, time shift)
     are applied to reduce overfitting
  4. Convolutional layers slide small filters across the signal to detect
     patterns like QRS complexes, ST deviations, and rhythm irregularities
  5. Pooling layers compress the signal, keeping only the strongest patterns
  6. Fully connected layers combine the learned patterns to make a prediction
  7. Validation logits are converted to probabilities with deterministic
     test-time augmentation (TTA), then a decision threshold is selected
     to maximize Abnormal recall while enforcing a Normal specificity floor

WHY THIS MATTERS:
  - The handcrafted features in feature_extraction.py took hundreds of lines
    of domain-specific code to design
  - The CNN discovers similar (and sometimes better) features automatically
  - This is the key advantage of deep learning: it replaces manual feature
    engineering with learned representations

ARCHITECTURE:
  3 plain convolutional blocks + classifier head (~131k parameters)
  Input:  (batch, 12 leads, 1000 timesteps)
  Output: (batch, 1) -- raw logit (sigmoid applied at inference time)
  Kernels: 7/5/3 (large->small), Pooling: 2/2/global

USAGE:
  Called from main.py after the classical classifiers. Uses raw signals
  directly from load_dataset(), NOT the extracted/selected features.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix

from src.classifiers import compute_metrics


# =============================================================================
# DATASET
# =============================================================================

class ECGDataset(Dataset):
    """
    PyTorch Dataset that wraps raw ECG numpy arrays.

    PyTorch models need data in a specific format. This class converts
    our numpy arrays into PyTorch tensors and rearranges the dimensions
    so Conv1d can process them.

    The data_loader.py gives us shape (n_samples, 1000, 12) -- time first.
    Conv1d needs shape (n_samples, 12, 1000) -- channels first.

    Each recording is normalized per-lead to zero mean and unit std.
    This removes voltage scale differences between patients/equipment
    so the CNN can focus on waveform shape rather than absolute voltage.

    During training, random augmentations (noise, amplitude scaling,
    time shift) are applied to reduce overfitting.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 1000, 12)
        Raw ECG signals.
    y : np.ndarray of shape (n_samples,)
        Binary labels (0=Normal, 1=Abnormal).
    augment : bool, default False
        Whether to apply random augmentations. Set True for training only.
    """

    def __init__(self, X, y, augment=False):
        # Transpose from (n, 1000, 12) to (n, 12, 1000) -- channels first
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.augment = augment

        # --- Per-lead normalization ---
        # Each lead in each recording gets its own mean and std.
        # Shape of mean/std: (n, 12, 1) -- one value per lead per recording.
        # The + 1e-8 prevents division by zero for flat signals.
        mean = self.X.mean(dim=2, keepdim=True)
        std = self.X.std(dim=2, keepdim=True) + 1e-8
        self.X = (self.X - mean) / std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            # Gaussian noise: adds small random jitter (like electrical noise)
            x = x + torch.randn_like(x) * 0.05

            # Amplitude scaling: randomly scale voltage by 0.8x to 1.2x
            # (simulates different equipment gain settings)
            scale = 0.8 + torch.rand(1) * 0.4  # uniform in [0.8, 1.2]
            x = x * scale

            # Time shift: roll the signal left or right by up to 50 samples
            # (simulates slightly different recording start times)
            shift = torch.randint(-50, 51, (1,)).item()
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=1)

        return x, y


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class ECGConvNet(nn.Module):
    """
    1D Convolutional Neural Network for ECG classification.

    This network processes raw 12-lead ECG signals and learns to classify
    them as Normal or Abnormal. It uses three plain convolutional blocks to
    extract features, then two fully connected layers to make the decision
    (~131k parameters).

    Layer-by-layer (for a batch of 64 samples):

      Input:  (64, 12, 1000)   -- 12 ECG leads, 1000 timesteps

      Block 1: Conv(12->32, kernel=7) -> BatchNorm -> ReLU -> MaxPool(2) -> Dropout
               Output: (64, 32, 500)
               Kernel=7 covers 70ms at 100Hz, enough to see most of a QRS spike.

      Block 2: Conv(32->64, kernel=5) -> BatchNorm -> ReLU -> MaxPool(2) -> Dropout
               Output: (64, 64, 250)
               Finds finer patterns in the already-compressed features.

      Block 3: Conv(64->128, kernel=3) -> BatchNorm -> ReLU -> AdaptiveAvgPool(1)
               Output: (64, 128, 1)
               Small kernel + global pooling collapses temporal dimension into
               a fixed 128-dimensional summary of the full recording.

      Flatten: (64, 128)

      Head: Dropout(0.5) -> Linear(128,64) -> ReLU -> Dropout(0.3) -> Linear(64,1)
            Output: (64, 1) -- one logit per sample

    The output is a raw logit (not a probability). We use BCEWithLogitsLoss
    during training, which applies sigmoid internally for numerical stability.
    """

    def __init__(self):
        super().__init__()

        # --- Block 1: 12 ECG leads -> 32 feature maps ---
        # kernel=7 covers 70ms at 100Hz. MaxPool(2) halves the time dimension.
        self.block1 = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
        )

        # --- Block 2: 32 -> 64 feature maps ---
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
        )

        # --- Block 3: 64 -> 128 feature maps ---
        # AdaptiveAvgPool1d(1) collapses the time dimension to a single value
        # per channel, producing a fixed-size 128-dim vector regardless of
        # input length.
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # --- Classifier head ---
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        Forward pass: raw ECG signal in, classification logit out.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, 12, 1000)

        Returns
        -------
        torch.Tensor of shape (batch, 1) -- raw logit.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Flatten from (batch, 128, 1) to (batch, 128)
        x = x.squeeze(2)

        x = self.classifier(x)
        return x


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def _compute_abnormal_recall_and_normal_specificity(y_true, y_pred):
    """
    Compute clinical metrics from binary predictions.

    Label 1 is Abnormal (positive class), label 0 is Normal.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    abnormal_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    normal_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return abnormal_recall, normal_specificity


def _predict_probs(model, data_loader, device, use_tta=True):
    """
    Predict probabilities with optional deterministic test-time augmentation.

    TTA uses a fixed set of small transformations and averages logits.
    Averaging logits (before sigmoid) is more stable than averaging probs.
    """
    tta_views = [
        (0, 1.00),   # identity
        (-12, 1.00),  # small left shift
        (12, 1.00),  # small right shift
        (0, 0.95),   # slight down-scale
        (0, 1.05),   # slight up-scale
    ]
    if not use_tta:
        tta_views = [(0, 1.00)]

    model.eval()
    all_probs = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)

            view_logits = []
            for shift, scale in tta_views:
                x_view = X_batch
                if shift != 0:
                    x_view = torch.roll(x_view, shifts=shift, dims=2)
                if scale != 1.0:
                    x_view = x_view * scale

                logits = model(x_view).squeeze(1)
                view_logits.append(logits)

            logits_mean = torch.stack(view_logits, dim=0).mean(dim=0)
            probs = torch.sigmoid(logits_mean)

            all_probs.extend(probs.cpu().numpy())
            all_true.extend(y_batch.numpy())

    return np.array(all_probs), np.array(all_true), len(tta_views)


def _find_best_threshold(y_true, y_prob, specificity_floor=0.80, relax_step=0.02):
    """
    Select threshold that maximizes Abnormal recall under specificity constraint.

    Policy:
      1) Maximize Abnormal recall where Normal specificity >= floor.
      2) If no threshold is feasible, relax floor by fixed steps.
      3) If still no feasible threshold, use unconstrained best recall.

    Tie-breaker:
      - If recall ties, choose threshold closest to 0.50.
    """
    thresholds = np.arange(0.10, 0.901, 0.01)
    floor = float(specificity_floor)
    mode = "feasible"

    def _get_candidates(current_floor):
        candidates = []
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            abnormal_recall, normal_specificity = (
                _compute_abnormal_recall_and_normal_specificity(y_true, y_pred)
            )
            if normal_specificity >= current_floor:
                candidates.append(
                    (float(threshold), abnormal_recall, normal_specificity)
                )
        return candidates

    candidates = _get_candidates(floor)

    while not candidates and floor > 0.0:
        floor = max(0.0, floor - relax_step)
        mode = "relaxed"
        candidates = _get_candidates(floor)

    if not candidates:
        mode = "unconstrained"
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            abnormal_recall, normal_specificity = (
                _compute_abnormal_recall_and_normal_specificity(y_true, y_pred)
            )
            candidates.append((float(threshold), abnormal_recall, normal_specificity))

    best_threshold, best_recall, best_specificity = sorted(
        candidates, key=lambda x: (-x[1], abs(x[0] - 0.50))
    )[0]

    return {
        "threshold": best_threshold,
        "abnormal_recall": best_recall,
        "normal_specificity": best_specificity,
        "requested_specificity_floor": float(specificity_floor),
        "effective_specificity_floor": floor,
        "selection_mode": mode,
        "selection_policy": "max_recall_at_specificity_floor",
    }


def train_cnn(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train a 1D CNN on raw ECG signals and evaluate on the test set.

    Unlike SVM and Random Forest which need the 309 handcrafted features,
    the CNN works directly on raw signals. It learns its own features
    through the convolutional layers.

    The training loop:
      1. Feed batches of ECG signals through the network
      2. Compare predictions to true labels (compute loss)
      3. Adjust the network's weights to reduce loss (backpropagation)
      4. After each epoch, check performance on the validation set
      5. If validation F1 stops improving, stop early to prevent overfitting
      6. After training, select a validation threshold that maximizes
         Abnormal recall while enforcing a Normal specificity floor
      7. Apply deterministic TTA and the selected threshold on test data

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, 1000, 12)
        Raw training ECG signals.
    y_train : np.ndarray of shape (n_train,)
        Training labels (0=Normal, 1=Abnormal).
    X_val : np.ndarray of shape (n_val, 1000, 12)
        Raw validation ECG signals.
    y_val : np.ndarray of shape (n_val,)
        Validation labels.
    X_test : np.ndarray of shape (n_test, 1000, 12)
        Raw test ECG signals.
    y_test : np.ndarray of shape (n_test,)
        Test labels.

    Returns
    -------
    model : ECGConvNet
        The trained CNN model (with best validation weights loaded).
    results : dict
        Same format as classifiers.py results plus CNN-specific fields:
        decision_threshold, selection_policy, specificity_floor,
        val_abnormal_recall, val_normal_specificity, tta_enabled,
        tta_views, y_prob.
    """
    print("  Training 1D CNN classifier...")

    # ------------------------------------------------------------------
    # 1. Device selection -- use GPU if available, otherwise CPU
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Using device: {device}")

    # ------------------------------------------------------------------
    # 2. Handle NaN values in raw signals (rare but possible)
    # ------------------------------------------------------------------
    if np.any(np.isnan(X_train)):
        X_train = np.nan_to_num(X_train, nan=0.0)
    if np.any(np.isnan(X_val)):
        X_val = np.nan_to_num(X_val, nan=0.0)
    if np.any(np.isnan(X_test)):
        X_test = np.nan_to_num(X_test, nan=0.0)

    # ------------------------------------------------------------------
    # 3. Set random seeds for reproducibility
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # ------------------------------------------------------------------
    # 4. Create PyTorch Datasets and DataLoaders
    # ------------------------------------------------------------------
    # num_workers=0 avoids Windows multiprocessing issues with PyTorch.
    # The data is small enough to load quickly without parallel workers.
    train_dataset = ECGDataset(X_train, y_train, augment=True)
    val_dataset = ECGDataset(X_val, y_val, augment=False)
    test_dataset = ECGDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=0
    )

    # ------------------------------------------------------------------
    # 5. Create model, loss function, optimizer, and scheduler
    # ------------------------------------------------------------------
    model = ECGConvNet().to(device)

    # Print how many learnable parameters the model has
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Model parameters: {total_params:,}")

    # Class weight: if there are more Normal than Abnormal samples,
    # pos_weight > 1 makes the model pay more attention to Abnormal cases.
    # This is the same idea as class_weight='balanced' in sklearn.
    n_normal = np.sum(y_train == 0)
    n_abnormal = np.sum(y_train == 1)
    pos_weight = torch.tensor(
        [n_normal / n_abnormal], dtype=torch.float32
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-4
    )

    # Reduce learning rate when validation F1 plateaus.
    # After 3 epochs without improvement, halve the learning rate.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # ------------------------------------------------------------------
    # 6. Training loop with early stopping
    # ------------------------------------------------------------------
    max_epochs = 30
    max_patience = 7  # Stop if no improvement for 7 epochs
    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"    Training for up to {max_epochs} epochs "
          f"(early stopping patience: {max_patience})...")

    for epoch in range(max_epochs):

        # --- Train phase ---
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch).squeeze(1)  # (batch,)
            loss = criterion(logits, y_batch)
            loss.backward()

            # Clip gradients to prevent exploding gradients.
            # If any gradient exceeds max_norm=1.0, all gradients are
            # scaled down proportionally. This stabilizes training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item() * len(y_batch)

        train_loss = running_loss / len(train_dataset)

        # --- Validation phase ---
        model.eval()
        val_preds = []
        val_true = []
        val_running_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch).squeeze(1)
                loss = criterion(logits, y_batch)
                val_running_loss += loss.item() * len(y_batch)

                # Convert logits to predictions (sigmoid > 0.5)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()

                val_preds.extend(preds.cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())

        val_loss = val_running_loss / len(val_dataset)
        val_preds = np.array(val_preds)
        val_true = np.array(val_true)
        val_f1 = f1_score(val_true, val_preds, average="weighted")

        # --- Epoch summary ---
        print(f"    Epoch {epoch + 1:>2}/{max_epochs} -- "
              f"Train loss: {train_loss:.4f}, "
              f"Val loss: {val_loss:.4f}, "
              f"Val F1: {val_f1:.4f}")

        # --- Learning rate scheduler ---
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_f1)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"    -> Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

        # --- Early stopping ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"    Early stopping at epoch {epoch + 1} "
                      f"(no improvement for {max_patience} epochs)")
                break

    # ------------------------------------------------------------------
    # 7. Load best model, calibrate threshold, and evaluate on test set
    # ------------------------------------------------------------------
    print(f"    Best validation F1: {best_val_f1:.4f}")

    model.load_state_dict(best_model_state)
    model.eval()

    # Use deterministic TTA on validation to choose threshold with
    # a clinical safety constraint (specificity floor).
    val_probs, val_true, tta_view_count = _predict_probs(
        model, val_loader, device, use_tta=True
    )
    threshold_info = _find_best_threshold(
        y_true=val_true,
        y_prob=val_probs,
        specificity_floor=0.80,
        relax_step=0.02,
    )
    decision_threshold = threshold_info["threshold"]
    print(
        "    Threshold policy: "
        f"{threshold_info['selection_policy']} "
        f"(requested specificity floor="
        f"{threshold_info['requested_specificity_floor']:.2f}, "
        f"effective floor={threshold_info['effective_specificity_floor']:.2f}, "
        f"mode={threshold_info['selection_mode']})"
    )
    print(
        f"    Selected threshold: {decision_threshold:.2f} "
        f"(val abnormal recall={threshold_info['abnormal_recall']:.4f}, "
        f"val normal specificity={threshold_info['normal_specificity']:.4f})"
    )
    if threshold_info["selection_mode"] != "feasible":
        print("    Warning: exact specificity floor was not feasible.")

    # Run deterministic TTA on test set and apply selected threshold.
    test_probs, _, _ = _predict_probs(model, test_loader, device, use_tta=True)
    test_preds = (test_probs >= decision_threshold).astype(int)

    # Use the same metrics function as SVM, RF, and K-Means
    results = compute_metrics(y_test, test_preds, "1D CNN")
    results["decision_threshold"] = decision_threshold
    results["selection_policy"] = threshold_info["selection_policy"]
    results["specificity_floor"] = threshold_info["requested_specificity_floor"]
    results["effective_specificity_floor"] = (
        threshold_info["effective_specificity_floor"]
    )
    results["selection_mode"] = threshold_info["selection_mode"]
    results["val_abnormal_recall"] = threshold_info["abnormal_recall"]
    results["val_normal_specificity"] = threshold_info["normal_specificity"]
    results["tta_enabled"] = True
    results["tta_views"] = tta_view_count
    results["y_prob"] = test_probs

    return model, results
