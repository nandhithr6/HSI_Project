"""
HSI Model Training Script
- 5-fold cross-validation
- Joint spatial+spectral augmentations
- Unified Focal Loss
- AdamW optimizer, gradient clipping
- CosineAnnealingWarmRestarts scheduler with warmup
- Mixed-precision training
- Early stopping, checkpointing, ensemble
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from src import models
from src.models import HSIModel

# ===================== DATASET & AUGMENTATION =====================
class HSIDataset(Dataset):
    def __init__(self, data, labels, augment=True):
        self.data = data
        self.labels = labels
        self.augment = augment
        self.num_bands = data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.augment:
            x = self.spatial_spectral_augment(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def spatial_spectral_augment(self, x):
        # Spatial augmentations
        if random.random() < 0.5:
            x = np.flip(x, axis=2)  # Horizontal flip
        if random.random() < 0.5:
            x = np.flip(x, axis=3)  # Vertical flip
        if random.random() < 0.5:
            x = np.rot90(x, k=random.randint(1, 3), axes=(2, 3))
        # Elastic/grid distortions, brightness/contrast, noise
        if random.random() < 0.3:
            x = x + np.random.normal(0, 0.01, x.shape)
        if random.random() < 0.3:
            x = x * (0.9 + 0.2 * random.random())
        # Spectral-band dropout
        if random.random() < 0.2:
            drop_idx = np.random.choice(self.num_bands, size=int(self.num_bands * 0.1), replace=False)
            x[drop_idx, :, :] = 0
        return x

# ===================== UNIFIED FOCAL LOSS =====================
class UnifiedFocalLoss(nn.Module):
    def __init__(self, num_classes=5, lambda_=0.5, alpha=0.25, gamma=2.0, delta=0.5, smooth=1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_ = lambda_
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.smooth = smooth

    def forward(self, logits, targets):
        # Focal CE
        ce = nn.CrossEntropyLoss(reduction='none')(logits, targets)
        pt = torch.exp(-ce)
        focal_ce = self.alpha * (1 - pt) ** self.gamma * ce
        focal_ce = focal_ce.mean()
        # Focal Dice
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)
        intersection = (probs * targets_onehot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        focal_dice = self.alpha * (1 - dice) ** self.gamma * dice
        focal_dice = 1 - focal_dice.mean()
        # Unified loss
        return self.lambda_ * focal_ce + (1 - self.lambda_) * focal_dice

# ===================== TRAINING UTILS =====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop

# ===================== MAIN TRAINING LOOP =====================
def train_hsi_model(data, labels, num_classes=5, n_folds=5, seed=42):
    set_seed(seed)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_models = []
    fold_val_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f"\n===== Fold {fold+1}/{n_folds} =====")
        train_dataset = HSIDataset(data[train_idx], labels[train_idx], augment=True)
        val_dataset = HSIDataset(data[val_idx], labels[val_idx], augment=False)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

        model = HSIModel(num_bands=data.shape[1], spatial_embed_dim=256, spectral_embed_dim=128, patch_size=16)
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
        loss_fn = UnifiedFocalLoss(num_classes=num_classes, lambda_=0.5, alpha=0.25, gamma=2.0, delta=0.5, smooth=1e-8)
        scaler = torch.cuda.amp.GradScaler()
        early_stopping = EarlyStopping(patience=15, min_delta=0.001)

        # Warmup
        warmup_epochs = 5
        total_epochs = 60
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(total_epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(x)
                    logits = outputs['final_logits']
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * x.size(0)
            train_loss /= len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    logits = outputs['final_logits']
                    loss = loss_fn(logits, y)
                    val_loss += loss.item() * x.size(0)
            val_loss /= len(val_loader.dataset)

            # Scheduler step
            if epoch < warmup_epochs:
                lr_scale = min(1.0, float(epoch + 1) / warmup_epochs)
                for pg in optimizer.param_groups:
                    pg['lr'] = 2e-4 * lr_scale
            else:
                scheduler.step(epoch - warmup_epochs)

            print(f"Epoch {epoch+1}/{total_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Early stopping and checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
            if early_stopping(val_loss):
                print("Early stopping triggered.")
                break

        # Save best model for this fold
        fold_model_path = f"checkpoint_fold{fold+1}.pt"
        torch.save(best_model_state, fold_model_path)
        fold_models.append(fold_model_path)
        fold_val_losses.append(best_val_loss)
        print(f"Best val loss for fold {fold+1}: {best_val_loss:.4f}")

    # Ensemble: average predictions from all folds
    print("\n===== Ensemble Evaluation =====")
    ensemble_preds = None
    for fold_model_path in fold_models:
        model = HSIModel(num_bands=data.shape[1], spatial_embed_dim=256, spectral_embed_dim=128, patch_size=16)
        model.load_state_dict(torch.load(fold_model_path, map_location=device))
        model = model.to(device)
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in val_loader:
                x, _ = batch
                x = x.to(device)
                outputs = model(x)
                pred = torch.argmax(outputs['final_logits'], dim=1)
                preds.append(pred.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        if ensemble_preds is None:
            ensemble_preds = preds.astype(np.float32)
        else:
            ensemble_preds += preds.astype(np.float32)
    ensemble_preds /= len(fold_models)
    ensemble_preds = np.round(ensemble_preds).astype(np.int32)
    print("Ensemble predictions shape:", ensemble_preds.shape)

# ===================== MAIN =====================
if __name__ == "__main__":
    # Example usage: load your data and labels here
    # data: numpy array of shape [N, Bands, H, W]
    # labels: numpy array of shape [N, H, W]
    # Replace with actual data loading
    data = np.random.randn(100, 37, 128, 128)
    labels = np.random.randint(0, 5, (100, 128, 128))
    train_hsi_model(data, labels, num_classes=5, n_folds=5, seed=42)
