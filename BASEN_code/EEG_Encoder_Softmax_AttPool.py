# /users/PAS2301/liu215229932/Music_Project/Models/Basen/BASEN_code/EEG_Encoder_Softmax_AttPool.py
# -------------------------------------------------------------------------------------------------
# Single-file TRAINING SCRIPT (EEG-only, Attention Pooling)
#
# This script lives in BASEN_code/ and contains EVERYTHING you need:
#   • Dataloader that reads EEG .npy trials from four class folders: Gt, Vx, Dr, Bs
#   • Model that wraps your existing EEGEncoder (from BASEN.py), adds Temporal Attention Pooling,
#     then a Linear head that outputs logits (no softmax). Use CrossEntropyLoss on logits.
#   • Training loop with TensorBoard logging: train/loss, val/accuracy, val/balanced_accuracy
#   • 80/20 train/val split with fixed seed
#   • Best-checkpoint saving by balanced accuracy
#
# Assumptions:
#   - Each .npy file is a single trial shaped (20, 4864). If the shape is (4864, 20), we transpose.
#   - Label is implied by parent folder name. Only the 4 folders below are used.
#   - Trials are 19 s of EEG with one static attention label per trial (offline classification).
#
# To run (on OSC):
#   module load ... (your env)
#   python EEG_Encoder_Softmax_AttPool.py
# -------------------------------------------------------------------------------------------------

import os
import glob
import random
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------------------------
# Bring in your encoder backbone from BASEN.py (same folder)
# -----------------------------------------------------------------------------------------------
from BASEN import EEGEncoder  # class defined by you in BASEN.py


# ===============================================================================================
# 1) CONFIGURATION (edit here)
# ===============================================================================================
DATA_ROOT = \
    "/users/PAS2301/liu215229932/Music_Project/Dataset/EEG_Responses_no_upsample/responses"

# Only use these four class folders. Their names must match folder names exactly.
CLASSES = ["Gt", "Vx", "Dr", "Bs"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}  # e.g., {"Gt":0, "Vx":1, "Dr":2, "Bs":3}

# Train/Val ratio and randomness
TRAIN_RATIO = 0.80
RANDOM_SEED = 42

# Data loading
BATCH_SIZE = 8
NUM_WORKERS = 4

# Training hyperparams
MAX_EPOCHS = 200
VAL_EVERY_ITERS = 1          # Validate every N training iterations (set 1 for per-iter)
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
LABEL_SMOOTH = 0.0             # Try 0.05 if training is unstable

# Encoder / model knobs
FREEZE_ENCODER = False         # True = only train attention + head; False = train encoder too
DILATED = False                # True = use dilated convs in EEGEncoder
ATTN_HIDDEN = 64               # Hidden size A for attention projection

# Logging / checkpoints
RUN_DIR = "/users/PAS2301/liu215229932/Music_Project/Models/Basen/BASEN_code/EEG_AttPool_Run"
TBOARD_DIR = os.path.join(RUN_DIR, "tensorboard")
BEST_CKPT_DIR = os.path.join(RUN_DIR, "best_ckpt")
os.makedirs(TBOARD_DIR, exist_ok=True)
os.makedirs(BEST_CKPT_DIR, exist_ok=True)


# ===============================================================================================
# 2) UTILS: seeding and simple metrics
# ===============================================================================================

def set_all_seeds(seed: int = 42):
    """Make randomness repeatable across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Simple top-1 accuracy: fraction of correct predictions."""
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / max(total, 1)


@torch.no_grad()
def compute_balanced_accuracy(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Macro-averaged recall across classes (each class counts equally).
    BA = mean_c ( TP_c / (TP_c + FN_c) )
    """
    ba_sum = 0.0
    classes_present = 0
    for c in range(num_classes):
        mask = (targets == c)
        denom = mask.sum().item()  # TP + FN for class c
        if denom == 0:
            # No samples of class c in this eval window; skip it
            continue
        tp = ((preds == c) & mask).sum().item()
        recall_c = tp / denom
        ba_sum += recall_c
        classes_present += 1
    if classes_present == 0:
        return 0.0
    return ba_sum / classes_present


# ===============================================================================================
# 3) DATASET: read .npy trials from folders and return (EEG tensor, label)
# ===============================================================================================
class EEGFolderDataset(Dataset):
    """
    Directory layout (only these 4 used):
        DATA_ROOT/
          Gt/   *.npy  -> label 0
          Vx/   *.npy  -> label 1
          Dr/   *.npy  -> label 2
          Bs/   *.npy  -> label 3

    Each .npy loads to (20, 4864) float array. If you find (4864, 20), we transpose to (20, 4864).
    """

    def __init__(self, root: str, classes: List[str], class_to_idx: dict):
        super().__init__()
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples: List[Tuple[str, int]] = []  # list of (filepath, label)

        for cls in classes:
            cls_dir = os.path.join(root, cls)
            pattern = os.path.join(cls_dir, "*.npy")
            for f in sorted(glob.glob(pattern)):
                self.samples.append((f, class_to_idx[cls]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No .npy files found under {root} for classes {classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        fpath, label = self.samples[index]
        arr = np.load(fpath)  # expected (20, 4864)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array at {fpath}, got shape {arr.shape}")
        # Fix orientation if needed
        if arr.shape[0] != 20:
            if arr.shape[1] == 20:
                arr = arr.T
            else:
                raise ValueError(f"Unexpected EEG shape {arr.shape} at {fpath}")
        eeg = torch.from_numpy(arr.astype(np.float32))  # (20, 4864)
        return eeg, label


# ===============================================================================================
# 4) MODEL: EEGEncoder + Temporal Attention Pooling + Linear Head (logits)
# ===============================================================================================
class TemporalAttentionPool1d(nn.Module):
    """
    Learn a weight for each time step after the encoder, then weighted-sum features over time.

    Input:
      x: (B, C, T) where C = feature_channel from EEGEncoder, T = time steps after encoder
    Returns:
      pooled: (B, C)
      attn:   (B, T) attention weights (each row sums to 1)
    """
    def __init__(self, in_channels: int, attn_hidden: int = 64):
        super().__init__()
        # 1x1 conv acts like Linear(C -> A) applied at each time step
        self.proj = nn.Conv1d(in_channels, attn_hidden, kernel_size=1, bias=True)
        # Learnable context vector in attention space A
        self.context = nn.Parameter(torch.randn(attn_hidden))

    def forward(self, x: torch.Tensor):
        # x: (B, C, T)
        h = torch.tanh(self.proj(x))                   # (B, A, T)
        # scores[b, t] = dot(h[b, :, t], context)
        scores = torch.einsum('bat,a->bt', h, self.context)
        attn = torch.softmax(scores, dim=-1)           # (B, T)
        pooled = torch.sum(x * attn.unsqueeze(1), dim=-1)  # (B, C)
        return pooled, attn


class EEGEncoderAttnClassifier(nn.Module):
    """
    Wrap your EEGEncoder, add attention pooling over time, then a Linear head.
    Forward returns **logits** (no softmax). Use F.cross_entropy(logits, targets).
    """
    def __init__(
        self,
        n_classes: int,
        layer: int = None,
        feature_channel: int = 32,
        enc_channel: int = 64,
        proj_kernel_size: int = 1,
        kernel_size: int = 3,
        skip: bool = True,
        dilated: bool = None,
        freeze_encoder: bool = False,
        attn_hidden: int = 64,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        # Build the encoder exactly as your BASEN.py defines it
        self.eeg_encoder = EEGEncoder(
            layer=layer,
            enc_channel=enc_channel,
            feature_channel=feature_channel,
            proj_kernel_size=proj_kernel_size,
            kernel_size=kernel_size,
            skip=skip,
            dilated=dilated,
        )
        if freeze_encoder:
            for p in self.eeg_encoder.parameters():
                p.requires_grad = False

        self.attn_pool = TemporalAttentionPool1d(in_channels=feature_channel, attn_hidden=attn_hidden)
        self.head = nn.Linear(feature_channel, n_classes)  # logits (no softmax)
        self.label_smoothing = label_smoothing

    @torch.no_grad()
    def extract_features(self, eeg: torch.Tensor):
        """Return (feature_map, pooled_vector, attention_weights) for inspection/plots."""
        feat_map = self.eeg_encoder(eeg)         # (B, Fe, T')
        pooled, attn = self.attn_pool(feat_map)  # (B, Fe), (B, T')
        return feat_map, pooled, attn

    def forward(self, eeg: torch.Tensor, return_attn: bool = False):
        feat_map = self.eeg_encoder(eeg)           # (B, Fe, T')
        pooled, attn = self.attn_pool(feat_map)    # (B, Fe), (B, T')
        logits = self.head(pooled)                 # (B, n_classes)
        if return_attn:
            return logits, attn
        return logits

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)


# ===============================================================================================
# 5) VALIDATION
# ===============================================================================================
@torch.no_grad()
def run_validation(model: nn.Module, val_loader: DataLoader, device: torch.device, n_classes: int) -> dict:
    model.eval()
    preds_all, targs_all = [], []
    for eeg, label in val_loader:
        eeg = eeg.to(device)
        label = torch.as_tensor(label, device=device, dtype=torch.long)
        logits = model(eeg)
        preds = torch.argmax(logits, dim=1)
        preds_all.append(preds)
        targs_all.append(label)
    preds_all = torch.cat(preds_all, dim=0)
    targs_all = torch.cat(targs_all, dim=0)
    acc = compute_accuracy(preds_all, targs_all)
    bal = compute_balanced_accuracy(preds_all, targs_all, n_classes)
    return {"acc": acc, "bal_acc": bal}


# ===============================================================================================
# 6) MAIN TRAINING LOOP
# ===============================================================================================

def main():
    set_all_seeds(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Build the dataset from the four class folders
    full_ds = EEGFolderDataset(DATA_ROOT, CLASSES, CLASS_TO_IDX)

    # Train/Val split (80/20) using fixed seed
    n_total = len(full_ds)
    n_train = int(TRAIN_RATIO * n_total)
    n_val = n_total - n_train
    g = torch.Generator().manual_seed(RANDOM_SEED)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)
    print(f"Total: {n_total}  Train: {len(train_ds)}  Val: {len(val_ds)}")

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=max(1, NUM_WORKERS // 2), pin_memory=False, drop_last=False)

    # Build model
    model = EEGEncoderAttnClassifier(
        n_classes=len(CLASSES),
        layer=2,
        feature_channel=32,
        enc_channel=64,
        proj_kernel_size=1,
        kernel_size=3,
        skip=True,
        dilated=DILATED,
        freeze_encoder=FREEZE_ENCODER,
        attn_hidden=ATTN_HIDDEN,
        label_smoothing=LABEL_SMOOTH,
    ).to(device)

    # Optimizer (only parameters with requires_grad=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = None  # keep simple

    # TensorBoard
    writer = SummaryWriter(log_dir=TBOARD_DIR)

    # Training state
    global_step = 0
    best_bal_acc = -1.0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0

        for eeg, label in train_loader:
            global_step += 1

            # Move batch to device
            eeg = eeg.to(device)  # (B, 20, 4864)
            label = torch.as_tensor(label, device=device, dtype=torch.long)

            # Forward
            logits = model(eeg)            # (B, 4)
            loss = model.loss(logits, label)

            # Backprop
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Track epoch loss
            epoch_loss_sum += float(loss.item())
            epoch_batches += 1

            # TensorBoard: train loss each step
            writer.add_scalar("train/loss", float(loss.item()), global_step)

            # Validate every N iterations
            if global_step % VAL_EVERY_ITERS == 0:
                metrics = run_validation(model, val_loader, device, n_classes=len(CLASSES))
                val_acc = metrics["acc"]
                val_bal = metrics["bal_acc"]
                writer.add_scalar("val/accuracy", val_acc, global_step)
                writer.add_scalar("val/balanced_accuracy", val_bal, global_step)
                print(f"[Epoch {epoch} | Iter {global_step}] Val Acc: {val_acc:.4f}  Val BalAcc: {val_bal:.4f}")

                # Best checkpoint by balanced accuracy
                if val_bal > best_bal_acc:
                    best_bal_acc = val_bal
                    ckpt_path = os.path.join(BEST_CKPT_DIR, "best.pt")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "best_bal_acc": best_bal_acc,
                        "classes": CLASSES,
                        "config": {
                            "BATCH_SIZE": BATCH_SIZE,
                            "LR": LEARNING_RATE,
                            "WEIGHT_DECAY": WEIGHT_DECAY,
                            "FREEZE_ENCODER": FREEZE_ENCODER,
                            "DILATED": DILATED,
                            "ATTN_HIDDEN": ATTN_HIDDEN,
                        }
                    }, ckpt_path)
                    print(f"Saved best checkpoint to: {ckpt_path} (BalAcc={best_bal_acc:.4f})")

        # End-of-epoch logging
        avg_loss = epoch_loss_sum / max(epoch_batches, 1)
        print(f"End of Epoch {epoch} | Avg Train Loss: {avg_loss:.6f}")

    # Final validation
    final = run_validation(model, val_loader, device, n_classes=len(CLASSES))
    print(f"Final Val Acc: {final['acc']:.4f}  Final Val BalAcc: {final['bal_acc']:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
