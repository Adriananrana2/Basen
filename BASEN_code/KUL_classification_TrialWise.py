# /users/PAS2301/liu215229932/Music_Project/Models/Basen/BASEN_code/KUL_classification_TrialWise.py
# -------------------------------------------------------------------------------------------------
# KU Leuven AAD (EEG-only) — Trial-wise cross-validation classifier (KEEP ALL 64 CHANNELS)
#
# Change vs previous version:
#   • Uses your new backbone **EEGEncoder_64ch** (first conv is 64→feature_channel). No 1×1 pre-proj.
#   • Everything else is the same (windowing, trial-wise split, attention pooling, TensorBoard).
# -------------------------------------------------------------------------------------------------

import os
import glob
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.io import loadmat

# Import your 64-ch EEG encoder backbone from BASEN.py (same folder)
from BASEN import EEGEncoder_64ch

# =============================
# 1) CONFIGURATION
# =============================
KUL_ROOT = \
    "/users/PAS2301/liu215229932/Music_Project/Datasets/KUL_AAD"   # <-- set where S*.mat live

CLASSES = ["L", "R"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}  # {'L':0,'R':1}

FS = 128                 # dataset sampling rate (Hz)
WINDOW_SEC = 2.0         # seconds per window
HOP_SEC = 1.0            # seconds between window starts (50% overlap)
WINDOW_SAMPLES = int(WINDOW_SEC * FS)
HOP_SAMPLES = int(HOP_SEC * FS)

BATCH_SIZE = 16
NUM_WORKERS = 4
MAX_EPOCHS = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_EVERY_ITERS = 200    # validate every N training iterations
LABEL_SMOOTH = 0.0

FREEZE_ENCODER = False   # True = train only attention+head; False = train encoder too
DILATED = False          # set True to use dilated convs in EEGEncoder_64ch
ATTN_HIDDEN = 64         # attention hidden size (A)

VAL_TRIAL_FRACTION = 0.2 # ~20% of trials for validation (chosen at trial level)
RANDOM_SEED = 42

RUN_DIR = "/users/PAS2301/liu215229932/Music_Project/Models/Basen/BASEN_code/KUL_TrialWise_Run"
TBOARD_DIR = os.path.join(RUN_DIR, "tensorboard")
BEST_CKPT_DIR = os.path.join(RUN_DIR, "best_ckpt")
os.makedirs(TBOARD_DIR, exist_ok=True)
os.makedirs(BEST_CKPT_DIR, exist_ok=True)

# =============================
# 2) UTILITIES
# =============================

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (preds == targets).float().mean().item()

@torch.no_grad()
def compute_balanced_accuracy(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    ba_sum, n = 0.0, 0
    for c in range(num_classes):
        mask = (targets == c)
        denom = int(mask.sum().item())
        if denom == 0:
            continue
        tp = int(((preds == c) & mask).sum().item())
        ba_sum += tp / denom
        n += 1
    return ba_sum / max(n, 1)

# =============================
# 3) DATASET (trial windows)
# =============================
class KULTrialsDataset(Dataset):
    """Read .mat files, cut into windows, return (eeg64, label, subject, trial_id).
       - eeg64 shape: (64, WINDOW_SAMPLES)
       - label: 0 ('L') or 1 ('R')
       - subject: e.g., 'S1'
       - trial_id: integer if available, else -1
    """
    def __init__(self, root: str, fs: int, window_samples: int, hop_samples: int):
        super().__init__()
        self.items: List[Tuple[np.ndarray, int, str, int]] = []
        mats = sorted(glob.glob(os.path.join(root, "**", "*.mat"), recursive=True))
        if not mats:
            raise RuntimeError(f"No .mat files under {root}")
        for m in mats:
            try:
                data = loadmat(m, squeeze_me=True, struct_as_record=False)
            except Exception as e:
                print(f"Skip {m}: load error {e}")
                continue
            # Heuristic: either a list/array of trials or single-trial struct
            trials = []
            if 'trials' in data:
                arr = data['trials']
                trials = list(arr.flatten()) if isinstance(arr, np.ndarray) else list(arr)
            else:
                trials = [data]
            for tr in trials:
                # Extract EEG (samples x 64) and meta
                try:
                    raw = tr['RawData'] if isinstance(tr, dict) else tr.RawData
                    eeg = raw['EegData'] if isinstance(raw, dict) else raw.EegData
                except Exception:
                    print(f"{m}: missing RawData/EegData; skip")
                    continue
                try:
                    ear = tr['attended_ear'] if isinstance(tr, dict) else tr.attended_ear
                except Exception:
                    print(f"{m}: missing attended_ear; skip")
                    continue
                ear = ear.decode('utf-8') if isinstance(ear, bytes) else str(ear)
                if ear not in CLASS_TO_IDX:
                    print(f"{m}: unexpected ear '{ear}'; skip")
                    continue
                label = CLASS_TO_IDX[ear]
                # Subject + trial ID
                try:
                    subj = tr['subject'] if isinstance(tr, dict) else tr.subject
                except Exception:
                    subj = os.path.basename(m).split('.')[0]
                subj = subj.decode('utf-8') if isinstance(subj, bytes) else str(subj)
                try:
                    tid = int(tr['TrialID'] if isinstance(tr, dict) else tr.TrialID)
                except Exception:
                    tid = -1
                # To (64, T)
                eeg = np.asarray(eeg, dtype=np.float32)
                if eeg.ndim != 2:
                    continue
                if eeg.shape[1] == 64:
                    eeg = eeg.T
                elif eeg.shape[0] != 64:
                    continue
                T = eeg.shape[1]
                # Windowing (drop last partial)
                for s in range(0, max(0, T - window_samples + 1), hop_samples):
                    seg = eeg[:, s:s + window_samples]
                    if seg.shape[1] == window_samples:
                        self.items.append((seg, label, subj, tid))
        if not self.items:
            raise RuntimeError("No windows created; check KUL_ROOT and file structure")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seg, label, subj, tid = self.items[idx]
        return torch.from_numpy(seg.copy()), int(label), subj, int(tid)

# =============================
# 4) MODEL (EEGEncoder_64ch + Attention Pool + Head)
# =============================
class TemporalAttentionPool1d(nn.Module):
    def __init__(self, in_channels: int, attn_hidden: int = 64):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, attn_hidden, kernel_size=1, bias=True)
        self.context = nn.Parameter(torch.randn(attn_hidden))
    def forward(self, x: torch.Tensor):
        h = torch.tanh(self.proj(x))               # (B, A, T)
        scores = torch.einsum('bat,a->bt', h, self.context)
        attn = torch.softmax(scores, dim=-1)       # (B, T)
        pooled = torch.sum(x * attn.unsqueeze(1), dim=-1)  # (B, C)
        return pooled, attn

class KULEEG_AttnClassifier(nn.Module):
    def __init__(self, n_classes=2, freeze_encoder=False, dilated=False, attn_hidden=64, label_smoothing=0.0):
        super().__init__()
        self.encoder = EEGEncoder_64ch(layer=8, enc_channel=64, feature_channel=32,
                                       proj_kernel_size=1, kernel_size=3, skip=True, dilated=dilated)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.attn = TemporalAttentionPool1d(in_channels=32, attn_hidden=attn_hidden)
        self.head = nn.Linear(32, n_classes)
        self.label_smoothing = label_smoothing
    def forward(self, eeg64: torch.Tensor):  # eeg64: (B,64,T)
        fmap = self.encoder(eeg64)                 # (B,32,T')
        vec, _ = self.attn(fmap)                   # (B,32)
        logits = self.head(vec)                    # (B,2)
        return logits
    def loss(self, logits, targets):
        return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)

# =============================
# 5) SPLIT: TRIAL-WISE (no leakage)
# =============================

def split_by_trial(keys: List[Tuple[str, int]], val_fraction: float, seed: int):
    """keys: a list of (subject, trial_id) for each window index, possibly repeating per window.
       We choose a subset of UNIQUE keys for validation, then map windows accordingly.
    """
    uniq = sorted(set(keys))
    rnd = random.Random(seed)
    rnd.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_fraction))
    valset = set(uniq[:n_val])
    train_idx, val_idx = [], []
    for i, k in enumerate(keys):
        (val_idx if k in valset else train_idx).append(i)
    return train_idx, val_idx

class SubsetByIndex(Dataset):
    def __init__(self, base: Dataset, indices: List[int]):
        self.base, self.idx = base, indices
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        eeg, label, _, _ = self.base[self.idx[i]]
        return eeg, label

# =============================
# 6) TRAINING / VALIDATION
# =============================
@torch.no_grad()
def run_validation(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds_all, targs_all = [], []
    for eeg, lab in loader:
        eeg = eeg.to(device)
        lab = lab.to(device, dtype=torch.long)
        logits = model(eeg)
        preds = torch.argmax(logits, dim=1)
        preds_all.append(preds)
        targs_all.append(lab)
    preds_all = torch.cat(preds_all)
    targs_all = torch.cat(targs_all)
    return {
        'acc': compute_accuracy(preds_all, targs_all),
        'bal_acc': compute_balanced_accuracy(preds_all, targs_all, num_classes=2)
    }


def main():
    set_all_seeds(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Build full dataset of windows
    fullds = KULTrialsDataset(KUL_ROOT, FS, WINDOW_SAMPLES, HOP_SAMPLES)

    # Build keys per window for trial-wise split
    keys = []
    for i in range(len(fullds)):
        # fullds[i] returns (eeg, label, subj, tid)
        _, _, subj, tid = fullds[i]
        keys.append((subj, tid))

    train_idx, val_idx = split_by_trial(keys, VAL_TRIAL_FRACTION, RANDOM_SEED)
    print(f"Windows total: {len(fullds)} | Train: {len(train_idx)} | Val: {len(val_idx)}")

    train_ds = SubsetByIndex(fullds, train_idx)
    val_ds = SubsetByIndex(fullds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=max(1, NUM_WORKERS//2), pin_memory=False, drop_last=False)

    # Model
    model = KULEEG_AttnClassifier(n_classes=2, freeze_encoder=FREEZE_ENCODER,
                                  dilated=DILATED, attn_hidden=ATTN_HIDDEN,
                                  label_smoothing=LABEL_SMOOTH).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    writer = SummaryWriter(log_dir=TBOARD_DIR)

    best_bal = -1.0
    global_step = 0

    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        for eeg, lab in train_loader:
            global_step += 1
            eeg = eeg.to(device)
            lab = lab.to(device, dtype=torch.long)

            logits = model(eeg)
            loss = model.loss(logits, lab)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            writer.add_scalar('train/loss', float(loss.item()), global_step)

            if global_step % VAL_EVERY_ITERS == 0:
                metrics = run_validation(model, val_loader, device)
                writer.add_scalar('val/accuracy', metrics['acc'], global_step)
                writer.add_scalar('val/balanced_accuracy', metrics['bal_acc'], global_step)
                print(f"[Epoch {epoch} | Iter {global_step}] Val Acc {metrics['acc']:.4f} | BalAcc {metrics['bal_acc']:.4f}")
                if metrics['bal_acc'] > best_bal:
                    best_bal = metrics['bal_acc']
                    ckpt = os.path.join(BEST_CKPT_DIR, 'best.pt')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'global_step': global_step,
                        'epoch': epoch,
                        'best_bal_acc': best_bal,
                        'config': {
                            'FS': FS, 'WINDOW_SEC': WINDOW_SEC, 'HOP_SEC': HOP_SEC,
                            'FREEZE_ENCODER': FREEZE_ENCODER, 'DILATED': DILATED,
                        }
                    }, ckpt)
                    print(f"Saved best checkpoint to {ckpt} (BalAcc={best_bal:.4f})")

    # Final validation
    metrics = run_validation(model, val_loader, device)
    print(f"Final Val Acc {metrics['acc']:.4f} | Final BalAcc {metrics['bal_acc']:.4f}")
    writer.close()


if __name__ == '__main__':
    main()
