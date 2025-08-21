# /users/PAS2301/liu215229932/Music_Project/Models/Basen/EEG_Encoder_Softmax.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your existing EEGEncoder
# Path matches your note: /Models/Basen/BASEN_code/BASEN.py
from BASEN_code.BASEN import EEGEncoder


class GlobalAvgPool1d(nn.Module):
    """(B, C, T) -> (B, C) by averaging over time."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1)


class EEGEncoderClassifier(nn.Module):
    """
    EEG-only classifier:
      Input  : EEG (B, 20, 4864)
      Encode : BASEN EEGEncoder -> (B, Fe, T')
      Pool   : GlobalAvgPool1d over time -> (B, Fe)
      Head   : Linear(Fe -> n_classes) -> logits (no softmax)
      Loss   : Cross-entropy (provided via .loss())

    Notes:
      - No audio, no upsampling.
      - Uses your BASEN EEGEncoder exactly as implemented.
      - You can freeze the encoder if you want to probe representations.
    """
    def __init__(
        self,
        n_classes: int,
        layer: int = 8,
        feature_channel: int = 32,
        enc_channel: int = 64,
        proj_kernel_size: int = 1,
        kernel_size: int = 3,
        skip: bool = True,
        dilated: bool = False,
        freeze_encoder: bool = False,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        # Build the exact EEG encoder you showed
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

        self.pool = GlobalAvgPool1d()
        # Linear head -> logits (no softmax)
        self.head = nn.Linear(feature_channel, n_classes)

        self.label_smoothing = label_smoothing

    @torch.no_grad()
    def extract_features(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Returns pooled EEG features (B, feature_channel).
        eeg: (B, 20, 4864)
        """
        feat_map = self.eeg_encoder(eeg)   # (B, Fe, T')
        feat_vec = self.pool(feat_map)     # (B, Fe)
        return feat_vec

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        eeg: (B, 20, 4864)
        returns logits: (B, n_classes)  [no softmax]
        """
        feat_vec = self.extract_features(eeg)  # (B, Fe)
        logits = self.head(feat_vec)           # (B, n_classes)
        return logits

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy loss (expects class indices: LongTensor of shape (B,)).
        No softmax needed (F.cross_entropy applies log-softmax internally).
        """
        return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)


# ------------------------------ #
# Minimal dry-run (optional)
# ------------------------------ #
if __name__ == "__main__":
    # Dummy batch: 8 trials, EEG shape (20, 4864)
    eeg = torch.randn(8, 20, 4864)

    # Example model: 4 classes, same defaults as your EEGEncoder snippet (no dilation)
    model = EEGEncoderClassifier(
        n_classes=4,
        layer=8,
        feature_channel=32,  # matches EEGEncoder(feature_channel)
        enc_channel=64,
        proj_kernel_size=1,
        kernel_size=3,
        skip=True,
        dilated=False,
        freeze_encoder=False,
        label_smoothing=0.0,
    )

    logits = model(eeg)  # (8, 4)
    print("Logits shape:", logits.shape)

    # Dummy labels
    y = torch.randint(0, 4, (8,))
    ce = model.loss(logits, y)
    print("CE loss:", ce.item())
