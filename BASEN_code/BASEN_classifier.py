import torch
import torch.nn as nn
from BASEN import BASEN_OFFLINE  # Adjust import path if needed

class BASENClassifier(nn.Module):
    def __init__(self, basen_model):
        super().__init__()
        self.basen = basen_model
        self.classifier = nn.Linear(96, 4)  # EEG(32) + Audio(64) → 4 classes

    def forward(self, mixed_audio, eeg):
        # with torch.no_grad():
        audio_feat, eeg_feat = self.basen.forward_features(mixed_audio, eeg)  # (B, C, T)

        # Concatenate along channel dimension → (B, 96, T)
        feat = torch.cat([eeg_feat, audio_feat], dim=1)

        # Collapse time dimension via mean → (B, 96)
        feat = feat.mean(dim=2)

        return self.classifier(feat)  # (B, 4)
