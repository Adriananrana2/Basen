
import torch


class SlidingAttractorBuffer:
    """
    Stores and updates a^s (past extracted speech) for online BASEN.
    Keeps a fixed-length buffer and returns encoded attractor features.
    """

    def __init__(self, speaker_encoder, max_len):
        """
        Args:
            speaker_encoder (nn.Module): Module that turns audio into attractor features.
            max_len (int): Max time length to retain (must be set explicitly).
        """
        self.encoder = speaker_encoder
        self.max_len = max_len  # <== Fixed: added max_len attribute
        self.num_chunks = max_len
        self.buffer = []

    def reset(self):
        self.buffer = []

    def append(self, new_audio_chunk):  # shape: [B, 1, T]
        self.buffer.append(new_audio_chunk.detach())
        audio = torch.cat(self.buffer, dim=-1)

        if audio.shape[-1] > self.max_len:
            audio = audio[:, :, -self.max_len:]
            self.buffer = [audio]

        a_s = self.encoder(audio)  # output: [B, D, T_past]
        return a_s


class SlidingEEGBuffer:
    """
    Stores and updates past encoded EEG features for CMCA input.
    Maintains fixed-length sliding window.
    """

    def __init__(self, max_len):
        """
        Args:
            max_len (int): Max time length to retain (must be set explicitly).
        """
        self.max_len = max_len  # <== Fixed: added max_len attribute
        self.num_chunks = max_len
        self.buffer = []

    def reset(self):
        self.buffer = []

    def append(self, new_eeg_chunk):  # shape: [B, N, T]
        self.buffer.append(new_eeg_chunk.detach())
        eeg = torch.cat(self.buffer, dim=-1)

        if eeg.shape[-1] > self.max_len:
            eeg = eeg[:, :, -self.max_len:]
            self.buffer = [eeg]

        return eeg  # output: [B, N, T_past]
