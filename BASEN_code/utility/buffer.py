import torch


class SlidingAttractorBuffer:
    """
    Stores and updates a^s (past extracted speech) for online BASEN.
    Keeps a fixed-length buffer and returns encoded attractor features.
    """

    def __init__(self, audio_encoder, instrument_encoder, max_len):
        """
        Args:
            instrument_encoder (nn.Module): Module that turns audio into attractor features.
            max_len (int): Max time length to retain (must be set explicitly).
        """
        self.audio_encoder = audio_encoder
        self.instrument_encoder = instrument_encoder
        self.max_len = max_len
        self.buffer = []

    def reset(self):
        self.buffer = []

    def append(self, new_audio_chunk):
        # Ensure it has 2 channels to match current noisy input
        if new_audio_chunk.shape[1] == 1:
            new_audio_chunk = new_audio_chunk.repeat(1, 2, 1)  # duplicate channel

        self.buffer.append(new_audio_chunk.detach())
        audio = torch.cat(self.buffer, dim=-1)

        if audio.shape[-1] > self.max_len:
            audio = audio[:, :, -self.max_len:]
            self.buffer = [audio]

        with torch.no_grad():
            encoded = self.audio_encoder(audio)  # shape: [B, C1, T]
            a_s = self.instrument_encoder(encoded)  # shape: [B, C2, T]
            # Expand a_s to match time dimension of encoded
            time_steps = encoded.shape[-1]  # T = 8813
            a_s = a_s.unsqueeze(-1).repeat(1, 1, time_steps)  # shape: (B, C2, T)

        return a_s
