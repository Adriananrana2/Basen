import os
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader


class EEGAudioDataset(Dataset):
    def __init__(self, stimulus_wav_dir, isolated_wav_dir, response_npy_dir):
        super().__init__()
        self.stimulus_wav_dir = stimulus_wav_dir
        self.isolated_wav_dir = isolated_wav_dir
        self.response_npy_dir = response_npy_dir

        # Use stimulus wavs to determine the base list
        stimulus_files = sorted([
            f for f in os.listdir(stimulus_wav_dir)
            if f.endswith('_stimulus.wav')])

        self.filenames = []
        for f in stimulus_files:
            base = f.replace('_stimulus.wav', '')
            iso_path = os.path.join(isolated_wav_dir, base + '_soli.wav')
            eeg_path = os.path.join(response_npy_dir, base + '_response.npy')
            if os.path.exists(iso_path) and os.path.exists(eeg_path):
                self.filenames.append(f)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        try:
            # Load mixed audio (2-channel)
            mixed_audio, _ = sf.read(os.path.join(self.stimulus_wav_dir, fname))
            mixed_audio = torch.tensor(mixed_audio.T, dtype=torch.float32)

            # Load isolated audio (1-channel)
            base = fname.replace("_stimulus.wav", "")
            isolated_audio, _ = sf.read(os.path.join(self.isolated_wav_dir, base + "_soli.wav"))
            isolated_audio = torch.tensor(isolated_audio[None, :], dtype=torch.float32)

            # Load EEG (20 channels)
            eeg_data = np.load(os.path.join(self.response_npy_dir, base + "_response.npy"))
            eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)

            return mixed_audio, eeg_tensor, isolated_audio, os.path.join(self.stimulus_wav_dir, fname)


        except Exception as e:
            print(f"❌ Failed loading file: {fname}")
            raise e


def load_CleanNoisyPairDataset(stimulus_wav_dir, isolated_wav_dir, response_npy_dir, subset, batch_size, num_gpus=1):
    dataset = EEGAudioDataset(stimulus_wav_dir, isolated_wav_dir, response_npy_dir)
    kwargs = {"batch_size": batch_size, "num_workers": 4, "pin_memory": False, "drop_last": False}

    if num_gpus > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset, sampler=sampler, **kwargs)
    else:
        shuffle = (subset == "train")
        return DataLoader(dataset, shuffle=shuffle, **kwargs)
