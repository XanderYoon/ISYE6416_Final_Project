import numpy as np
import torch
from torch.utils.data import Dataset
from audio.io import load_wav_scipy as load_audio


class HeartRegressionDataset(Dataset):
    """
    Returns:
        mixed_segment: (T,)
        heart_segment: (T,)
    """

    def __init__(self, df, segment_len=65536, random_crop=True):
        self.df = df
        self.segment_len = segment_len
        self.random_crop = random_crop

    def __len__(self):
        return len(self.df)

    def _get_segment(self, x):
        L = len(x)
        if L == 0:
            return np.zeros(self.segment_len, dtype=np.float32)

        if L < self.segment_len:
            pad = self.segment_len - L
            return np.pad(x, (0, pad), mode="constant").astype(np.float32)

        start = (
            np.random.randint(0, L - self.segment_len + 1)
            if self.random_crop
            else 0
        )
        return x[start:start + self.segment_len].astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        sr, mix = load_audio(row["mixture_file"], normalize=True)
        _, heart = load_audio(row["heart_ref_file"], normalize=True)

        L = min(len(mix), len(heart))
        mix = mix[:L]
        heart = heart[:L]

        mix_seg = self._get_segment(mix)
        heart_seg = self._get_segment(heart)

        return (
            torch.tensor(mix_seg, dtype=torch.float32),
            torch.tensor(heart_seg, dtype=torch.float32),
        )
