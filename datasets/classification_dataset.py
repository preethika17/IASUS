import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import config


def spec_augment(mel, freq_mask=15, time_mask=20):
    """Simple SpecAugment: mask a random frequency band and a random time band."""
    mel = mel.copy()
    n_mels, n_frames = mel.shape

    if n_mels > freq_mask:
        f0 = np.random.randint(0, n_mels - freq_mask)
        mel[f0:f0 + freq_mask, :] = mel.mean()

    if n_frames > time_mask:
        t0 = np.random.randint(0, n_frames - time_mask)
        mel[:, t0:t0 + time_mask] = mel.mean()

    return mel


class ClassificationDataset(Dataset):
    """
    fold_list: which ESC-50 folds to include, e.g. config.TRAIN_FOLDS or config.VAL_FOLDS
    train: if True, applies SpecAugment. Set False for validation/test.
    Requires preprocess_cache.py to have been run first.
    """
    def __init__(self, fold_list, train=True):
        df = pd.read_csv(config.CSV_FILE)
        self.df = df[df["fold"].isin(fold_list)].reset_index(drop=True)
        self.train = train

        with open(config.STATS_FILE) as f:
            stats = json.load(f)
        self.mean = stats["mean"]
        self.std = stats["std"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        cache_path = os.path.join(config.CACHE_DIR, filename + ".npy")
        mel = np.load(cache_path)

        # Normalize using train-set stats (applied consistently to train + val)
        mel = (mel - self.mean) / (self.std + 1e-6)

        if self.train and config.SPEC_AUGMENT_TRAIN_ONLY:
            mel = spec_augment(mel, config.FREQ_MASK, config.TIME_MASK)

        mel = torch.tensor(mel).float().unsqueeze(0)
        label = int(row["target"])
        return mel, label