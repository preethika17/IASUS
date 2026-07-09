import os
import json
import random
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
import config

WAVE_CACHE_DIR = "data/wave_cache"
NUM_CLASSES = 50
MAX_SAMPLES = config.NUM_SAMPLES * 2  # covers the longest case (full-length concat)
TARGET_RMS = 0.05  # target loudness level for normalization before mixing


def normalize_rms(wave, target_rms=TARGET_RMS):
    """
    Normalize a waveform to a fixed RMS (loudness) level. ESC-50 clips vary
    naturally in recording volume; without this, a louder clip can dominate
    an overlay mix and effectively bury the quieter source, so the model
    never learns to detect it -- directly hurting recall on mixed samples.
    """
    rms = np.sqrt(np.mean(wave ** 2)) + 1e-8
    return wave * (target_rms / rms)


# Discrete mixing ratios instead of a plain continuous uniform range: mostly
# balanced mixes (50:50, 60:40) with 70:30 fairly common and 80:20 rare. 80:20
# is the floor -- going quieter than that risks the secondary sound being
# genuinely inaudible, which isn't a learnable signal for the model (or a
# human listener). Which clip ends up on the loud/quiet side is randomized
# separately below, so this only controls the mix's balance, not which
# source dominates.
MIX_RATIO_CHOICES = [0.5, 0.6, 0.7, 0.8]
MIX_RATIO_WEIGHTS = [0.4, 0.3, 0.2, 0.1]


def sample_mix_ratio():
    ratio = random.choices(MIX_RATIO_CHOICES, weights=MIX_RATIO_WEIGHTS)[0]
    if random.random() < 0.5:
        ratio = 1 - ratio  # let either clip be the dominant one, symmetrically
    return ratio


class MixedAudioDataset(Dataset):
    """
    Multi-label dataset built on top of ESC-50.

    Each sample is, at random, one of four modes:
      - single             : one clip                              (label = one-hot)
      - same_class_overlay : two DIFFERENT clips of the SAME class,
                              overlaid                              (label = one-hot,
                                                                       still just that class)
      - overlay            : two clips of DIFFERENT classes,
                              overlaid                              (label = multi-hot)
      - concat              : two clips concatenated back-to-back    (label = multi-hot)

    same_class_overlay teaches the model that two overlapping instances of the
    same sound (e.g. two different dog barks layered together) should still
    produce a single confident label, not get treated as evidence of a second
    class -- without it, the model only ever sees "overlay implies 2 different
    labels" during training, which doesn't match real audio where the same
    sound can repeat/layer.

    Requires preprocess_wave_cache.py and preprocess_cache.py to have run
    first (we reuse the mel normalization stats from the single-clip cache).

    All waveforms are padded with trailing silence to MAX_SAMPLES (the
    length of a full concat, i.e. 10s) so every mel spectrogram in a batch
    has the same width and can be stacked by the default DataLoader collate.

    `length` controls how many samples the dataset reports per epoch --
    since mixes are generated randomly on the fly, this is independent of
    the number of underlying audio files.
    """

    SINGLE_PROB = 0.40
    SAME_CLASS_OVERLAY_PROB = 0.20
    OVERLAY_PROB = 0.25
    CONCAT_PROB = 0.15

    def __init__(self, fold_list, train=True, length=4000, seed=42, mode_probs=None):
        df = pd.read_csv(config.CSV_FILE)
        self.df = df[df["fold"].isin(fold_list)].reset_index(drop=True)
        self.train = train
        self.length = length

        # mode_probs = (single, same_class_overlay, overlay, concat). Defaults
        # to the class-level values above. Pass a custom tuple to rebalance,
        # e.g. to build an overlay-specialist model once concat is handled
        # separately via segment+classify inference (see predict_mixed.py).
        self.mode_probs = mode_probs or (
            self.SINGLE_PROB, self.SAME_CLASS_OVERLAY_PROB, self.OVERLAY_PROB, self.CONCAT_PROB
        )

        with open(config.STATS_FILE) as f:
            stats = json.load(f)
        self.mean = stats["mean"]
        self.std = stats["std"]

        # IMPORTANT: for validation, generate the mixes ONCE and freeze them.
        # Without this, _make_sample() would produce a different random set
        # of mixes every epoch, making "Val Acc" not comparable across epochs
        # and making best-checkpoint selection unreliable. Training keeps
        # fresh random mixes every epoch (that's fine/desirable there).
        self._fixed_samples = None
        self.fixed_modes = None  # exposed for per-mode validation breakdown
        if not train:
            rng_state = random.getstate()
            random.seed(seed)
            raw = [self._make_sample() for _ in range(length)]
            random.setstate(rng_state)
            self._fixed_samples = [(w, l) for w, l, m in raw]
            self.fixed_modes = [m for w, l, m in raw]

    def __len__(self):
        return self.length

    def _load_wave(self, filename):
        path = os.path.join(WAVE_CACHE_DIR, filename + ".npy")
        return np.load(path)

    def _random_row(self):
        return self.df.iloc[random.randint(0, len(self.df) - 1)]

    def _rows_of_class(self, target):
        return self.df[self.df["target"] == target]

    def _make_sample(self):
        mode = random.choices(
            ["single", "same_class_overlay", "overlay", "concat"],
            weights=list(self.mode_probs),
        )[0]

        row1 = self._random_row()
        wave1 = self._load_wave(row1["filename"])
        wave1 = normalize_rms(wave1)
        labels = {int(row1["target"])}

        if mode == "single":
            wave = wave1

        elif mode == "same_class_overlay":
            # Two DIFFERENT clips of the SAME class, overlaid -- label stays
            # a single class. Teaches the model that overlapping instances of
            # one sound (e.g. two different dog barks layered) are still just
            # that one class, not evidence of a second label. Without this,
            # every overlay example the model sees during training implies
            # "2 different classes", which doesn't match how real audio works.
            same_class_rows = self._rows_of_class(row1["target"])
            if len(same_class_rows) > 1:
                row2 = same_class_rows.sample(1).iloc[0]
                # guard against (unlikely) re-picking the exact same file
                attempts = 0
                while row2["filename"] == row1["filename"] and attempts < 5:
                    row2 = same_class_rows.sample(1).iloc[0]
                    attempts += 1
            else:
                row2 = row1  # only one clip of this class in the fold -- fall back
            wave2 = self._load_wave(row2["filename"])
            wave2 = normalize_rms(wave2)
            # label unchanged: still just row1's (== row2's) class
            ratio = sample_mix_ratio()
            wave = ratio * wave1 + (1 - ratio) * wave2

        elif mode == "concat":
            row2 = self._random_row()
            wave2 = self._load_wave(row2["filename"])
            wave2 = normalize_rms(wave2)
            labels.add(int(row2["target"]))
            # Keep both clips FULL length (don't chop to half) -- the encoder's
            # global average pool at the end handles the wider resulting
            # spectrogram fine, and full-length clips match what the encoder
            # was fine-tuned on, giving it a much better chance of recognizing
            # both sounds instead of two truncated fragments.
            wave = np.concatenate([wave1, wave2])

        else:  # overlay (different classes)
            row2 = self._random_row()
            wave2 = self._load_wave(row2["filename"])
            wave2 = normalize_rms(wave2)
            labels.add(int(row2["target"]))
            # Discrete mix ratios (50:50 most common, 80:20 rare) instead of a
            # plain continuous range -- both clips are RMS-normalized first,
            # so this reflects an intentional mix balance, and never goes
            # past 80:20 so the quieter source stays at least plausibly audible.
            ratio = sample_mix_ratio()
            wave = ratio * wave1 + (1 - ratio) * wave2

        wave = wave.astype(np.float32)
        if len(wave) < MAX_SAMPLES:
            wave = np.pad(wave, (0, MAX_SAMPLES - len(wave)))
        return wave, labels, mode

    def _mel(self, wave):
        mel = librosa.feature.melspectrogram(
            y=wave,
            sr=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
        )
        mel = librosa.power_to_db(mel)
        mel = (mel - self.mean) / (self.std + 1e-6)
        return mel

    def __getitem__(self, idx):
        if self.train:
            wave, labels, _mode = self._make_sample()
        else:
            wave, labels = self._fixed_samples[idx]
        mel = self._mel(wave)

        if self.train:
            from datasets.classification_dataset import spec_augment
            mel = spec_augment(mel, config.FREQ_MASK, config.TIME_MASK)

        mel = torch.tensor(mel).float().unsqueeze(0)

        label_vec = torch.zeros(NUM_CLASSES)
        for c in labels:
            label_vec[c] = 1.0

        return mel, label_vec