"""
Hybrid inference for mixed audio, exploiting the actual structure of each mode:

  - CONCAT mode: the two source clips sit back-to-back with a clean boundary
    at the 5s mark (since we concat full-length clips). No need to guess both
    sounds from one blended representation -- just split at 5s and run the
    strong level-1 single-label classifier (82.5% acc) on each half
    independently. This should beat the joint multi-label model by a wide
    margin on concat samples, with ZERO retraining.

  - OVERLAY mode: the two sources are genuinely mixed in time, so there's no
    clean split point. This is the one case that actually needs a multi-label
    model -- use best_multilabel_classifier.pth with top-2-capped decoding.

Usage:
    python predict_mixed.py --mode concat  --file1 path/to/a.wav --file2 path/to/b.wav
    python predict_mixed.py --mode overlay --file1 path/to/a.wav --file2 path/to/b.wav --ratio 0.5
"""
import argparse
import json
import os
import numpy as np
import librosa
import torch
import pandas as pd

import config
from models.classifier import AudioClassifier
from datasets.mixed_dataset import normalize_rms, NUM_CLASSES, MAX_SAMPLES
from scene_generator import generate_scenario

# From the FINAL seeded train_multilabel.py run (mode_probs = 15/15/55/15,
# seed=42): top-2-capped sweep found best combo (t1=0.15, t2=0.5) ->
# 50.12% subset accuracy on the blended val set; overlay-only accuracy
# 38.87%, same_class_overlay 67.52%. This is the locked-in final model.
OVERLAY_T1 = 0.15
OVERLAY_T2 = 0.50


def get_class_names():
    df = pd.read_csv(config.CSV_FILE)
    return dict(zip(df["target"], df["category"]))


def load_wave(path):
    y, sr = librosa.load(path, sr=config.SAMPLE_RATE)
    y = librosa.util.fix_length(y, size=config.NUM_SAMPLES)
    return y.astype(np.float32)


def wave_to_mel_tensor(wave, mean, std):
    mel = librosa.feature.melspectrogram(
        y=wave, sr=config.SAMPLE_RATE, n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH, n_mels=config.N_MELS,
    )
    mel = librosa.power_to_db(mel)
    mel = (mel - mean) / (std + 1e-6)
    return torch.tensor(mel).float().unsqueeze(0).unsqueeze(0)


def load_stats():
    import json
    with open(config.STATS_FILE) as f:
        stats = json.load(f)
    return stats["mean"], stats["std"]


def predict_concat(file1, file2):
    """Split cleanly at 5s, classify each half with the level-1 model."""
    mean, std = load_stats()
    model = AudioClassifier(num_classes=NUM_CLASSES).to(config.DEVICE)
    model.load_state_dict(torch.load("best_classifier.pth", map_location=config.DEVICE))
    model.eval()

    class_names = get_class_names()
    predictions = []

    for path in [file1, file2]:
        wave = load_wave(path)
        x = wave_to_mel_tensor(wave, mean, std).to(config.DEVICE)
        with torch.no_grad():
            logits = model(x)
            pred_class = logits.argmax(dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0, pred_class].item()
        predictions.append((pred_class, class_names.get(pred_class, str(pred_class)), confidence))

    print("Concat-mode prediction (segment + classify):")
    for path, (idx, name, conf) in zip([file1, file2], predictions):
        print(f"  {path}  ->  {name}  (confidence={conf:.2f})")

    names = [p[1] for p in predictions]
    print(f"\nScene description: {generate_scenario(names)}")

    return [p[0] for p in predictions]


def load_per_class_thresholds():
    """If train_multilabel.py's per-class threshold tuning found a real win,
    it saves them here -- use them automatically if present, since they beat
    the global (t1, t2) pair (49.12% vs 48.50% subset accuracy, confirmed)."""
    path = "overlay_per_class_thresholds.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data["base_t1"], torch.tensor(data["per_class_t2"])
    return None, None


def predict_overlay(file1, file2, ratio=0.5, t1=OVERLAY_T1, t2=OVERLAY_T2):
    """Mix in the waveform domain, run the multi-label model with top-2-capped decoding."""
    mean, std = load_stats()
    model = AudioClassifier(num_classes=NUM_CLASSES).to(config.DEVICE)
    model.load_state_dict(torch.load("best_multilabel_classifier.pth", map_location=config.DEVICE))
    model.eval()

    wave1 = normalize_rms(load_wave(file1))
    wave2 = normalize_rms(load_wave(file2))
    mixed = ratio * wave1 + (1 - ratio) * wave2

    # CRITICAL: MixedAudioDataset pads every training sample (including
    # overlay) to MAX_SAMPLES (10s) with trailing silence, because concat
    # samples need that width and all samples must match for batching. That
    # means the model learned overlay audio as "5s of real content + 5s of
    # trailing silence" -- feeding it a bare 5s mix (no padding) at inference,
    # like the previous version of this function did, is a real train/test
    # mismatch and was very likely why overlay predictions were unreliable.
    if len(mixed) < MAX_SAMPLES:
        mixed = np.pad(mixed, (0, MAX_SAMPLES - len(mixed)))

    x = wave_to_mel_tensor(mixed, mean, std).to(config.DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0].cpu()

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)

    base_t1, per_class_t2 = load_per_class_thresholds()
    using_per_class = per_class_t2 is not None
    if using_per_class:
        effective_t1 = base_t1
        effective_t2 = per_class_t2[sorted_idx[1]].item()
    else:
        effective_t1 = t1
        effective_t2 = t2

    preds = []
    if sorted_probs[0] >= effective_t1:
        preds.append(sorted_idx[0].item())
    if sorted_probs[1] >= effective_t2:
        preds.append(sorted_idx[1].item())

    class_names = get_class_names()
    mode_desc = "per-class thresholds" if using_per_class else "global top-2-capped"
    print(f"Overlay-mode prediction (multi-label model, {mode_desc}):")
    if not preds:
        print("  No label passed the top-1 threshold -- try lowering it")
    for idx in preds:
        print(f"  {class_names.get(idx, str(idx))}  (prob={probs[idx]:.2f})")
    print(f"  (top-5 raw probs for reference: "
          f"{[(class_names.get(i.item(), str(i.item())), round(p.item(), 2)) for i, p in zip(sorted_idx[:5], sorted_probs[:5])]})")

    if preds:
        names = [class_names.get(idx, str(idx)) for idx in preds]
        print(f"\nScene description: {generate_scenario(names)}")

    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["concat", "overlay"], required=True)
    parser.add_argument("--file1", required=True)
    parser.add_argument("--file2", required=True)
    parser.add_argument("--ratio", type=float, default=0.5, help="overlay mix ratio for file1")
    args = parser.parse_args()

    if args.mode == "concat":
        predict_concat(args.file1, args.file2)
    else:
        predict_overlay(args.file1, args.file2, ratio=args.ratio)


if __name__ == "__main__":
    main()