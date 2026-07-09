# IASUS — Audio Scene & Event Classification

Deep learning pipeline for classifying environmental sounds using the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset. Includes a supervised single-label classifier and a multi-label scene classifier (via synthetic scene generation), plus supporting tools for evaluation and interpretability.

## Overview

This project trains models to recognize environmental sounds (e.g. dog barks, rain, helicopters, sneezing) from audio. It supports two tasks:

1. **Single-label classification** — classify a single audio clip into one of 50 ESC-50 categories.
2. **Multi-label scene classification** — classify audio "scenes" composed of multiple overlapping sound events, using synthetically generated mixed-audio scenes.

## Project Structure

```
IASUS/
├── augmentations/          # Audio/spectrogram augmentation methods
├── data/
│   ├── audio/               # Raw ESC-50 audio clips
│   ├── esc50.csv             # ESC-50 metadata (labels, folds)
│   ├── mel_cache/            # Cached mel-spectrograms (regenerated, not tracked in git)
│   └── wave_cache/           # Cached waveform data (regenerated, not tracked in git)
├── datasets/
│   ├── classification_dataset.py   # Single-label dataset loader
│   └── mixed_dataset.py            # Multi-label / mixed-scene dataset loader
├── models/
│   ├── encoder.py           # Shared audio encoder backbone
│   └── classifier.py        # Classifier head (encoder + linear layer)
├── config.py                 # Central config: hyperparameters, paths, device
├── preprocess_cache.py       # Precompute + cache mel-spectrograms
├── preprocess_wave_cache.py  # Precompute + cache raw waveforms
├── scene_generator.py        # Generates synthetic multi-event audio scenes
├── train_classifier.py       # Trains the single-label classifier
├── train_multilabel.py       # Trains the multi-label scene classifier
├── train_classifier_bagging.py  # Ensemble/bagging variant of classifier training
├── predict_single_file.py    # Run inference on a single audio file
├── predict_mixed.py          # Run inference on a multi-event audio scene
├── visualize_tsne.py         # t-SNE visualization of learned embeddings
├── gradcam.py                 # Grad-CAM visualization for model interpretability
├── bc_utils.py / utils.py     # Shared helper utilities
├── best_classifier.pth        # Best single-label checkpoint (not tracked in git — see below)
└── best_multilabel_classifier.pth  # Best multi-label checkpoint (not tracked in git — see below)
```

## Setup

```bash
pip install -r requirements.txt
```

Download and extract the ESC-50 dataset into `data/audio/` (or place `ESC50.zip` in `data/` and extract it there).

## Usage

**1. Preprocess audio into cached spectrograms/waveforms**
```bash
python preprocess_cache.py
python preprocess_wave_cache.py
```

**2. Train the single-label classifier**
```bash
python train_classifier.py
```
Saves the best checkpoint to `best_classifier.pth`. Both the encoder and classifier head are trained from scratch (no separate pretraining stage).

**3. Train the multi-label scene classifier**
```bash
python scene_generator.py      # generate synthetic multi-event scenes
python train_multilabel.py
```
Saves the best checkpoint to `best_multilabel_classifier.pth`.

**4. Run inference**
```bash
python predict_single_file.py --file path/to/audio.wav
python predict_mixed.py --file path/to/scene.wav
```

**5. Visualize / interpret**
```bash
python visualize_tsne.py     # embedding visualization
python gradcam.py            # Grad-CAM heatmaps
```

## Model Checkpoints

`.pth` files are not tracked in this repository (see `.gitignore`) since they exceed practical git size limits. Trained checkpoints:

- `best_classifier.pth` — single-label classifier
- `best_multilabel_classifier.pth` — multi-label scene classifier

> Add a download link here (Google Drive / HuggingFace Hub / GitHub Releases) if you're distributing pretrained weights.

## Notes

- `config.py` controls train/val fold splits, batch size, learning rates, and device selection.
- Regularization experiments (mixup, label smoothing, weight decay) were tested in `train_classifier.py` and reverted after not improving on the baseline 82.5% validation accuracy — see inline comments in that file for details.
- An earlier self-supervised contrastive pretraining stage (SimCLR-style) was explored and later removed in favor of training the classifier end-to-end from scratch, which gave better results on this dataset size.
- Cache folders (`mel_cache/`, `wave_cache/`) and `__pycache__/`/`.ipynb_checkpoints/` are regenerated automatically and excluded from version control.

## License

Add your license here.
