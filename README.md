# IASUS – Intelligent Audio Scene Understanding System

IASUS (Intelligent Audio Scene Understanding System) is a deep learning-based framework for recognizing environmental sounds and understanding complex audio scenes. The system is designed to identify both **individual sounds** and **multiple simultaneous or sequential sound events**, making it a step towards real-world audio scene understanding.

Built using the **ESC-50 environmental sound dataset**, the project leverages **Mel Spectrogram representations** and a **ResNet18** backbone to learn robust audio features. Beyond simple sound classification, IASUS extends to recognizing mixed audio signals and generating meaningful textual descriptions of detected sound events.

---

## Motivation

Environmental sound recognition has applications in smart surveillance, healthcare, robotics, autonomous systems, and assistive technologies. While traditional classifiers focus on identifying a single sound, real-world environments often contain multiple overlapping or sequential sound events.

This project aims to bridge that gap by developing a multi-stage audio understanding pipeline capable of handling increasingly complex acoustic scenarios.

---

## Project Pipeline

The system is organized into four stages:

### **Stage 1 – Single-label Audio Classification**

A supervised ResNet18 classifier is trained on the ESC-50 dataset to recognize individual environmental sounds from Mel Spectrograms.

**Performance**
- **Validation Accuracy:** **83.0%**

---

### **Stage 2 – Sequential Audio Recognition**

Two different audio clips are concatenated to simulate sequential sound events. Since the boundary between the clips is known, the audio is segmented and classified independently.

**Performance**
- **Accuracy:** **68%+**

---

### **Stage 3 – Multi-label Audio Classification**

To simulate realistic environments, two audio clips are overlaid and treated as simultaneous sound events. A dedicated multi-label classifier predicts all sounds present in the mixture.

**Performance**
- **Exact Match Accuracy:** **67.5%** (Same-Class Overlay)
- **F1 Score:** **74.5%** (Different-Class Overlay)

---

### **Stage 4 – Audio Scene Description**

The predicted sound labels are converted into natural-language descriptions using a deterministic rule-based scene generation module, producing interpretable scene summaries.

---

# Model Architecture

```
Audio Waveform
        │
        ▼
Audio Preprocessing
        │
        ▼
Mel Spectrogram Generation
        │
        ▼
ResNet18 Feature Extractor
        │
        ▼
Classification Head
        │
        ▼
Single / Multi-label Predictions
        │
        ▼
Scene Description Generator
```

---

# Repository Structure

```
IASUS/
│
├── augmentations/
├── datasets/
├── losses/
├── models/
│
├── preprocess_cache.py
├── preprocess_wave_cache.py
├── train_classifier.py
├── train_multilabel.py
├── predict_mixed.py
├── predict_single_file.py
├── scene_generator.py
├── config.py
├── requirements.txt
└── README.md
```

---

# Key Features

- Environmental sound classification
- Single-label learning
- Multi-label sound recognition
- Sequential and overlapping audio understanding
- Mel Spectrogram feature extraction
- ResNet18-based deep learning model
- Audio preprocessing and augmentation
- Rule-based scene description generation

---

# Results

| Task | Performance |
|------|-------------|
| Single-label Classification | **83.0% Validation Accuracy** |
| Sequential Audio Recognition | **68%+ Accuracy** |
| Same-Class Overlay Recognition | **67.5% Exact Match Accuracy** |
| Multi-label Audio Recognition | **74.5% F1 Score** |

---

# Technologies Used

- Python
- PyTorch
- TorchAudio
- Librosa
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download Dataset

Download the **ESC-50** dataset and place the audio files inside the `data/` directory as specified in `config.py`.

### Train the Single-label Classifier

```bash
python train_classifier.py
```

### Train the Multi-label Classifier

```bash
python train_multilabel.py
```

### Run Predictions

```bash
python predict_mixed.py --mode concat --file1 file1.wav --file2 file2.wav

python predict_mixed.py --mode overlay --file1 file1.wav --file2 file2.wav

python predict_single_file.py --file audio.wav
```

---

# Highlights

- Achieved **83% validation accuracy** on the ESC-50 environmental sound classification benchmark.
- Designed a multi-stage pipeline capable of understanding single, sequential, and overlapping audio events.
- Developed a dedicated multi-label recognition framework for mixed audio.
- Generated interpretable scene descriptions from detected sound events.
- Evaluated multiple training strategies and optimized the system through iterative experimentation.

---

# Future Improvements

- Automatic detection of audio composition mode.
- Support for more than two simultaneous sound events.
- Improved robustness for real-world recordings.
- Integration of transformer-based audio encoders.
- Real-time deployment for streaming audio.

---
