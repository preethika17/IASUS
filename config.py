import torch

# Dataset paths
DATA_DIR = "data/audio"
CSV_FILE = "data/esc50.csv"
CACHE_DIR = "data/mel_cache"          # precomputed mel spectrograms (.npy)
STATS_FILE = "data/mel_cache/stats.json"  # dataset-wide mean/std for normalization

# Audio settings
SAMPLE_RATE = 16000
DURATION = 5  # seconds
NUM_SAMPLES = SAMPLE_RATE * DURATION

# Mel Spectrogram settings
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

# ESC-50 official folds (1-5). Use 4 for training, 1 held out for validation.
# Swap which fold is held out across runs if you want full 5-fold CV later.
TRAIN_FOLDS = [1, 2, 3, 4]
VAL_FOLDS = [5]

# Training settings (supervised classification, end-to-end fine-tuning)
BATCH_SIZE = 32
EPOCHS = 40  # reverted -- matches the exact config that achieved 82.5%
ENCODER_LR = 1e-4      # smaller LR for pretrained backbone
HEAD_LR = 1e-3         # larger LR for the new classification head
NUM_WORKERS = 2  # was 4 -- your system suggested max 2; 4 risks the slowdown/freeze the warning mentioned
FREEZE_ENCODER = False  # <-- key change: fine-tune, don't freeze

# SpecAugment
FREQ_MASK = 15
TIME_MASK = 20
SPEC_AUGMENT_TRAIN_ONLY = True

# Contrastive Learning (only used if/when you revisit SimCLR pretraining)
SIMCLR_BATCH_SIZE = 256   # SimCLR needs large batches; 32 is too small to learn from
TEMPERATURE = 0.5
PROJECTION_DIM = 128
LEARNING_RATE = 1e-3
EPOCHS_SIMCLR = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")