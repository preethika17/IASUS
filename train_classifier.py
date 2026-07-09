"""
Supervised end-to-end training for the base-level ESC-50 classifier.

Prerequisite: run `python preprocess_cache.py` once first.

This version adds regularization aimed directly at the overfitting signature
from the previous run (train loss -> ~0.02 while val accuracy plateaued at
80-82%, meaning the model was largely memorizing the 1,600 training clips):

  - Mixup: blends pairs of spectrograms + labels during training. Well proven
    on small datasets -- it forces smoother decision boundaries instead of
    letting the model carve out tight regions around individual training
    examples.
  - Label smoothing: prevents the model from driving predictions to extreme
    confidence on training data, which otherwise encourages overfitting.
  - Weight decay: standard L2 regularization on top of the above.

train_one_model() is factored out (not just called from main()) so
train_classifier_bagging.py can train several models with different seeds
and ensemble them, without duplicating the training loop.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from datasets.classification_dataset import ClassificationDataset
from models.classifier import AudioClassifier

LABEL_SMOOTHING = 0.0   # was 0.1 -- reverted, see note below
WEIGHT_DECAY = 0.0      # was 1e-4 -- reverted, see note below
MIXUP_ALPHA = 0.0       # was 0.2/0.3 -- DISABLED. Two tries (always-on @0.3,
                         # then 50%-of-batches @0.2) both landed at 81.0%,
                         # below the original 82.5% plain-fine-tune baseline,
                         # and more epochs didn't close the gap (val acc
                         # plateaued at 79-81% from epoch 25 on either run).
                         # Conclusion: mixing mel spectrograms linearly doesn't
                         # preserve meaningful audio structure the way mixing
                         # natural images does -- this is a legitimate negative
                         # result, not a tuning failure. Reverting to the exact
                         # recipe that got 82.5% rather than continuing to guess.
MIXUP_PROB = 0.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(seed=None, bootstrap=False):
    train_dataset = ClassificationDataset(config.TRAIN_FOLDS, train=True)
    val_dataset = ClassificationDataset(config.VAL_FOLDS, train=False)

    if bootstrap:
        # Bootstrap resample the training set (sample with replacement) --
        # this is what makes an ensemble of these models "bagging" rather
        # than just "several models with different random seeds": each one
        # sees a different resampled view of the training data.
        rng = np.random.RandomState(seed)
        n = len(train_dataset)
        indices = rng.choice(n, size=n, replace=True)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=config.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=config.NUM_WORKERS > 0,
    )
    return train_loader, val_loader


def mixup_batch(x, y, alpha, num_classes=50):
    """Returns mixed inputs and a soft target distribution (not two hard labels)."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[perm]

    y_onehot = F.one_hot(y, num_classes=num_classes).float()
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[perm]
    return mixed_x, mixed_y


def soft_cross_entropy(logits, soft_targets, smoothing=0.0):
    n_classes = logits.size(1)
    if smoothing > 0:
        soft_targets = soft_targets * (1 - smoothing) + smoothing / n_classes
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


def train_one_model(seed=0, save_path="best_classifier.pth", bootstrap=False, verbose=True):
    set_seed(seed)
    train_loader, val_loader = get_loaders(seed=seed, bootstrap=bootstrap)

    model = AudioClassifier(num_classes=50).to(config.DEVICE)

    if config.FREEZE_ENCODER:
        for param in model.encoder.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": config.ENCODER_LR},
        {"params": model.classifier.parameters(), "lr": config.HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    best_acc = 0.0

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            if MIXUP_ALPHA > 0 and np.random.rand() < MIXUP_PROB:
                mixed_x, soft_y = mixup_batch(x, y, MIXUP_ALPHA)
                output = model(mixed_x)
                loss = soft_cross_entropy(output, soft_y, smoothing=LABEL_SMOOTHING)
            else:
                output = model(x)
                loss = F.cross_entropy(output, y, label_smoothing=LABEL_SMOOTHING)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        val_acc = evaluate(model, val_loader)
        if verbose:
            print(
                f"[seed={seed}] Epoch {epoch+1}/{config.EPOCHS}  "
                f"Loss={total_loss/len(train_loader):.4f}  "
                f"Val Acc={val_acc:.2f}%"
            )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    if verbose:
        print(f"\n[seed={seed}] Best Validation Accuracy = {best_acc:.2f}%")
        print(f"[seed={seed}] Best checkpoint saved to {save_path}")

    return best_acc


def main():
    train_one_model(seed=0, save_path="best_classifier.pth")


if __name__ == "__main__":
    main()