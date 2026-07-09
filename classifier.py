import torch.nn as nn
from models.encoder import AudioEncoder


class AudioClassifier(nn.Module):
    def __init__(self, num_classes=50, dropout=0.3):
        super().__init__()
        self.encoder = AudioEncoder()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.encoder(x)
        out = self.classifier(features)
        return out