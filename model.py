import torch
import torch.nn as nn
from utils import CHARS

# Character classes: 0 = blank
NUM_CLASSES = 37  # 0-9 + A-Z + blank

class CRNN(nn.Module):
    def __init__(self, num_classes=len(CHARS)+1):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 32x128 → 16x64

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 16x64 → 8x32

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),     # 8x32 → 4x32

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),     # 4x32 → 2x32

            nn.Conv2d(512, 512, 2, 1, 0),  # 2x32 → 1x31
            nn.ReLU()
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)            # (B, 512, 1, W)
        x = x.squeeze(2)           # (B, 512, W)
        x = x.permute(0, 2, 1)     # (B, W, 512)

        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
