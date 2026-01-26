import torch
import torch.nn as nn

# Character classes: 0 = blank
NUM_CLASSES = 37  # 0-9 + A-Z + blank

class CRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d((2, 1))
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.cnn(x)  # (B, C, H, W)

        # Force height = 1
        x = torch.nn.functional.adaptive_avg_pool2d(
            x, (1, x.size(3))
        )

        x = x.squeeze(2)      # (B, C, W)
        x = x.permute(0, 2, 1)  # (B, W, C)

        x, _ = self.rnn(x)
        x = self.fc(x)

        return x.log_softmax(2)

