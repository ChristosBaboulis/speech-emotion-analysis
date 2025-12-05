import torch
import torch.nn as nn


class EmotionCNN(nn.Module):
    """
    CRNN model: 2D CNN feature extractor + BiLSTM + FC classifier.
    Input: [B, 1, n_mels, time] = [B, 1, 128, 300]
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),      # [B, 32, 64, 150]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),      # [B, 64, 32, 75]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),      # [B, 128, 16, 37]
        )

        # After conv: C = 128, F = 16, T ≈ 37  → feature_dim = 128 * 16
        self.freq_bins = 16
        self.lstm_input_dim = 128 * self.freq_bins
        self.lstm_hidden = 128

        # BiLSTM over time dimension
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Classifier on top of pooled LSTM outputs
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 128, 300]
        x = self.conv(x)  # [B, C, F, T]
        B, C, F, T = x.size()

        # Treat time as sequence dimension: [B, T, C * F]
        x = x.permute(0, 3, 1, 2).contiguous()   # [B, T, C, F]
        x = x.view(B, T, C * F)                  # [B, T, 128*16]

        # BiLSTM
        lstm_out, _ = self.lstm(x)               # [B, T, 2*hidden]

        # Temporal pooling: mean over time
        x = lstm_out.mean(dim=1)                 # [B, 2*hidden]

        # Classifier
        logits = self.classifier(x)              # [B, num_classes]
        return logits


if __name__ == "__main__":
    # Quick shape test
    model = EmotionCNN(num_classes=5)
    dummy = torch.randn(8, 1, 128, 300)
    out = model(dummy)
    print("Output shape:", out.shape)
