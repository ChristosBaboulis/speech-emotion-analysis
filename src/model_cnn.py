import torch
import torch.nn as nn


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [16, 64, 150]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [32, 32, 75]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [64, 16, 37]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                      # 64 * 16 * 37
            nn.Linear(64 * 16 * 37, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Quick shape check
    model = EmotionCNN(num_classes=5)
    dummy = torch.randn(8, 1, 128, 300)
    out = model(dummy)
    print("Output shape:", out.shape)  # [8, 5]
