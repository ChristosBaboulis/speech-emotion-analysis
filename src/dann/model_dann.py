import torch
import torch.nn as nn
from torch.autograd import Function


# ---------------- Gradient Reversal Layer ----------------
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, alpha):
        return GradReverse.apply(x, alpha)


# ---------------- DANN Model ----------------
class DANNEmotionModel(nn.Module):
    """
    DANN model: 2D CNN feature extractor + BiLSTM + dual classifier heads.
    Extends EmotionCRNN with domain adaptation via gradient reversal.
    Input: [B, 1, n_mels, time] = [B, 1, 128, 300]
    """

    def __init__(self, num_classes: int = 5, num_domains: int = 2):
        super().__init__()

        # Convolutional feature extractor (same as EmotionCRNN)
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

        # BiLSTM over time dimension (same as EmotionCRNN)
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Gradient reversal layer for domain adaptation
        self.grl = GradientReversal()

        # Emotion classifier head (same structure as EmotionCRNN.classifier)
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

        # Domain classifier head (for domain adaptation)
        self.domain_head = nn.Sequential(
            nn.Linear(self.lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_domains),
        )

    def forward(self, x: torch.Tensor, alpha: float = 0.0):
        """
        Args:
            x: Input tensor [B, 1, 128, 300]
            alpha: Gradient reversal strength (0.0 = no reversal, higher = stronger)
        
        Returns:
            emotion_logits: [B, num_classes]
            domain_logits:  [B, num_domains]
        """
        # x: [B, 1, 128, 300]
        x = self.conv(x)  # [B, C, F, T]
        B, C, F, T = x.size()

        # Treat time as sequence dimension: [B, T, C * F] (same as EmotionCRNN)
        x = x.permute(0, 3, 1, 2).contiguous()   # [B, T, C, F]
        x = x.view(B, T, C * F)                  # [B, T, 128*16]

        # BiLSTM (same as EmotionCRNN)
        lstm_out, _ = self.lstm(x)               # [B, T, 2*hidden]

        # Temporal pooling: mean over time (same as EmotionCRNN)
        x = lstm_out.mean(dim=1)                 # [B, 2*hidden]

        # Emotion classification branch (same as EmotionCRNN)
        emotion_logits = self.classifier(x)      # [B, num_classes]

        # Domain classification branch via gradient reversal (DANN extension)
        reversed_feats = self.grl(x, alpha)
        domain_logits = self.domain_head(reversed_feats)  # [B, num_domains]

        return emotion_logits, domain_logits


if __name__ == "__main__":
    # Quick shape test
    model = DANNEmotionModel(num_classes=5, num_domains=2)
    dummy = torch.randn(8, 1, 128, 300)
    emotion_out, domain_out = model(dummy, alpha=0.5)
    print("Emotion logits shape:", emotion_out.shape)
    print("Domain logits shape:", domain_out.shape)
