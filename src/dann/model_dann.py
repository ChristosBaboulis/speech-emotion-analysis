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


# ---------------- Feature extractor (CRNN) ----------------
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),       # [32, 64, 150]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),       # [64, 32, 75]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),       # [128, 16, 37] περίπου
        )

    def forward(self, x):
        # x: [B, 1, 128, 300]
        return self.conv_block(x)   # [B, C, F, T]
    

# ---------------- DANN model ----------------
class DANNEmotionModel(nn.Module):
    def __init__(self, num_classes: int = 5, num_domains: int = 2):
        super().__init__()

        self.feature_extractor = FeatureExtractor()
        self.grl = GradientReversal()

        # infer feature dim dynamically with dummy input
        dummy = torch.zeros(1, 1, 128, 300)
        with torch.no_grad():
            feats = self.feature_extractor(dummy)   # [1, C, F, T]
            c, f, t = feats.shape[1], feats.shape[2], feats.shape[3]
            feat_dim = c * f

        hidden_size = 128
        self.rnn = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=False,
            bidirectional=True,
        )

        # emotion classifier head
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

        # domain classifier head
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_domains),
        )

    def forward(self, x, alpha: float = 0.0):
        """
        Returns:
          emotion_logits: [B, num_classes]
          domain_logits:  [B, num_domains]
        """
        feats = self.feature_extractor(x)      # [B, C, F, T]

        # collapse frequency into channels → [B, C*F, T]
        B, C, F, T = feats.shape
        feats = feats.view(B, C * F, T)

        # GRU expects [T, B, feature_dim]
        feats = feats.permute(2, 0, 1)         # [T, B, C*F]

        rnn_out, _ = self.rnn(feats)          # [T, B, 2*hidden]
        last_hidden = rnn_out[-1]             # [B, 2*hidden]

        # emotion branch
        emotion_logits = self.emotion_head(last_hidden)

        # domain branch via GRL
        reversed_feats = self.grl(last_hidden, alpha)
        domain_logits = self.domain_head(reversed_feats)

        return emotion_logits, domain_logits
