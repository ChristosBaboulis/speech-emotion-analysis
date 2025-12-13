import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import Wav2Vec2Model


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


# ---------------- Wav2Vec2 DANN Model ----------------
class Wav2VecDANNEmotionModel(nn.Module):
    """
    Wav2Vec2-based DANN model for domain adaptation in emotion recognition.
    Uses pre-trained Wav2Vec2 encoder with domain adversarial training.
    
    Input: Raw audio waveforms [B, num_samples] at 16kHz
    Output: Emotion logits [B, num_classes], Domain logits [B, num_domains]
    """

    def __init__(self, num_classes: int = 5, num_domains: int = 2, 
                 model_name: str = "facebook/wav2vec2-base"):
        super().__init__()

        # Load pre-trained Wav2Vec2 encoder
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        
        # Wav2Vec2-base hidden size is 768
        self.hidden_size = self.wav2vec.config.hidden_size  # 768 for base
        
        # Gradient reversal layer for domain adaptation
        self.grl = GradientReversal()
        
        # Emotion classifier head (same structure as Wav2VecEmotionModel)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # Domain classifier head (for domain adaptation)
        self.domain_head = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_domains),
        )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None, 
                alpha: float = 0.0) -> tuple:
        """
        Args:
            input_values: Raw audio waveforms [B, num_samples]
            attention_mask: Optional attention mask [B, num_samples]
            alpha: Gradient reversal strength (0.0 = no reversal, higher = stronger)
        
        Returns:
            emotion_logits: [B, num_classes]
            domain_logits:  [B, num_domains]
        """
        # Wav2Vec2 encoder
        outputs = self.wav2vec(input_values, attention_mask=attention_mask)
        
        # Temporal pooling: mean over sequence dimension
        pooled = outputs.last_hidden_state.mean(dim=1)  # [B, hidden_size]
        
        # Emotion classification branch (standard forward)
        emotion_logits = self.classifier(pooled)  # [B, num_classes]

        # Domain classification branch via gradient reversal (DANN extension)
        reversed_feats = self.grl(pooled, alpha)
        domain_logits = self.domain_head(reversed_feats)  # [B, num_domains]

        return emotion_logits, domain_logits


if __name__ == "__main__":
    # Quick shape test
    model = Wav2VecDANNEmotionModel(num_classes=5, num_domains=2)
    
    batch_size = 2
    num_samples = 160000
    dummy_audio = torch.randn(batch_size, num_samples)
    
    emotion_out, domain_out = model(dummy_audio, alpha=0.5)
    print(f"Emotion logits shape: {emotion_out.shape}")
    print(f"Domain logits shape: {domain_out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

