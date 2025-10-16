import torch
import torch.nn as nn
import torch.nn.init as init





# ---- Base class ----
class Bottleneck(nn.Module):
    """Base class for all bottlenecks."""
    def encoder(self, x):
        return x

    def decoder(self, x):
        return x

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ---- Registry ----
BNECK_REGISTRY = {}

def register_bottleneck(name):
    """Decorator to register a bottleneck class."""
    def decorator(cls):
        BNECK_REGISTRY[name] = cls
        return cls
    return decorator


def create_bottleneck(bottleneck_type: str, **kwargs) -> Bottleneck:
    """Factory function that instantiates a registered bottleneck type."""
    if bottleneck_type not in BNECK_REGISTRY:
        raise ValueError(f"Unknown bottleneck type '{bottleneck_type}'. "
                         f"Available: {list(BNECK_REGISTRY.keys())}")
    return BNECK_REGISTRY[bottleneck_type](**kwargs)

# ---- Identity Bottleneck ----
@register_bottleneck("identity")
class IdentityBottleneck(Bottleneck):
    """Pass-through bottleneck (no compression)."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def encoder(self, x):
        return x

    def decoder(self, x):
        return x

    def forward(self, x):
        return x


# ---- Single Linear Bottleneck ----
@register_bottleneck("bottleneck")
class SingleLinearBottleneck(Bottleneck):
    """Linear compression + expansion."""
    def __init__(self, embedding_dim, bottleneck_dim, dropout=0.01):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, bottleneck_dim)
        self.ln1 = nn.LayerNorm(bottleneck_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(bottleneck_dim, embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.relu = nn.ReLU()

        init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        init.xavier_uniform_(self.fc2.weight)
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)

    def encoder(self, x):
        return self.dropout(self.relu(self.ln1(self.fc1(x))))

    def decoder(self, x):
        return self.ln2(self.fc2(x))


# ---- Double Linear Bottleneck ----
@register_bottleneck("double_linear")
class DoubleLinearBottleneck(Bottleneck):
    """
    Two-layer encoder and decoder for deeper compression.
    encoder: Linear -> ReLU -> Linear -> ReLU
    decoder: Linear -> ReLU -> Linear
    """
    def __init__(self, embedding_dim, bottleneck_dim, hidden_division=2, dropout=0.01):
        super().__init__()
        hidden_dim = bottleneck_dim // hidden_division

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
        )

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                init.zeros_(m.bias)

    def encoder(self, x):
        return self.enc(x)

    def decoder(self, x):
        return self.dec(x)


# ---- Nonlinear Conv Bottleneck ----
@register_bottleneck("conv")
class NonlinearConvBottleneck(Bottleneck):
    """
    1D convolutional bottleneck (nonlinear, local context).
    Assumes input shape: [batch, seq_len, embedding_dim].
    """
    def __init__(self, embedding_dim, bottleneck_dim, kernel_size=3, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(embedding_dim, bottleneck_dim, kernel_size, padding=padding),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(bottleneck_dim, embedding_dim, kernel_size, padding=padding),
            nn.BatchNorm1d(embedding_dim),
        )

    def encoder(self, x):
        # (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)
        z = self.encoder_conv(x)
        return z.transpose(1, 2)
