import torch
import torch.nn as nn
from torchvision.models import WeightsEnum
from torchvision.models.vision_transformer import VisionTransformer

from bottleneck import Bottleneck


class BottleneckVisionTransformer(VisionTransformer):
    def __init__(self, bottleneck: Bottleneck, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bottleneck = bottleneck
        self.use_bottleneck = True


    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        x = x.permute(0, 2, 1)

        # Bottleneck: (n, seq_len, hidden_dim) -> (n, seq_len, bottleneck_dim) -> (n, seq_len, hidden_dim)
        # Use bottleneck with pretrained weights here
        if self.use_bottleneck:
            x = self.bottleneck.encoder(x)
            x = self.bottleneck.decoder(x)

        return x

    def freeze_except_bottleneck(self):
        """Freezes all parameters except those in the bottleneck layer."""
        # Freeze conv_proj (patch embedding)
        for param in self.conv_proj.parameters():
            param.requires_grad = False

        # Freeze transformer encoder blocks
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Freeze position embedding if it exists
        if hasattr(self, 'pos_embedding'):
            self.pos_embedding.requires_grad = False

        # Freeze class token if it exists
        if hasattr(self, 'class_token'):
            self.class_token.requires_grad = False

        # Keep bottleneck trainable
        for param in self.bottleneck.parameters():
            param.requires_grad = True


    def load_pretrained_weights(self, pytorch_weights: WeightsEnum):

        super().load_state_dict(pytorch_weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True), strict=False)