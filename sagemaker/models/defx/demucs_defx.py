"""
DeFX using pretrained HDemucs as backbone.

Strategy:
  - Load pretrained HDemucs (4-source separation)
  - Freeze backbone, add a learned mixing head
  - Initialize head to pass through the "other" source
    (where clean guitar lives in Demucs's world)
  - Fine-tune head on dry/wet pairs

Input/output is stereo (2 channels) to match HDemucs natively.
Mono inputs are auto-expanded to stereo.
"""

import torch
import torch.nn as nn
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS


class DemucsDefx(nn.Module):
    """
    Pretrained HDemucs backbone with a learned mixing head
    for effect removal (wet → dry).

    Output of HDemucs: (B, 4_sources, 2_channels, samples)
    Head learns to combine sources → (B, 2, samples) stereo output.
    """

    def __init__(self, freeze_encoder: bool = True, unfreeze_decoder_layers: int = 0,
                 unfreeze_encoder_layers: int = 0):
        super().__init__()

        # Load pretrained backbone
        self.backbone = HDEMUCS_HIGH_MUSDB_PLUS.get_model()
        self.sample_rate = HDEMUCS_HIGH_MUSDB_PLUS.sample_rate  # 44100

        # Mixing head: (B, 8, T) → (B, 2, T) stereo
        # 8 = 4 sources × 2 channels
        self.head = nn.Sequential(
            nn.Conv1d(8, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 2, kernel_size=3, padding=1),
        )

        # Freeze entire backbone first
        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Selectively unfreeze the last N decoder layers
        if unfreeze_decoder_layers > 0:
            self._unfreeze_decoder_layers(unfreeze_decoder_layers)

        # Selectively unfreeze the last N encoder layers
        if unfreeze_encoder_layers > 0:
            self._unfreeze_encoder_layers(unfreeze_encoder_layers)

        # Initialize head to pass through "other" source
        self._init_head_from_other()

        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Frozen: {frozen:,} | Trainable: {trainable:,}")

    def _unfreeze_decoder_layers(self, n: int):
        """Unfreeze the last N layers of both freq and time decoders."""
        for decoder_name in ["freq_decoder", "time_decoder"]:
            decoder = getattr(self.backbone, decoder_name)
            num_layers = len(decoder)
            for i in range(max(0, num_layers - n), num_layers):
                for param in decoder[i].parameters():
                    param.requires_grad = True
                print(f"  Unfroze {decoder_name}[{i}]")

    def _unfreeze_encoder_layers(self, n: int):
        """Unfreeze the last N layers of both freq and time encoders."""
        for encoder_name in ["freq_encoder", "time_encoder"]:
            encoder = getattr(self.backbone, encoder_name)
            num_layers = len(encoder)
            for i in range(max(0, num_layers - n), num_layers):
                for param in encoder[i].parameters():
                    param.requires_grad = True
                print(f"  Unfroze {encoder_name}[{i}]")

    def _init_head_from_other(self):
        """
        Initialize head to pass through the 'other' source as-is.

        Reshaped backbone output (B, 8, T) channel layout:
          0=drums_L, 1=drums_R, 2=bass_L, 3=bass_R,
          4=other_L, 5=other_R, 6=vocals_L, 7=vocals_R

        We want stereo output (B, 2, T) where:
          out[0] = other_L (channel 4)
          out[1] = other_R (channel 5)
        """
        with torch.no_grad():
            # First layer: Conv1d(8→64, k=7) — select 'other' channels
            first = self.head[0]
            first.weight.zero_()
            first.bias.zero_()
            center = 3  # center tap of kernel_size=7
            # Route other_L to output channels 0,2,4...
            # Route other_R to output channels 1,3,5...
            for i in range(0, 64, 2):
                first.weight[i, 4, center] = 1.0    # other_L
                first.weight[i + 1, 5, center] = 1.0  # other_R

            # Middle layers: small init so they start near identity
            for i in range(2, len(self.head) - 1, 2):
                conv = self.head[i]
                nn.init.xavier_uniform_(conv.weight, gain=0.1)
                conv.bias.zero_()

            # Last layer: Conv1d(32→2, k=3) — route to stereo
            last = self.head[-1]
            last.weight.zero_()
            last.bias.zero_()
            center = 1  # center tap of kernel_size=3
            # Average even channels → left, odd channels → right
            for i in range(0, 32, 2):
                last.weight[0, i, center] = 1.0 / 16    # → left
                last.weight[1, i + 1, center] = 1.0 / 16  # → right

        print("  Head initialized: passes through 'other' source")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, samples) mono or (B, 2, samples) stereo
        Returns:
            (B, 2, samples) stereo prediction
        """
        # Ensure stereo input for HDemucs
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, 2, -1)
        elif x.dim() == 3 and x.shape[1] == 1:
            x = x.expand(-1, 2, -1)

        # Run backbone — use no_grad only if fully frozen
        if self.trainable_params == sum(p.numel() for p in self.head.parameters()):
            with torch.no_grad():
                sources = self.backbone(x)
            sources = sources.detach()
        else:
            sources = self.backbone(x)

        B, S, C, T = sources.shape  # (B, 4, 2, T)
        mixed = sources.reshape(B, S * C, T)  # (B, 8, T)

        return self.head(mixed)  # (B, 2, T)

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
