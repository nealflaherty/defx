"""
DeFX: STFT-domain U-Net for guitar effect removal (wet → dry).

Architecture:
  - STFT front-end (complex spectrogram)
  - Encoder: strided Conv2D blocks with GLU activation
  - Bottleneck: BiLSTM for temporal context
  - Decoder: transposed Conv2D with skip connections
  - Output: magnitude mask × input magnitude, recombine with phase
  - iSTFT back to waveform
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLUBlock(nn.Module):
    """Conv2D → BatchNorm → GLU (Gated Linear Unit)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 2):
        super().__init__()
        # GLU halves channels, so we double the output
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel, stride=stride,
                              padding=kernel // 2)
        self.bn = nn.BatchNorm2d(out_ch * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        return F.glu(x, dim=1)  # split along channel dim


class DecoderBlock(nn.Module):
    """TransposedConv2D → BatchNorm → GLU, with skip connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 2):
        super().__init__()
        # in_ch is doubled because of skip concatenation
        self.tconv = nn.ConvTranspose2d(in_ch, out_ch * 2, kernel, stride=stride,
                                        padding=kernel // 2,
                                        output_padding=stride - 1)
        self.bn = nn.BatchNorm2d(out_ch * 2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, skip], dim=1)
        x = self.bn(self.tconv(x))
        return F.glu(x, dim=1)


class DeFXNet(nn.Module):
    """
    STFT-domain U-Net for effect removal.

    Args:
        n_fft: STFT window size
        hop_length: STFT hop
        channels: list of channel sizes for encoder layers
                  e.g. [32, 64, 128, 256]
        lstm_layers: number of BiLSTM layers in bottleneck
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        channels: list[int] | None = None,
        lstm_layers: int = 2,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        if channels is None:
            channels = [32, 64, 128, 256]

        self.channels = channels
        self.window = None  # lazily created on correct device

        # Input: magnitude spectrogram (1 channel)
        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = 1
        for ch in channels:
            self.encoders.append(GLUBlock(in_ch, ch))
            in_ch = ch

        # Bottleneck BiLSTM
        # After encoder, shape is (B, C, T', F')
        # We reshape to (B*F', T', C) for LSTM along time axis
        self.lstm = nn.LSTM(
            input_size=channels[-1],
            hidden_size=channels[-1],
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Project BiLSTM output back to channels[-1]
        self.lstm_proj = nn.Linear(channels[-1] * 2, channels[-1])

        # Decoder (mirror of encoder)
        # Skips saved before each encoder, so:
        #   skip[0] = magnitude (1ch)
        #   skip[1] = enc0 output (channels[0])
        #   skip[2] = enc1 output (channels[1])
        #   ...
        # Decoder processes from deepest to shallowest.
        skip_chs = [1] + channels[:-1]  # [1, 32, 64] for channels=[32,64,128]
        # Decoder outputs mirror skip channels (so concat works at next level)
        dec_outputs = list(reversed(skip_chs))  # [64, 32, 1]
        dec_skip_chs = list(reversed(skip_chs))  # [64, 32, 1]

        self.decoders = nn.ModuleList()
        dec_in = channels[-1]  # bottleneck output channels
        for skip_ch, out_ch in zip(dec_skip_chs, dec_outputs):
            self.decoders.append(DecoderBlock(dec_in + skip_ch, out_ch))
            dec_in = out_ch

        # Final 1x1 conv to produce single-channel mask
        self.mask_conv = nn.Conv2d(dec_in, 1, kernel_size=1)

        # Sigmoid to produce mask in [0, 1]
        self.mask_activation = nn.Sigmoid()

    def _get_window(self, device: torch.device) -> torch.Tensor:
        if self.window is None or self.window.device != device:
            self.window = torch.hann_window(self.n_fft, device=device)
        return self.window

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, samples) → complex spectrogram (B, 1, T, F)."""
        window = self._get_window(x.device)
        spec = torch.stft(x, self.n_fft, self.hop_length,
                          window=window, return_complex=True)
        # spec: (B, F, T) → (B, 1, T, F)
        return spec.permute(0, 2, 1).unsqueeze(1)

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """Complex spectrogram (B, 1, T, F) → waveform (B, samples)."""
        window = self._get_window(spec.device)
        # (B, 1, T, F) → (B, F, T)
        spec = spec.squeeze(1).permute(0, 2, 1)
        return torch.istft(spec, self.n_fft, self.hop_length,
                           window=window, length=length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: wet waveform (B, samples)
        Returns:
            predicted dry waveform (B, samples)
        """
        orig_length = x.shape[-1]

        # STFT
        spec = self.stft(x)  # (B, 1, T, F) complex
        magnitude = spec.abs()
        phase = spec / (magnitude + 1e-8)

        # Encoder with skip connections
        skips = []
        h = magnitude
        for enc in self.encoders:
            skips.append(h)
            h = enc(h)

        # Bottleneck: BiLSTM along time axis
        B, C, T, Fr = h.shape
        # Reshape: merge batch and freq → (B*F, T, C)
        h_lstm = h.permute(0, 3, 2, 1).reshape(B * Fr, T, C)
        h_lstm, _ = self.lstm(h_lstm)
        h_lstm = self.lstm_proj(h_lstm)
        # Reshape back: (B, C, T, F)
        h = h_lstm.reshape(B, Fr, T, C).permute(0, 3, 2, 1)

        # Decoder with skip connections
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            # Pad h to match skip spatial dims if needed
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="nearest")
            h = dec(h, skip)

        # Pad output to match input magnitude shape
        if h.shape[2:] != magnitude.shape[2:]:
            h = F.interpolate(h, size=magnitude.shape[2:], mode="nearest")

        # Apply mask
        mask = self.mask_activation(self.mask_conv(h))
        estimated_mag = mask * magnitude

        # Reconstruct with original phase
        estimated_spec = estimated_mag * phase

        # iSTFT
        return self.istft(estimated_spec, orig_length)
