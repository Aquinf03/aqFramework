"""Lightweight discrete VAE (DALL·E-style) for BEiT visual tokens."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed: int = 8192, dim: int = 256, beta: float = 0.25):
        super().__init__()
        self.n_embed = n_embed
        self.dim = dim
        self.beta = beta
        self.embed = nn.Embedding(n_embed, dim)
        nn.init.uniform_(self.embed.weight, -1.0 / n_embed, 1.0 / n_embed)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).contiguous().view(-1, c)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1)
        )
        ids = dist.argmin(1)
        z_q = self.embed(ids).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        loss = F.mse_loss(z_q.detach(), z) + self.beta * F.mse_loss(z_q, z.detach())
        z_q = z + (z_q - z).detach()
        return z_q, loss, ids.view(b, h, w)


class DiscreteVAE(nn.Module):
    """Encode image → discrete tokens → decode. Trained then frozen as BEiT tokenizer."""

    def __init__(self, in_ch: int = 3, hidden: int = 256, n_embed: int = 8192, patch: int = 16):
        super().__init__()
        self.patch = patch
        assert patch == 16, "BEiT tokenizer expects patch=16 (4× stride-2)"
        # 224 → 14 tokens with 4 stride-2 downs
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden // 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 4, hidden // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, hidden, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
        )
        self.quant = VectorQuantizer(n_embed=n_embed, dim=hidden)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden, hidden // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden // 2, hidden // 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden // 4, in_ch, 4, stride=2, padding=1),
        )
        self.n_embed = n_embed

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z_q, _, ids = self.quant(z)
        return z_q, ids

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z_q, qloss, ids = self.quant(z)
        recon = self.decoder(z_q)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return recon, F.mse_loss(recon, x) + qloss, ids

    @torch.no_grad()
    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        _, ids = self.encode(x)
        return ids
