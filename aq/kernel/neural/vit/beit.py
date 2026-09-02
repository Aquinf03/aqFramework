"""BEiT (Bao et al.) — blockwise masked image modeling over discrete visual tokens."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import Block, PatchEmbed, RelativePositionBias, init_vit_weights, trunc_normal_


def blockwise_mask(
    batch: int,
    grid: int,
    *,
    mask_ratio: float = 0.4,
    min_num_patches: int = 16,
    max_num_patches: int = 75,
    device: torch.device | None = None,
) -> torch.Tensor:
    """BEiT-style block masking. Returns bool mask True = masked, shape (B, grid*grid)."""
    masks = torch.zeros(batch, grid, grid, dtype=torch.bool, device=device)
    n_masked = max(1, int(mask_ratio * grid * grid))
    max_num_patches = min(max_num_patches, grid * grid)
    min_num_patches = min(min_num_patches, max_num_patches)
    for b in range(batch):
        count = 0
        tries = 0
        while count < n_masked and tries < 120:
            tries += 1
            target = int(torch.randint(min_num_patches, max_num_patches + 1, (1,)).item())
            target = min(target, n_masked - count)
            aspect = math.exp(float(torch.empty(1).uniform_(math.log(0.3), math.log(1 / 0.3)).item()))
            h = int(round(math.sqrt(target * aspect)))
            w = int(round(math.sqrt(target / aspect)))
            h = max(1, min(h, grid))
            w = max(1, min(w, grid))
            top = int(torch.randint(0, grid - h + 1, (1,)).item()) if grid > h else 0
            left = int(torch.randint(0, grid - w + 1, (1,)).item()) if grid > w else 0
            newly = int((~masks[b, top : top + h, left : left + w]).sum().item())
            masks[b, top : top + h, left : left + w] = True
            count += newly
    return masks.view(batch, grid * grid)


class BEiT(nn.Module):
    """BEiT encoder with MIM head over a visual codebook vocabulary."""

    def __init__(
        self,
        img_size: int = 224,
        patch: int = 16,
        in_ch: int = 3,
        num_classes: int = 1000,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        vocab_size: int = 8192,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.patch = patch
        self.grid = img_size // patch
        self.vocab_size = vocab_size
        self.patch_embed = PatchEmbed(img_size, patch, in_ch, dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.pos_drop = nn.Dropout(drop)
        # shared relative bias over patch grid (no CLS in bias — applied to patches only via pad)
        self.rel_bias = RelativePositionBias(self.grid, heads)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(dim, heads, mlp_ratio, drop=drop, attn_drop=attn_drop, drop_path=dpr[i])
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.mim_head = nn.Linear(dim, vocab_size)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)
        self.apply(init_vit_weights)

    def _rel_bias_with_cls(self) -> torch.Tensor:
        # pad a zero row/col for CLS so bias is (1, heads, N+1, N+1)
        bias = self.rel_bias()  # 1, heads, G*G, G*G
        b, h, n, _ = bias.shape
        out = bias.new_zeros(b, h, n + 1, n + 1)
        out[:, :, 1:, 1:] = bias
        return out

    def forward_features(self, x: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)
        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(b, x.size(1), -1)
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            x = x * (1 - w) + mask_tokens * w
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        rel = self._rel_bias_with_cls()
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel)
        return self.norm(x)

    def forward_mim(
        self, x: torch.Tensor, bool_masked_pos: torch.Tensor, visual_tokens: torch.Tensor
    ) -> torch.Tensor:
        """visual_tokens: B, grid, grid discrete ids from VQ. Returns CE loss on masked positions."""
        feats = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        logits = self.mim_head(feats[:, 1:, :])  # B, N, V
        target = visual_tokens.view(x.shape[0], -1)
        loss = F.cross_entropy(logits[bool_masked_pos], target[bool_masked_pos])
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x)[:, 0])
