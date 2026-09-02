"""LLaVA multimodal causal LM wrapper (Liu et al.) — aq vision + projector + HF LM."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from neural.vit.factory import build_vit
from neural.vit.vit import VisionTransformer

from .connectors import LLaVAProjector


class LLaVAForCausalLM(nn.Module):
    """
    Insert projected patch tokens at IMAGE token positions.

    Forward accepts either:
      - input_ids + images + image_token_mask, or
      - inputs_embeds already built by the collator.
    """

    def __init__(
        self,
        lang_model: nn.Module,
        *,
        vision_arch: str = "vit-b/16",
        img_size: int = 224,
        projector_depth: int = 2,
        freeze_vision: bool = True,
        freeze_lm: bool = False,
    ):
        super().__init__()
        self.lang_model = lang_model
        hidden = int(lang_model.config.hidden_size)
        backbone = build_vit(vision_arch, num_classes=0, img_size=img_size)
        if not isinstance(backbone, VisionTransformer):
            raise SystemExit(f"LLaVA vision needs ViT arch, got {vision_arch!r}")
        self.vision = backbone
        self.projector = LLaVAProjector(backbone.dim, hidden, mlp_depth=projector_depth)
        self.img_size = img_size
        self.vision_arch = vision_arch
        self.num_vision_tokens = backbone.patch_embed.num_patches  # no CLS in LLaVA-1.5 patch stream
        # LLaVA uses patch tokens (optionally + CLS). We use all patch tokens after CLS skip.
        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad_(False)
        if freeze_lm:
            for p in self.lang_model.parameters():
                p.requires_grad_(False)

    @property
    def config(self):
        return self.lang_model.config

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        # B, N_patches, C_v  (drop CLS)
        feats = self.vision.forward_features(images)[:, 1:, :]
        return self.projector(feats)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor | None,
        image_token_id: int,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Expand each image_token_id into num_vision_tokens projected patches."""
        embed = self.lang_model.get_input_embeddings()
        inputs_embeds = embed(input_ids)
        if images is None:
            return inputs_embeds, attention_mask

        vision = self.encode_images(images)  # B, Nv, H
        bsz, Nv, H = vision.shape
        new_embeds = []
        new_mask = []
        for i in range(bsz):
            ids = input_ids[i]
            emb = inputs_embeds[i]
            mask = attention_mask[i] if attention_mask is not None else torch.ones_like(ids)
            # find image token positions
            pos = (ids == image_token_id).nonzero(as_tuple=False).flatten()
            if pos.numel() == 0:
                new_embeds.append(emb)
                new_mask.append(mask)
                continue
            # support one image token per sample (LLaVA default)
            p = int(pos[0].item())
            pieces = [emb[:p], vision[i], emb[p + 1 :]]
            masks = [
                mask[:p],
                torch.ones(Nv, device=mask.device, dtype=mask.dtype),
                mask[p + 1 :],
            ]
            new_embeds.append(torch.cat(pieces, dim=0))
            new_mask.append(torch.cat(masks, dim=0))
        # pad to max length in batch
        max_len = max(e.shape[0] for e in new_embeds)
        out_e = inputs_embeds.new_zeros(bsz, max_len, H)
        out_m = input_ids.new_zeros(bsz, max_len)
        for i, (e, m) in enumerate(zip(new_embeds, new_mask)):
            out_e[i, : e.shape[0]] = e
            out_m[i, : m.shape[0]] = m
        return out_e, out_m

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        image_token_id: int | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds required")
            if images is not None and image_token_id is None:
                raise ValueError("image_token_id required when images are passed")
            inputs_embeds, attention_mask = self.prepare_inputs_embeds(
                input_ids, images, int(image_token_id or -1), attention_mask
            )
            # expand labels the same way (image token → Nv times -100)
            if labels is not None and images is not None and image_token_id is not None:
                labels = self._expand_labels(labels, input_ids, int(image_token_id), images.shape[0])
            input_ids = None

        return self.lang_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def _expand_labels(
        self, labels: torch.Tensor, input_ids: torch.Tensor, image_token_id: int, bsz: int
    ) -> torch.Tensor:
        Nv = self.num_vision_tokens
        rows = []
        max_len = 0
        built = []
        for i in range(bsz):
            ids = input_ids[i]
            lab = labels[i]
            pos = (ids == image_token_id).nonzero(as_tuple=False).flatten()
            if pos.numel() == 0:
                built.append(lab)
            else:
                p = int(pos[0].item())
                built.append(
                    torch.cat(
                        [
                            lab[:p],
                            lab.new_full((Nv,), -100),
                            lab[p + 1 :],
                        ]
                    )
                )
            max_len = max(max_len, built[-1].shape[0])
        out = labels.new_full((bsz, max_len), -100)
        for i, row in enumerate(built):
            out[i, : row.shape[0]] = row
        return out

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.lang_model, "gradient_checkpointing_enable"):
            self.lang_model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        if hasattr(self.lang_model, "enable_input_require_grads"):
            self.lang_model.enable_input_require_grads()

    def save_pretrained(self, path, **kwargs):
        # lang weights via HF; vision+projector saved separately by backend
        return self.lang_model.save_pretrained(path, **kwargs)
