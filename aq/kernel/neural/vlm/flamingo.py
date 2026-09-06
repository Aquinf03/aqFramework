"""Flamingo multimodal LM (Alayrac et al.) — perceiver + gated xattn every N LM layers."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .connectors import GatedCrossAttention, LLaVAProjector, PerceiverResampler
from .vision_tower import build_aq_vit_backbone, load_open_clip_visual, wants_open_clip


def _get_decoder_layers(lang_model: nn.Module) -> nn.ModuleList:
    """Resolve HF causal LM decoder block list."""
    if hasattr(lang_model, "model") and hasattr(lang_model.model, "layers"):
        return lang_model.model.layers  # Llama, Mistral, …
    if hasattr(lang_model, "transformer") and hasattr(lang_model.transformer, "h"):
        return lang_model.transformer.h  # GPT-2
    if hasattr(lang_model, "gpt_neox") and hasattr(lang_model.gpt_neox, "layers"):
        return lang_model.gpt_neox.layers
    raise SystemExit(
        "Flamingo needs a HF causal LM with accessible decoder layers "
        "(Llama/Mistral model.layers or GPT-2 transformer.h)."
    )


def _embed_tokens(lang_model: nn.Module):
    if hasattr(lang_model, "get_input_embeddings"):
        return lang_model.get_input_embeddings()
    raise SystemExit("language model missing get_input_embeddings()")


def _lm_head(lang_model: nn.Module):
    if hasattr(lang_model, "lm_head"):
        return lang_model.lm_head
    raise SystemExit("language model missing lm_head")


class FlamingoForCausalLM(nn.Module):
    """
    Vision → Perceiver resampler → gated cross-attn every `cross_every` decoder layers.

    GPT-4V-style generative trains use this or LLaVA; Flamingo is the cross-attn variant.
    """

    def __init__(
        self,
        lang_model: nn.Module,
        *,
        vision_arch: str = "vit-b/16",
        img_size: int = 224,
        num_latents: int = 64,
        resampler_depth: int = 6,
        cross_every: int = 1,
        freeze_vision: bool = True,
        freeze_lm: bool = True,
        vision_pretrained: str | bool | None = None,
    ):
        super().__init__()
        self.lang_model = lang_model
        hidden = int(lang_model.config.hidden_size)
        heads = int(getattr(lang_model.config, "num_attention_heads", 8))
        if wants_open_clip(vision_arch, vision_pretrained if isinstance(vision_pretrained, str) else None) or (
            isinstance(vision_pretrained, str) and vision_pretrained
        ):
            tag = vision_pretrained if isinstance(vision_pretrained, str) else None
            backbone, _ = load_open_clip_visual(vision_arch, pretrained=tag, img_size=img_size)
            vis_dim = backbone.dim
        else:
            backbone = build_aq_vit_backbone(vision_arch, img_size=img_size)
            vis_dim = backbone.dim
        self.vision = backbone
        # map vision dim → LM hidden before perceiver
        self.vision_proj = LLaVAProjector(vis_dim, hidden, mlp_depth=1)
        self.resampler = PerceiverResampler(
            dim=hidden, depth=resampler_depth, heads=max(1, heads // 2) or 8, num_latents=num_latents
        )
        layers = _get_decoder_layers(lang_model)
        self.n_layers = len(layers)
        self.cross_every = max(1, int(cross_every))
        n_xattn = (self.n_layers + self.cross_every - 1) // self.cross_every
        self.gated_xattn = nn.ModuleList(
            [GatedCrossAttention(hidden, heads=max(1, heads // 2) or 8) for _ in range(n_xattn)]
        )
        self.img_size = img_size
        self.vision_arch = vision_arch
        self.num_latents = num_latents
        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad_(False)
        if freeze_lm:
            for p in self.lang_model.parameters():
                p.requires_grad_(False)

    @property
    def config(self):
        return self.lang_model.config

    def encode_media(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.vision.forward_features(images)
        if feats.dim() == 3 and feats.shape[1] > 1:
            feats = feats[:, 1:, :]  # drop CLS when present
        feats = self.vision_proj(feats)
        return self.resampler(feats)  # B, latents, H

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        from transformers.modeling_outputs import CausalLMOutputWithPast

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds required")
            inputs_embeds = _embed_tokens(self.lang_model)(input_ids)

        media = self.encode_media(images) if images is not None else None

        # Run decoder manually with gated xattn interleaved
        hidden = inputs_embeds
        layers = _get_decoder_layers(self.lang_model)
        # HF GPT-2 SDPA wants bool/float mask; Llama accepts 2d padding mask.
        attn = attention_mask
        if attn is not None and attn.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            # additive mask: 0 keep, large negative block — broadcast to (B,1,1,L)
            attn = (1.0 - attn.to(dtype=hidden.dtype)) * torch.finfo(hidden.dtype).min
            attn = attn[:, None, None, :]
        xattn_i = 0
        for i, layer in enumerate(layers):
            if media is not None and (i % self.cross_every == 0):
                hidden = self.gated_xattn[xattn_i](hidden, media)
                xattn_i += 1
            out = layer(hidden, attention_mask=attn)
            hidden = out[0] if isinstance(out, tuple) else out

        # final norm
        model = self.lang_model
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            hidden = model.model.norm(hidden)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            hidden = model.transformer.ln_f(hidden)

        logits = _lm_head(model)(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.lang_model, "gradient_checkpointing_enable"):
            self.lang_model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        if hasattr(self.lang_model, "enable_input_require_grads"):
            self.lang_model.enable_input_require_grads()

    def save_pretrained(self, path, **kwargs):
        return self.lang_model.save_pretrained(path, **kwargs)

    def get_input_embeddings(self):
        return _embed_tokens(self.lang_model)
