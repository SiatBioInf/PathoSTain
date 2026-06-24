#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = d_model * ff_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, ff_mult=ff_mult, dropout=dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        hidden = self.norm1(x)
        attn_out, attn_weights = self.attn(
            hidden,
            hidden,
            hidden,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        x = x + self.drop1(attn_out)
        x = x + self.ff(self.norm2(x))
        if return_attn:
            return x, attn_weights
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_ctx = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, ff_mult=ff_mult, dropout=dropout)

    def forward(self, q: torch.Tensor, context: torch.Tensor, return_attn: bool = False):
        q_norm = self.norm_q(q)
        ctx_norm = self.norm_ctx(context)
        attn_out, attn_weights = self.attn(
            q_norm,
            ctx_norm,
            ctx_norm,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        q = q + self.drop1(attn_out)
        q = q + self.ff(self.norm2(q))
        if return_attn:
            return q, attn_weights
        return q


class BiomarkerTokenDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_metrics: int,
        d_model: int = 128,
        num_context_tokens: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        self_attn_layers: int = 1,
        cross_attn_layers: int = 1,
        ff_mult: int = 4,
        spatial_dim: int = 3,
        use_feature_layernorm: bool = True,
        use_feature_l2_norm: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_metrics = num_metrics
        self.d_model = d_model
        self.num_context_tokens = num_context_tokens
        self.spatial_dim = int(spatial_dim)
        self.use_feature_layernorm = bool(use_feature_layernorm)
        self.use_feature_l2_norm = bool(use_feature_l2_norm)

        total_input_dim = in_dim + self.spatial_dim
        self.context_proj = nn.Linear(total_input_dim, num_context_tokens * d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.context_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        self.biomarker_tokens = nn.Parameter(torch.randn(num_metrics, d_model) * 0.02)
        self.self_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    ff_mult=ff_mult,
                )
                for _ in range(self_attn_layers)
            ]
        )
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    ff_mult=ff_mult,
                )
                for _ in range(cross_attn_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _normalize_feature(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_feature_layernorm:
            x = F.layer_norm(x, (x.shape[-1],))
        if self.use_feature_l2_norm:
            x = F.normalize(x, p=2, dim=-1)
        return x

    def build_context_tokens(self, x: torch.Tensor, spatial_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._normalize_feature(x)
        if self.spatial_dim > 0:
            if spatial_feats is None:
                spatial_feats = torch.zeros((x.shape[0], self.spatial_dim), device=x.device, dtype=x.dtype)
            x = torch.cat([x, spatial_feats], dim=-1)
        batch_size = x.shape[0]
        context = self.context_proj(x)
        context = context.view(batch_size, self.num_context_tokens, self.d_model)
        context = context + self.context_mlp(self.context_norm(context))
        return context

    def build_biomarker_tokens(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        tokens = self.biomarker_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        if device is not None:
            tokens = tokens.to(device)
        return tokens

    def forward(self, x: torch.Tensor, spatial_feats: Optional[torch.Tensor] = None, return_attn: bool = False):
        batch_size = x.shape[0]
        context = self.build_context_tokens(x, spatial_feats=spatial_feats)
        tokens = self.build_biomarker_tokens(batch_size, x.device)

        if return_attn:
            self_attn_maps = []
            cross_attn_maps = []
            for block in self.self_blocks:
                tokens, attn_weights = block(tokens, return_attn=True)
                self_attn_maps.append(attn_weights)
            for block in self.cross_blocks:
                tokens, attn_weights = block(tokens, context, return_attn=True)
                cross_attn_maps.append(attn_weights)
            y_hat = self.out_head(self.out_norm(tokens)).squeeze(-1)
            return y_hat, {
                "self_attn": self_attn_maps,
                "cross_attn": cross_attn_maps,
                "context": context,
            }

        for block in self.self_blocks:
            tokens = block(tokens)
        for block in self.cross_blocks:
            tokens = block(tokens, context)
        return self.out_head(self.out_norm(tokens)).squeeze(-1)
