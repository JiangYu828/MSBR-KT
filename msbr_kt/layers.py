from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()


class MSBRAttentionLayer(nn.Module):
    """Causal multi-head attention with routed structured bias."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = int(n_heads)
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        h = self.norm1(x)
        batch_size, seq_len, d_model = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        if attn_bias is not None:
            logits = logits + attn_bias
        logits = logits.masked_fill(causal_mask(seq_len, x.device).unsqueeze(0).unsqueeze(0), float("-inf"))
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        x = residual + self.drop(self.out(y))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, attn


class BiasRouter(nn.Module):
    """Generate per-layer, per-head routing weights over bias components."""

    def __init__(self, d_model: int, n_heads: int, n_components: int):
        super().__init__()
        self.n_heads = int(n_heads)
        self.n_components = int(n_components)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_heads * n_components),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        routing = self.net(x).view(batch_size, seq_len, self.n_heads, self.n_components)
        routing = F.softmax(routing, dim=-1)
        return routing.permute(0, 2, 1, 3).contiguous()


class RoutingAwareMoE(nn.Module):
    """Prediction head whose experts are mixed by the final routing weights."""

    def __init__(self, input_dim: int, n_experts: int, dropout: float = 0.1):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(n_experts)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.drop(x)
        expert_logits = torch.cat([expert(x) for expert in self.experts], dim=-1)
        logits = (pi * expert_logits).sum(dim=-1)
        return logits, expert_logits
