from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .bias import StructuredBiasBuilder
from .layers import BiasRouter, MSBRAttentionLayer, RoutingAwareMoE
from .relations import BIAS_NAMES, RelationResources


class MSBRKT(nn.Module):
    """MSBR-KT model.

    The model expects a batch dictionary produced by ``msbr_kt.data.collate_batch``
    or an equivalent tensor dictionary with these keys:
    ``item_seq``, ``skill_seq``, ``label_seq``, ``dt_seq``, ``prev_y_seq``,
    ``prev_ac_seq``, ``prev_posr_seq``, ``prev_pcr_seq``, ``prev_dt_seq``,
    ``mask``, and ``pad_mask``.
    """

    def __init__(self, cfg: dict[str, Any], relations: dict[str, torch.Tensor] | RelationResources):
        super().__init__()
        self.cfg = dict(cfg)
        self.skill_count = int(cfg["skill_num"])
        self.item_count = int(cfg["item_num"])
        d_model = int(cfg["d_model"])
        n_heads = int(cfg["n_heads"])
        n_layers = int(cfg["n_layers"])
        dropout = float(cfg["dropout"])

        if d_model % 8 != 0:
            raise ValueError("d_model must be divisible by 8 because the input embedding is partitioned as 1/2, 1/4, 1/8, 1/8.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.skill_count <= 0 or self.item_count <= 0:
            raise ValueError("skill_num and item_num must be positive. They are usually loaded from global_maps.pt.")

        relation_tensors = relations.data if isinstance(relations, RelationResources) else relations

        self.skill_emb = nn.Embedding(self.skill_count + 1, d_model // 2, padding_idx=0)
        self.item_id_emb = nn.Embedding(self.item_count + 1, d_model // 4, padding_idx=0)
        self.response_emb = nn.Embedding(3, d_model // 8, padding_idx=0)  # 0 pad/no-history, 1 wrong, 2 correct
        self.num_mlp = nn.Sequential(
            nn.Linear(4, d_model // 8),
            nn.ReLU(),
            nn.Linear(d_model // 8, d_model // 8),
        )
        input_dim = d_model // 2 + d_model // 4 + d_model // 8 + d_model // 8
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.bias_names = list(BIAS_NAMES)
        self.n_components = len(self.bias_names)
        self.bias_builder = StructuredBiasBuilder(relation_tensors, n_components=self.n_components)
        self.layers = nn.ModuleList([MSBRAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.routers = nn.ModuleList([BiasRouter(d_model, n_heads, self.n_components) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)

        pred_dim = d_model + d_model // 4 + d_model // 2
        self.predictor = RoutingAwareMoE(pred_dim, self.n_components, dropout)

    def _encode_input(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        item = batch["item_seq"].clamp(min=0, max=self.item_count)
        skill = batch["skill_seq"].clamp(min=0, max=self.skill_count)
        prev_y = batch["prev_y_seq"].long().clamp(min=0, max=1) + 1
        valid = batch["mask"].bool()
        first_valid = valid & (valid.long().cumsum(dim=1) == 1)
        prev_y = prev_y.masked_fill(batch["pad_mask"].bool() | first_valid, 0)

        numeric = torch.stack(
            [
                torch.log1p(batch["prev_ac_seq"].float().clamp_min(0.0)),
                batch["prev_posr_seq"].float().clamp(0.0, 1.0),
                batch["prev_pcr_seq"].float().clamp(0.0, 1.0),
                torch.log1p(batch["prev_dt_seq"].float().clamp_min(0.0)),
            ],
            dim=-1,
        )
        x = torch.cat(
            [
                self.skill_emb(skill),
                self.item_id_emb(item),
                self.response_emb(prev_y),
                self.num_mlp(numeric),
            ],
            dim=-1,
        )
        return self.dropout(self.input_norm(self.input_proj(x)))

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False):
        device = next(self.parameters()).device
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        item = batch["item_seq"].clamp(min=0, max=self.item_count)
        skill = batch["skill_seq"].clamp(min=0, max=self.skill_count)
        pad_mask = batch["pad_mask"].bool()

        h = self._encode_input(batch)
        bias_stack = self.bias_builder(batch)                  # [B, K, T, T]
        bias_rows = bias_stack.permute(0, 2, 1, 3).contiguous() # [B, T, K, T]

        entropy_sum = h.new_tensor(0.0)
        quad_sum = h.new_tensor(0.0)
        last_routing = None
        last_attention = None

        for layer, router in zip(self.layers, self.routers):
            routing = router(h)  # [B, H, T, K]
            entropy_sum = entropy_sum + (-(routing * routing.clamp_min(1e-9).log()).sum(dim=-1).mean())
            quad_sum = quad_sum + routing.pow(2).sum(dim=-1).mean()
            routed_bias = torch.einsum("bhtk,btkj->bhtj", routing, bias_rows)
            h, attention = layer(h, routed_bias, pad_mask)
            last_routing = routing
            last_attention = attention

        h = self.out_norm(h)
        pi = last_routing.mean(dim=1)
        pred_x = torch.cat([h, self.item_id_emb(item), self.skill_emb(skill)], dim=-1)
        logits, expert_logits = self.predictor(pred_x, pi)

        aux = {
            "router_entropy": entropy_sum / max(1, len(self.layers)),
            "router_quad": quad_sum / max(1, len(self.layers)),
            "pi": pi,
            "routing": last_routing,
            "attention": last_attention,
            "expert_logits": expert_logits,
            "bias_names": self.bias_names,
        }
        return (logits, aux) if return_aux else logits
