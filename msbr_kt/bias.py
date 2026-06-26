from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .relations import BIAS_NAMES, build_time_bias


def directed_transition_bias(item_seq: torch.Tensor, topk_idx: torch.Tensor, topk_val: torch.Tensor) -> torch.Tensor:
    item_upper_bound = topk_idx.size(0) - 1
    src_items = item_seq.clamp(min=0, max=item_upper_bound)
    idx = topk_idx.index_select(0, src_items.reshape(-1)).view(*src_items.shape, -1)
    val = topk_val.index_select(0, src_items.reshape(-1)).view(*src_items.shape, -1)
    src_to_tgt = ((idx.unsqueeze(-1) == item_seq.unsqueeze(1).unsqueeze(2)).float() * val.unsqueeze(-1)).sum(dim=2)
    return src_to_tgt.transpose(1, 2).contiguous()


class StructuredBiasBuilder(nn.Module):

    def __init__(self, relation_tensors: dict[str, torch.Tensor], n_components: int = len(BIAS_NAMES)):
        super().__init__()
        self.bias_names = list(BIAS_NAMES)
        self.n_components = int(n_components)
        self.bias_scales = nn.Parameter(torch.full((self.n_components,), 0.1))

        for name in [
            "item_dir_idx",
            "item_dir_val",
            "item_right_idx",
            "item_right_val",
            "item_wrong_idx",
            "item_wrong_val",
        ]:
            tensor = relation_tensors[name]
            if not torch.is_tensor(tensor):
                tensor = torch.tensor(tensor)
            self.register_buffer(name, tensor)

    @staticmethod
    def rowwise_normalize(bias: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        _, _, seq_len, _ = bias.shape
        valid = valid_mask.bool()
        key_valid = valid.unsqueeze(1).unsqueeze(2)
        query_valid = valid.unsqueeze(1).unsqueeze(3)
        past = torch.tril(torch.ones(seq_len, seq_len, device=bias.device, dtype=torch.bool), diagonal=-1)
        visible = key_valid & query_valid & past.unsqueeze(0).unsqueeze(0)

        denom = visible.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (bias * visible.float()).sum(dim=-1, keepdim=True) / denom
        var = (((bias - mean) * visible.float()) ** 2).sum(dim=-1, keepdim=True) / denom
        std = torch.sqrt(var).clamp_min(1e-3)
        out = (bias - mean) / std
        out = out.masked_fill(~visible, 0.0)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        item = batch["item_seq"]
        skill = batch["skill_seq"]
        valid_mask = batch["mask"] > 0
        _, seq_len = item.shape
        eye = torch.eye(seq_len, device=item.device, dtype=torch.bool).unsqueeze(0)
        valid_pair = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)

        b_time = build_time_bias(batch["dt_seq"], valid_mask=valid_mask)
        b_item = (item.unsqueeze(2) == item.unsqueeze(1)).float().masked_fill(eye, 0.0)
        b_skill = (skill.unsqueeze(2) == skill.unsqueeze(1)).float().masked_fill(eye, 0.0)
        b_dir = directed_transition_bias(item, self.item_dir_idx, self.item_dir_val).masked_fill(eye, 0.0)
        b_right = directed_transition_bias(item, self.item_right_idx, self.item_right_val).masked_fill(eye, 0.0)
        b_wrong = directed_transition_bias(item, self.item_wrong_idx, self.item_wrong_val).masked_fill(eye, 0.0)

        components = [b_time, b_item, b_skill, b_dir, b_right, b_wrong]
        stack = torch.stack([comp.masked_fill(~valid_pair, 0.0) for comp in components], dim=1)
        stack = self.rowwise_normalize(stack, valid_mask)
        scales = F.softplus(self.bias_scales).view(1, self.n_components, 1, 1)
        return stack * scales
