from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_SCOPES = {
    "attention_only": [
        "self_attn.W_Q",
        "self_attn.W_K",
        "self_attn.W_V",
        "self_attn.to_out.0",
    ],
    "attention_plus_proj": [
        "self_attn.W_Q",
        "self_attn.W_K",
        "self_attn.W_V",
        "self_attn.to_out.0",
        "act_proj",
        "in_proj",
        "out_proj",
    ],
}


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_linear)!r}")

        self.base_linear = base_linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.rank, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, self.rank))
        self.reset_parameters()

        for param in self.base_linear.parameters():
            param.requires_grad = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling


def _resolve_parent_module(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _get_submodule(root: nn.Module, path: str) -> nn.Module:
    parent, leaf = _resolve_parent_module(root, path)
    return parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)


def _set_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    parent, leaf = _resolve_parent_module(root, path)
    if leaf.isdigit():
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)


def apply_lora_to_mlla(
    backbone: nn.Module,
    last_k: int,
    target_scope: str,
    rank: int,
    alpha: float | None = None,
    dropout: float = 0.0,
) -> list[str]:
    if target_scope not in TARGET_SCOPES:
        raise ValueError(f"Unknown target_scope={target_scope!r}, choices={sorted(TARGET_SCOPES)}")
    if not hasattr(backbone, "MLLA") or not hasattr(backbone.MLLA, "encoder"):
        raise AttributeError("Backbone does not expose MLLA.encoder")
    blocks = backbone.MLLA.encoder.blocks
    if last_k <= 0 or last_k > len(blocks):
        raise ValueError(f"last_k must be in [1, {len(blocks)}], got {last_k}")

    alpha = float(rank if alpha is None else alpha)
    start_idx = len(blocks) - last_k
    replaced = []
    for block_idx in range(start_idx, len(blocks)):
        block = blocks[block_idx]
        for path in TARGET_SCOPES[target_scope]:
            module = _get_submodule(block, path)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Expected nn.Linear at blocks[{block_idx}].{path}, got {type(module)!r}")
            _set_submodule(
                block,
                path,
                LoRALinear(base_linear=module, rank=rank, alpha=alpha, dropout=dropout),
            )
            replaced.append(f"MLLA.encoder.blocks.{block_idx}.{path}")
    return replaced


def freeze_backbone_except_lora(backbone: nn.Module) -> None:
    for param in backbone.parameters():
        param.requires_grad = False
    for module in backbone.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True


def iter_lora_named_parameters(module: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    for name, param in module.named_parameters():
        if ".lora_A" in name or ".lora_B" in name:
            yield name, param

