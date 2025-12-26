import math
from typing import Iterable, List

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def inject_lora(
    module: nn.Module,
    target_modules: Iterable[str],
    r: int,
    alpha: int,
    dropout: float,
) -> List[str]:
    targets = list(target_modules)
    replaced: List[str] = []

    def _inject(parent: nn.Module, prefix: str = "") -> None:
        for name, child in parent.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(t in full_name for t in targets):
                setattr(parent, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced.append(full_name)
            else:
                _inject(child, full_name)

    _inject(module)
    return replaced


def set_lora_trainable(module: nn.Module, train_all: bool = False) -> None:
    if train_all:
        return
    for param in module.parameters():
        param.requires_grad = False
    for child in module.modules():
        if isinstance(child, LoRALinear):
            for param in child.lora_A.parameters():
                param.requires_grad = True
            for param in child.lora_B.parameters():
                param.requires_grad = True
