from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn


def calculate_l2_grad_norm(model: nn.Module) -> float:
    non_mp_norm_sq = torch.tensor(0.0, dtype=torch.float32, device="cuda")

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        non_mp_norm_sq += param.grad.norm(dtype=torch.float32) ** 2

    dist.all_reduce(non_mp_norm_sq)

    return non_mp_norm_sq.item() ** 0.5


def scale_grad(model: nn.Module, factor: float) -> None:
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(factor)
