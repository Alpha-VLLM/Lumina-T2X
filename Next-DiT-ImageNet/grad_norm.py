from typing import Dict

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import ColumnParallelLinear, ParallelEmbedding, RowParallelLinear
import torch
import torch.distributed as dist
import torch.nn as nn


def get_model_parallel_dim_dict(model: nn.Module) -> Dict[str, int]:
    ret_dict = {}
    for module_name, module in model.named_modules():
        if isinstance(module, ColumnParallelLinear):
            ret_dict[module_name + ".weight"] = 0
            if module.bias is not None:
                ret_dict[module_name + ".bias"] = 0
        elif isinstance(module, RowParallelLinear):
            ret_dict[module_name + ".weight"] = 1
            if module.bias is not None:
                ret_dict[module_name + ".bias"] = -1
        elif isinstance(module, ParallelEmbedding):
            ret_dict[module_name + ".weight"] = 1
        else:
            for param_name, param in module.named_parameters(recurse=False):
                ret_dict[(module_name + "." if len(module_name) > 0 else "") + param_name] = -1
    return ret_dict


def calculate_l2_grad_norm(
    model: nn.Module,
    model_parallel_dim_dict: Dict[str, int],
) -> float:
    mp_norm_sq = torch.tensor(0.0, dtype=torch.float32, device="cuda")
    non_mp_norm_sq = torch.tensor(0.0, dtype=torch.float32, device="cuda")

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        name = ".".join(x for x in name.split(".") if not x.startswith("_"))
        assert name in model_parallel_dim_dict
        if model_parallel_dim_dict[name] < 0:
            non_mp_norm_sq += param.grad.norm(dtype=torch.float32)
        else:
            mp_norm_sq += param.grad.norm(dtype=torch.float32)

    dist.all_reduce(mp_norm_sq)
    dist.all_reduce(non_mp_norm_sq)
    non_mp_norm_sq /= fs_init.get_model_parallel_world_size()

    return (mp_norm_sq.item() + non_mp_norm_sq.item()) ** 0.5


def scale_grad(model: nn.Module, factor: float) -> None:
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(factor)


def get_param_norm_dict(model: nn.Module, model_parallel_dim_dict: Dict[str, int]) -> Dict[str, float]:
    param_norm_dict = {}
    for name, param in model.named_parameters():
        name = ".".join(x for x in name.split(".") if not x.startswith("_"))
        norm_sq = param.norm(dtype=torch.float32) ** 2
        dist.all_reduce(norm_sq)
        norm_sq = norm_sq.item()
        if model_parallel_dim_dict[name] < 0:
            norm_sq /= fs_init.get_model_parallel_world_size()
        norm = norm_sq**0.5
        param_norm_dict[name] = norm
    return param_norm_dict
