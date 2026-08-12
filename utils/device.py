from __future__ import annotations

import random
from typing import Any

import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求使用 CUDA，但当前 PyTorch 无法访问 GPU")
    return device


def amp_enabled(config: dict[str, Any], device: torch.device) -> bool:
    return bool(config["training"]["mixed_precision"]) and device.type == "cuda"
