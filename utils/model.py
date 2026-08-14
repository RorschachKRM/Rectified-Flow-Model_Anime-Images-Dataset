from __future__ import annotations

from typing import Any

import torch

from flow import RectifiedFlow
from models import UNet
from utils.checkpoint import load_checkpoint


def build_model(config: dict[str, Any], device: torch.device) -> UNet:
    return UNet.from_config(config).to(device)


def load_flow_for_inference(
    config: dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
    use_ema: bool = True,
) -> tuple[UNet, RectifiedFlow]:
    model = build_model(config, device)
    load_checkpoint(checkpoint_path, model, device, use_ema=use_ema)
    model.eval()
    return model, RectifiedFlow(model)
