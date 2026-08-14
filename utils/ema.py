from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class ExponentialMovingAverage:
    """维护一份不参与反向传播的模型指数移动平均权重。"""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay 必须位于 0 和 1 之间")
        self.decay = float(decay)
        self.model = deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_parameters = dict(model.named_parameters())
        for name, ema_parameter in self.model.named_parameters():
            ema_parameter.lerp_(model_parameters[name].detach(), 1.0 - self.decay)

        model_buffers = dict(model.named_buffers())
        for name, ema_buffer in self.model.named_buffers():
            ema_buffer.copy_(model_buffers[name].detach())

    def copy_from(self, model: nn.Module) -> None:
        self.model.load_state_dict(model.state_dict())

