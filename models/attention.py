from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import group_norm


class AttentionBlock(nn.Module):
    """在二维特征图上执行多头自注意力，并保留残差连接。"""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads 必须大于 0")
        if channels % num_heads != 0:
            raise ValueError("Attention 通道数必须能被 num_heads 整除")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Attention dropout 必须位于 [0, 1) 区间")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.dropout = float(dropout)
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.output_projection = nn.Conv2d(channels, channels, kernel_size=1)

        # 初始时 Attention 分支为零，新增模块不会立刻扰动卷积分支。
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        qkv = self.qkv(self.norm(x)).reshape(
            batch_size,
            3,
            self.num_heads,
            self.head_dim,
            height * width,
        )
        query, key, value = qkv.unbind(dim=1)
        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        value = value.transpose(-2, -1)
        hidden = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        hidden = hidden.transpose(-2, -1).reshape(
            batch_size, channels, height, width
        )
        return x + self.output_projection(hidden)
