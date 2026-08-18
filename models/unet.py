from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .attention import AttentionBlock
from .blocks import Downsample, ResidualBlock, Upsample, group_norm
from .time_embedding import TimeEmbedding


class UNet(nn.Module):
    """用于预测速度场 v_theta(x_t, t) 的小型残差 U-Net。"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] | list[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        time_embedding_dim: int = 256,
        dropout: float = 0.0,
        image_size: int = 64,
        attention_resolutions: tuple[int, ...] | list[int] = (),
        attention_num_heads: int = 8,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        attention_resolutions = {int(value) for value in attention_resolutions}
        level_resolutions = {
            image_size // (2**level) for level in range(len(channel_multipliers))
        }
        unsupported_resolutions = attention_resolutions - level_resolutions
        if unsupported_resolutions:
            raise ValueError(
                "attention_resolutions 必须对应 U-Net 的分辨率层，"
                f"无效值: {sorted(unsupported_resolutions)}"
            )

        self.time_embedding = TimeEmbedding(base_channels, time_embedding_dim)
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        channels = base_channels
        resolution = image_size
        skip_channels = [channels]
        self.down_blocks = nn.ModuleList()
        for level, multiplier in enumerate(channel_multipliers):
            out_level_channels = base_channels * multiplier
            for _ in range(num_res_blocks):
                block = ResidualBlock(
                    channels, out_level_channels, time_embedding_dim, dropout
                )
                self.down_blocks.append(block)
                channels = out_level_channels
                skip_channels.append(channels)
                if resolution in attention_resolutions:
                    self.down_blocks.append(
                        AttentionBlock(
                            channels,
                            num_heads=attention_num_heads,
                            dropout=attention_dropout,
                        )
                    )
            if level != len(channel_multipliers) - 1:
                self.down_blocks.append(Downsample(channels))
                skip_channels.append(channels)
                resolution //= 2

        middle_blocks: list[nn.Module] = [
            ResidualBlock(channels, channels, time_embedding_dim, dropout)
        ]
        if resolution in attention_resolutions:
            middle_blocks.append(
                AttentionBlock(
                    channels,
                    num_heads=attention_num_heads,
                    dropout=attention_dropout,
                )
            )
        middle_blocks.append(
            ResidualBlock(channels, channels, time_embedding_dim, dropout)
        )
        self.middle_blocks = nn.ModuleList(middle_blocks)

        self.up_blocks = nn.ModuleList()
        for level, multiplier in reversed(list(enumerate(channel_multipliers))):
            out_level_channels = base_channels * multiplier
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResidualBlock(
                        channels + skip_channels.pop(),
                        out_level_channels,
                        time_embedding_dim,
                        dropout,
                    )
                )
                channels = out_level_channels
                if resolution in attention_resolutions:
                    self.up_blocks.append(
                        AttentionBlock(
                            channels,
                            num_heads=attention_num_heads,
                            dropout=attention_dropout,
                        )
                    )
            if level != 0:
                self.up_blocks.append(Upsample(channels))
                resolution *= 2
        if skip_channels:
            raise RuntimeError("U-Net 跳跃连接通道配置不匹配")

        self.output_norm = group_norm(channels)
        self.output_conv = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "UNet":
        model_config = dict(config["model"])
        model_config.setdefault("image_size", int(config["data"]["image_size"]))
        return cls(**model_config)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 0:
            time = time.expand(x.shape[0])
        if time.ndim != 1 or time.shape[0] != x.shape[0]:
            raise ValueError("time 必须是形状为 [batch_size] 的张量")

        time_embedding = self.time_embedding(time)
        hidden = self.input_conv(x)
        skips = [hidden]
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                hidden = block(hidden, time_embedding)
                skips.append(hidden)
            elif isinstance(block, Downsample):
                hidden = block(hidden)
                skips.append(hidden)
            else:
                hidden = block(hidden)

        for block in self.middle_blocks:
            if isinstance(block, ResidualBlock):
                hidden = block(hidden, time_embedding)
            else:
                hidden = block(hidden)

        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                skip = skips.pop()
                if hidden.shape[-2:] != skip.shape[-2:]:
                    hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
                hidden = block(torch.cat((hidden, skip), dim=1), time_embedding)
            else:
                hidden = block(hidden)
        if skips:
            raise RuntimeError("U-Net 前向传播后仍有未使用的跳跃连接")
        return self.output_conv(F.silu(self.output_norm(hidden)))
