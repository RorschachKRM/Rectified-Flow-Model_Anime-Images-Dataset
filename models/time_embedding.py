from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 4:
            raise ValueError("时间嵌入维度必须至少为 4")
        self.embedding_dim = embedding_dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        scale = math.log(10_000) / (half_dim - 1)
        frequencies = torch.exp(
            -scale * torch.arange(half_dim, device=time.device, dtype=time.dtype)
        )
        angles = time[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if self.embedding_dim % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class TimeEmbedding(nn.Module):
    def __init__(self, base_channels: int, time_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        return self.network(time)
