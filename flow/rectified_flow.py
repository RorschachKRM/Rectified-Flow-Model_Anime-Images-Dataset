from __future__ import annotations

import torch
from torch import nn


class RectifiedFlow:
    """V1 Rectified Flow：直线路径、速度匹配与前向欧拉求解。"""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def training_loss(self, real_images: torch.Tensor) -> torch.Tensor:
        """拟合直线路径 x_t=(1-t)x_0+t*x_1 的常速度 x_1-x_0。"""
        x1 = real_images
        x0 = torch.randn_like(x1)
        time = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        time_view = time.view(-1, 1, 1, 1)
        xt = (1.0 - time_view) * x0 + time_view * x1
        target_velocity = x1 - x0
        predicted_velocity = self.model(xt, time)
        return torch.mean((predicted_velocity - target_velocity) ** 2)

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        num_steps: int,
        trajectory_frames: int = 0,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """从 t=0 的高斯噪声出发，用前向欧拉法积分到 t=1。"""
        if num_steps <= 0:
            raise ValueError("num_steps 必须大于 0")
        if trajectory_frames < 0:
            raise ValueError("trajectory_frames 不能为负数")

        sample = noise.clone()
        step_size = 1.0 / num_steps
        record_steps = self._trajectory_steps(num_steps, trajectory_frames)
        trajectory = [sample.detach().cpu()] if 0 in record_steps else []

        for step in range(num_steps):
            time_value = step / num_steps
            time = torch.full(
                (sample.shape[0],), time_value, device=sample.device, dtype=sample.dtype
            )
            velocity = self.model(sample, time)
            sample = sample + step_size * velocity
            if step + 1 in record_steps:
                trajectory.append(sample.detach().cpu())
        return sample, trajectory

    @staticmethod
    def _trajectory_steps(num_steps: int, trajectory_frames: int) -> set[int]:
        if trajectory_frames == 0:
            return set()
        if trajectory_frames == 1:
            return {num_steps}
        return {
            round(index * num_steps / (trajectory_frames - 1))
            for index in range(trajectory_frames)
        }
