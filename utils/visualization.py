from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return images.detach().float().clamp(-1.0, 1.0).add(1.0).div(2.0)


def save_sample_grid(images: torch.Tensor, path: str | Path, nrow: int = 4) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(images).cpu(), path, nrow=nrow)


def log_generation_to_tensorboard(
    writer: SummaryWriter,
    final_images: torch.Tensor,
    trajectory: Sequence[torch.Tensor],
    global_step: int,
    trajectory_samples: int,
) -> None:
    final_grid = make_grid(denormalize(final_images).cpu(), nrow=4)
    writer.add_image("samples/final", final_grid, global_step)
    if not trajectory:
        return

    frame_batches = [denormalize(frame[:trajectory_samples]) for frame in trajectory]
    # 每一行是一张时间切片，横向是同一批随机种子的不同样本。
    trajectory_grid = make_grid(
        torch.cat(frame_batches, dim=0), nrow=trajectory_samples, padding=2
    )
    writer.add_image("samples/noise_to_image_trajectory", trajectory_grid, global_step)

    for frame_index, frame_batch in enumerate(frame_batches):
        frame_grid = make_grid(frame_batch, nrow=trajectory_samples, padding=2)
        writer.add_image(
            f"samples/trajectory_frames/frame_{frame_index:02d}", frame_grid, global_step
        )

    # TensorBoard 的 add_video 在 Windows 上会因 NamedTemporaryFile 文件锁失败。
    # 本地用轨迹网格和逐帧图片完整展示过程；Linux/Colab 额外记录动态视频。
    if os.name != "nt":
        video = torch.stack(frame_batches, dim=1)
        writer.add_video("samples/noise_to_image_video", video, global_step, fps=4)


def plot_loss_curves(history: dict[str, list[float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    train_losses = history.get("train", [])
    val_losses = history.get("val", [])
    if train_losses:
        axis.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    if val_losses:
        axis.plot(range(1, len(val_losses) + 1), val_losses, label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Flow matching MSE")
    axis.set_title("Rectified Flow Loss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
