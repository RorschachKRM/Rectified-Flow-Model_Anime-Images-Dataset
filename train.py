from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import build_dataloaders, prepare_splits
from flow import RectifiedFlow
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import ensure_directories, load_config
from utils.device import amp_enabled, resolve_device, seed_everything
from utils.ema import ExponentialMovingAverage
from utils.model import build_model
from utils.visualization import (
    log_generation_to_tensorboard,
    plot_loss_curves,
    save_sample_grid,
)


def autocast_context(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else nullcontext()


def train_one_epoch(
    flow: RectifiedFlow,
    ema: ExponentialMovingAverage,
    loader: DataLoader[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    gradient_clip: float,
    writer: SummaryWriter,
    global_step: int,
    log_every_steps: int,
    gradient_accumulation_steps: int = 1,
) -> tuple[float, int]:
    flow.model.train()
    total_loss = 0.0
    total_images = 0
    progress = tqdm(loader, desc="Train", leave=False)

    optimizer.zero_grad(set_to_none=True)
    accumulation_loss = 0.0
    accumulation_size = gradient_accumulation_steps
    for batch_index, images in enumerate(progress):
        if batch_index % gradient_accumulation_steps == 0:
            accumulation_size = min(
                gradient_accumulation_steps, len(loader) - batch_index
            )
            accumulation_loss = 0.0
        images = images.to(device, non_blocking=True)
        with autocast_context(use_amp):
            loss = flow.training_loss(images)
            scaled_loss = loss / accumulation_size
        scaler.scale(scaled_loss).backward()
        accumulation_loss += loss.detach().item()

        batch_size = images.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_images += batch_size
        should_step = (
            (batch_index + 1) % gradient_accumulation_steps == 0
            or batch_index + 1 == len(loader)
        )
        if should_step:
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(flow.model.parameters(), gradient_clip)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before_step:
                ema.update(flow.model)
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % log_every_steps == 0:
                writer.add_scalar(
                    "loss/train_step",
                    accumulation_loss / accumulation_size,
                    global_step,
                )
                writer.add_scalar(
                    "training/learning_rate",
                    optimizer.param_groups[0]["lr"],
                    global_step,
                )
        progress.set_postfix(loss=f"{loss.detach().item():.4f}")

    return total_loss / max(total_images, 1), global_step


@torch.no_grad()
def evaluate_loss(
    flow: RectifiedFlow,
    loader: DataLoader[torch.Tensor],
    device: torch.device,
    use_amp: bool,
    description: str,
    generator: torch.Generator | None = None,
) -> float:
    flow.model.eval()
    total_loss = 0.0
    total_images = 0
    for images in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        with autocast_context(use_amp):
            loss = flow.training_loss(images, generator=generator)
        total_loss += loss.item() * images.shape[0]
        total_images += images.shape[0]
    return total_loss / max(total_images, 1)


@torch.no_grad()
def generate_preview(
    flow: RectifiedFlow,
    fixed_noise: torch.Tensor,
    config: dict[str, Any],
    epoch: int,
    writer: SummaryWriter,
    use_amp: bool,
) -> None:
    flow.model.eval()
    sample_config = config["sampling"]
    with autocast_context(use_amp):
        samples, trajectory = flow.sample(
            fixed_noise,
            num_steps=int(sample_config["num_steps"]),
            trajectory_frames=int(sample_config["trajectory_frames"]),
            solver=str(sample_config["solver"]),
        )
    sample_path = Path(config["paths"]["sample_dir"]) / f"epoch_{epoch:04d}.png"
    save_sample_grid(samples, sample_path)
    log_generation_to_tensorboard(
        writer,
        final_images=samples,
        trajectory=trajectory,
        global_step=epoch,
        trajectory_samples=min(int(sample_config["trajectory_samples"]), samples.shape[0]),
    )


def run_training(config: dict[str, Any]) -> None:
    ensure_directories(config)
    seed_everything(int(config["project"]["seed"]))
    if config["data"]["auto_prepare"]:
        metadata = prepare_splits(config)
        print(f"数据划分: {metadata['splits']}，忽略重复副本: {metadata['duplicate_files']}")

    device = resolve_device(str(config["training"]["device"]))
    use_amp = amp_enabled(config, device)
    print(f"训练设备: {device}，混合精度: {use_amp}")

    loaders = build_dataloaders(config)
    model = build_model(config, device)
    flow = RectifiedFlow(model)
    training_config = config["training"]
    gradient_accumulation_steps = int(
        training_config.get("gradient_accumulation_steps", 1)
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    effective_batch_size = (
        int(training_config["batch_size"]) * gradient_accumulation_steps
    )
    print(
        f"模型参数: {parameter_count / 1_000_000:.2f}M，"
        f"微批次: {training_config['batch_size']}，"
        f"梯度累积: {gradient_accumulation_steps}，"
        f"有效批次: {effective_batch_size}"
    )
    ema = ExponentialMovingAverage(model, decay=float(training_config["ema_decay"]))
    ema_flow = RectifiedFlow(ema.model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler_config = training_config["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training_config["epochs"]),
        eta_min=float(scheduler_config["min_learning_rate"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    writer = SummaryWriter(log_dir=config["paths"]["log_dir"])

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    latest_path = checkpoint_dir / "latest.pt"
    start_epoch = 1
    global_step = 0
    best_val_loss = math.inf
    history: dict[str, list[float]] = {"train": [], "val": []}
    if bool(training_config["resume"]) and latest_path.is_file():
        checkpoint = load_checkpoint(
            latest_path,
            model,
            device,
            optimizer,
            scaler,
            scheduler,
            ema_model=ema.model,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        history = checkpoint.get("history", history)
        if "scheduler" not in checkpoint:
            print("旧 checkpoint 不含调度器状态，余弦学习率将从当前学习率开始")
        print(f"从 {latest_path} 恢复，将从第 {start_epoch} 个 epoch 继续")

    noise_generator = torch.Generator(device=device).manual_seed(
        int(config["evaluation"]["preview_seed"])
    )
    fixed_noise = torch.randn(
        int(config["sampling"]["num_samples"]),
        int(config["model"]["in_channels"]),
        int(config["data"]["image_size"]),
        int(config["data"]["image_size"]),
        generator=noise_generator,
        device=device,
    )

    try:
        for epoch in range(start_epoch, int(training_config["epochs"]) + 1):
            train_loss, global_step = train_one_epoch(
                flow=flow,
                ema=ema,
                loader=loaders["train"],
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                gradient_clip=float(training_config["gradient_clip"]),
                writer=writer,
                global_step=global_step,
                log_every_steps=int(training_config["log_every_steps"]),
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            history["train"].append(train_loss)
            writer.add_scalar("loss/train_epoch", train_loss, epoch)

            should_validate = epoch % int(training_config["validate_every_epochs"]) == 0
            val_loss = math.nan
            if should_validate:
                validation_generator = torch.Generator(device=device).manual_seed(
                    int(config["evaluation"]["validation_seed"])
                )
                val_loss = evaluate_loss(
                    ema_flow,
                    loaders["val"],
                    device,
                    use_amp,
                    "Validation",
                    generator=validation_generator,
                )
                history["val"].append(val_loss)
                writer.add_scalar("loss/validation_epoch", val_loss, epoch)

            epoch_message = f"Epoch {epoch:04d} | train={train_loss:.6f}"
            if should_validate:
                epoch_message += f" | val={val_loss:.6f}"
            print(epoch_message)

            if epoch % int(training_config["sample_every_epochs"]) == 0:
                generate_preview(ema_flow, fixed_noise, config, epoch, writer, use_amp)

            scheduler.step()
            writer.add_scalar(
                "training/learning_rate_epoch", scheduler.get_last_lr()[0], epoch
            )

            if should_validate and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    global_step,
                    best_val_loss,
                    history,
                    scheduler,
                    ema.model,
                )
            save_checkpoint(
                latest_path,
                model,
                optimizer,
                scaler,
                epoch,
                global_step,
                best_val_loss,
                history,
                scheduler,
                ema.model,
            )
            if epoch % int(training_config["save_every_epochs"]) == 0:
                save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:04d}.pt",
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    global_step,
                    best_val_loss,
                    history,
                    scheduler,
                    ema.model,
                )

            plot_loss_curves(history, Path(config["paths"]["plot_dir"]) / "loss_curve.png")
            writer.flush()
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="训练无条件 Rectified Flow")
    parser.add_argument("--config", default="config/default.yaml", help="YAML 配置文件")
    args = parser.parse_args()
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()
