from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    history: dict[str, list[float]],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema_model: nn.Module | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if ema_model is not None:
        checkpoint["ema_model"] = ema_model.state_dict()
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema_model: nn.Module | None = None,
    use_ema: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model_state = checkpoint.get("ema_model") if use_ema else checkpoint["model"]
    if model_state is None:
        model_state = checkpoint["model"]
    model.load_state_dict(model_state)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if ema_model is not None:
        ema_model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
    return checkpoint
