from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from datasets import build_dataloaders, prepare_splits
from utils.config import load_config
from utils.device import resolve_device, seed_everything
from utils.model import load_flow_for_inference


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="计算测试集 Rectified Flow MSE")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="默认使用 best.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config["project"]["seed"]))
    if config["data"]["auto_prepare"]:
        prepare_splits(config)
    device = resolve_device(str(config["training"]["device"]))
    checkpoint = args.checkpoint or str(Path(config["paths"]["checkpoint_dir"]) / "best.pt")
    _, flow = load_flow_for_inference(config, checkpoint, device)
    test_loader = build_dataloaders(config)["test"]

    total_loss = 0.0
    total_images = 0
    for images in tqdm(test_loader, desc="Test"):
        images = images.to(device, non_blocking=True)
        loss = flow.training_loss(images)
        total_loss += loss.item() * images.shape[0]
        total_images += images.shape[0]
    print(f"Test flow-matching MSE: {total_loss / total_images:.6f}")


if __name__ == "__main__":
    main()
