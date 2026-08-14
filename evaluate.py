from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm

from datasets import build_dataloaders, prepare_splits
from utils.config import ensure_directories, load_config
from utils.device import resolve_device, seed_everything
from utils.model import load_flow_for_inference


def _to_unit_interval(images: torch.Tensor) -> torch.Tensor:
    return images.detach().float().clamp(-1.0, 1.0).add(1.0).div(2.0)


@torch.no_grad()
def _flow_mse(
    flow: Any,
    loader: DataLoader[torch.Tensor],
    device: torch.device,
    seed: int,
) -> float:
    generator = torch.Generator(device=device).manual_seed(seed)
    total_loss = 0.0
    total_images = 0
    for images in tqdm(loader, desc="Test flow MSE"):
        images = images.to(device, non_blocking=True)
        loss = flow.training_loss(images, generator=generator)
        total_loss += loss.item() * images.shape[0]
        total_images += images.shape[0]
    return total_loss / max(total_images, 1)


@torch.no_grad()
def _distribution_metrics(
    flow: Any,
    loader: DataLoader[torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[float, float, float, int]:
    evaluation_config = config["evaluation"]
    total_images = len(loader.dataset)
    kid_subset_size = min(int(evaluation_config["kid_subset_size"]), total_images)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = KernelInceptionDistance(
        feature=2048,
        subsets=int(evaluation_config["kid_subsets"]),
        subset_size=kid_subset_size,
        normalize=True,
    ).to(device)

    for real_images in tqdm(loader, desc="Real features"):
        real_images = _to_unit_interval(real_images.to(device, non_blocking=True))
        fid.update(real_images, real=True)
        kid.update(real_images, real=True)

    generator = torch.Generator(device=device).manual_seed(
        int(evaluation_config["generation_seed"])
    )
    remaining = total_images
    progress = tqdm(total=total_images, desc="Generated features")
    while remaining > 0:
        batch_size = min(int(evaluation_config["batch_size"]), remaining)
        noise = torch.randn(
            batch_size,
            int(config["model"]["in_channels"]),
            int(config["data"]["image_size"]),
            int(config["data"]["image_size"]),
            generator=generator,
            device=device,
        )
        generated, _ = flow.sample(
            noise,
            num_steps=int(config["sampling"]["num_steps"]),
            solver=str(config["sampling"]["solver"]),
        )
        generated = _to_unit_interval(generated)
        fid.update(generated, real=False)
        kid.update(generated, real=False)
        remaining -= batch_size
        progress.update(batch_size)
    progress.close()

    fid_value = float(fid.compute().cpu())
    seed_everything(int(evaluation_config["metric_seed"]))
    kid_mean, kid_std = kid.compute()
    return fid_value, float(kid_mean.cpu()), float(kid_std.cpu()), total_images


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="评估 Rectified Flow 的 MSE、FID 和 KID")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="默认使用 best.pt")
    parser.add_argument(
        "--model-weights",
        action="store_true",
        help="使用普通模型权重；默认优先使用 EMA 权重",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    seed_everything(int(config["evaluation"]["metric_seed"]))
    if config["data"]["auto_prepare"]:
        prepare_splits(config)
    device = resolve_device(str(config["training"]["device"]))
    checkpoint = args.checkpoint or str(Path(config["paths"]["checkpoint_dir"]) / "best.pt")
    _, flow = load_flow_for_inference(
        config,
        checkpoint,
        device,
        use_ema=not args.model_weights,
    )

    original_test_loader = build_dataloaders(config)["test"]
    test_loader = DataLoader(
        original_test_loader.dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    results: dict[str, Any] = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "weights": "model" if args.model_weights else "ema",
        "solver": str(config["sampling"]["solver"]),
        "num_steps": int(config["sampling"]["num_steps"]),
        "validation_seed": int(config["evaluation"]["validation_seed"]),
        "generation_seed": int(config["evaluation"]["generation_seed"]),
        "flow_mse": _flow_mse(
            flow,
            test_loader,
            device,
            seed=int(config["evaluation"]["validation_seed"]),
        ),
    }
    if bool(config["evaluation"]["calculate_fid"]) or bool(
        config["evaluation"]["calculate_kid"]
    ):
        fid, kid_mean, kid_std, sample_count = _distribution_metrics(
            flow, test_loader, config, device
        )
        results["num_samples"] = sample_count
        if bool(config["evaluation"]["calculate_fid"]):
            results["fid"] = fid
        if bool(config["evaluation"]["calculate_kid"]):
            results["kid_mean"] = kid_mean
            results["kid_std"] = kid_std

    output_path = Path(config["paths"]["evaluation_dir"]) / "metrics.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"评估结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
