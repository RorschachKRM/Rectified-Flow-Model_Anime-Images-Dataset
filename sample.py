from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.config import ensure_directories, load_config
from utils.device import resolve_device, seed_everything
from utils.model import load_flow_for_inference
from utils.visualization import log_generation_to_tensorboard, save_sample_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="从纯噪声生成动漫头像")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="默认使用 best.pt")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--solver", choices=("euler", "heun"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    seed = int(config["project"]["seed"] if args.seed is None else args.seed)
    seed_everything(seed)
    device = resolve_device(str(config["training"]["device"]))
    checkpoint = args.checkpoint or str(Path(config["paths"]["checkpoint_dir"]) / "best.pt")
    _, flow = load_flow_for_inference(config, checkpoint, device)

    sample_config = config["sampling"]
    num_samples = args.num_samples or int(sample_config["num_samples"])
    num_steps = args.num_steps or int(sample_config["num_steps"])
    solver = args.solver or str(sample_config["solver"])
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        num_samples,
        int(config["model"]["in_channels"]),
        int(config["data"]["image_size"]),
        int(config["data"]["image_size"]),
        generator=generator,
        device=device,
    )
    final_images, trajectory = flow.sample(
        noise,
        num_steps=num_steps,
        trajectory_frames=int(sample_config["trajectory_frames"]),
        solver=solver,
    )

    output = Path(args.output or Path(config["paths"]["sample_dir"]) / f"sample_seed_{seed}.png")
    save_sample_grid(final_images, output)
    with SummaryWriter(log_dir=config["paths"]["log_dir"]) as writer:
        log_generation_to_tensorboard(
            writer,
            final_images,
            trajectory,
            global_step=seed,
            trajectory_samples=min(int(sample_config["trajectory_samples"]), num_samples),
        )
    print(f"生成图片已保存到: {output}（求解器: {solver}，步数: {num_steps}）")


if __name__ == "__main__":
    main()
