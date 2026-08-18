from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path = "config/default.yaml") -> dict[str, Any]:
    """读取 YAML 配置，并把项目内路径解析为绝对路径。"""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"找不到配置文件: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件必须是 YAML 映射: {path}")

    config = deepcopy(config)
    config["_config_path"] = str(path)
    for section, keys in {
        "data": ("raw_dir", "split_dir"),
        "paths": (
            "output_dir",
            "checkpoint_dir",
            "sample_dir",
            "plot_dir",
            "evaluation_dir",
            "log_dir",
        ),
    }.items():
        for key in keys:
            value = Path(config[section][key])
            config[section][key] = str(value if value.is_absolute() else PROJECT_ROOT / value)

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    ratios = [config["data"][f"{name}_ratio"] for name in ("train", "val", "test")]
    if any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("train_ratio、val_ratio、test_ratio 必须大于 0 且总和为 1")
    if config["data"]["image_size"] % (2 ** (len(config["model"]["channel_multipliers"]) - 1)):
        raise ValueError("image_size 必须能被 U-Net 的总下采样倍数整除")
    if config["model"]["in_channels"] != config["model"]["out_channels"]:
        raise ValueError("Rectified Flow 的输入和输出通道数必须相同")
    if config["model"]["base_channels"] <= 0:
        raise ValueError("model.base_channels 必须大于 0")
    attention_resolutions = {
        int(value) for value in config["model"].get("attention_resolutions", [])
    }
    level_resolutions = {
        int(config["data"]["image_size"]) // (2**level)
        for level in range(len(config["model"]["channel_multipliers"]))
    }
    unsupported_resolutions = attention_resolutions - level_resolutions
    if unsupported_resolutions:
        raise ValueError(
            "model.attention_resolutions 包含 U-Net 中不存在的分辨率: "
            f"{sorted(unsupported_resolutions)}"
        )
    attention_num_heads = int(config["model"].get("attention_num_heads", 8))
    if attention_num_heads <= 0:
        raise ValueError("model.attention_num_heads 必须大于 0")
    for level, multiplier in enumerate(config["model"]["channel_multipliers"]):
        resolution = int(config["data"]["image_size"]) // (2**level)
        channels = int(config["model"]["base_channels"]) * int(multiplier)
        if resolution in attention_resolutions and channels % attention_num_heads:
            raise ValueError(
                f"分辨率 {resolution} 的通道数 {channels} "
                f"不能被 attention_num_heads={attention_num_heads} 整除"
            )
    if config["data"]["phash_size"] < 4:
        raise ValueError("data.phash_size 必须至少为 4")
    hash_bits = int(config["data"]["phash_size"]) ** 2
    if not 0 <= config["data"]["phash_threshold"] <= hash_bits:
        raise ValueError("data.phash_threshold 必须位于 0 和 pHash 位数之间")
    if any(not 0.0 < ratio < 1.0 for ratio in config["data"]["phash_crop_ratios"]):
        raise ValueError("data.phash_crop_ratios 中的值必须位于 0 和 1 之间")
    if config["sampling"]["num_steps"] <= 0:
        raise ValueError("sampling.num_steps 必须大于 0")
    if config["sampling"]["solver"] not in {"euler", "heun"}:
        raise ValueError("sampling.solver 必须是 euler 或 heun")
    if config["sampling"]["num_samples"] <= 0:
        raise ValueError("sampling.num_samples 必须大于 0")
    if not 1 <= config["sampling"]["trajectory_samples"] <= config["sampling"]["num_samples"]:
        raise ValueError("trajectory_samples 必须位于 1 和 num_samples 之间")
    if config["training"]["log_every_steps"] <= 0:
        raise ValueError("training.log_every_steps 必须大于 0")
    if int(config["training"].get("gradient_accumulation_steps", 1)) <= 0:
        raise ValueError("training.gradient_accumulation_steps 必须大于 0")
    scheduler = config["training"]["scheduler"]
    if scheduler["name"] != "cosine":
        raise ValueError("training.scheduler.name 目前只支持 cosine")
    if not 0 <= scheduler["min_learning_rate"] < config["training"]["learning_rate"]:
        raise ValueError("min_learning_rate 必须大于等于 0 且小于 learning_rate")
    if not 0.0 < config["training"]["ema_decay"] < 1.0:
        raise ValueError("training.ema_decay 必须位于 0 和 1 之间")
    if config["evaluation"]["batch_size"] <= 0:
        raise ValueError("evaluation.batch_size 必须大于 0")
    if config["evaluation"]["kid_subsets"] <= 0:
        raise ValueError("evaluation.kid_subsets 必须大于 0")
    if config["evaluation"]["kid_subset_size"] < 2:
        raise ValueError("evaluation.kid_subset_size 必须至少为 2")
    for key in ("validate_every_epochs", "sample_every_epochs", "save_every_epochs"):
        if config["training"][key] <= 0:
            raise ValueError(f"training.{key} 必须大于 0")


def ensure_directories(config: dict[str, Any]) -> None:
    for path in config["paths"].values():
        Path(path).mkdir(parents=True, exist_ok=True)
