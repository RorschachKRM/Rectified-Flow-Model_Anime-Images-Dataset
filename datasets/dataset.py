from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.config import PROJECT_ROOT


class AnimeImageDataset(Dataset[torch.Tensor]):
    """从清单读取图片，并归一化到 Rectified Flow 使用的 [-1, 1]。"""

    def __init__(self, manifest_path: str | Path, image_size: int, train: bool) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"找不到数据清单: {self.manifest_path}")

        lines = self.manifest_path.read_text(encoding="utf-8").splitlines()
        self.image_paths = [
            path if path.is_absolute() else PROJECT_ROOT / path
            for line in lines
            if line.strip()
            for path in [Path(line.strip())]
        ]
        if not self.image_paths:
            raise ValueError(f"数据清单为空: {self.manifest_path}")

        operations: list[Any] = [
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
        ]
        if train:
            operations.append(transforms.RandomHorizontalFlip())
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.image_paths[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def build_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader[torch.Tensor]]:
    data_config = config["data"]
    training_config = config["training"]
    split_dir = Path(data_config["split_dir"])
    generator = torch.Generator().manual_seed(int(config["project"]["seed"]))
    workers = int(data_config["num_workers"])

    loaders: dict[str, DataLoader[torch.Tensor]] = {}
    for name in ("train", "val", "test"):
        dataset = AnimeImageDataset(
            manifest_path=split_dir / f"{name}.txt",
            image_size=int(data_config["image_size"]),
            train=name == "train" and bool(data_config["horizontal_flip"]),
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=int(training_config["batch_size"]),
            shuffle=name == "train",
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            worker_init_fn=_seed_worker,
            generator=generator,
            drop_last=name == "train",
        )
    return loaders
