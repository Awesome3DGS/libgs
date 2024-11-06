from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Optional

from .dataset import Dataset, load_dataset
from .types import BasicPointCloud, getNerfppNorm


@dataclass
class SceneInfo:
    train_dataset: Dataset
    test_dataset: Optional[Dataset] = None

    @property
    def ply_path(self) -> Path:
        return self.train_dataset.ply_path

    @cached_property
    def nerf_normalization(self) -> Dict[str, Any]:
        return getNerfppNorm(self.train_dataset.items)

    @property
    def point_cloud(self) -> BasicPointCloud:
        return self.train_dataset.point_cloud


def load_scene(
    root: Path,
    resolution: int,
    split_train_test: bool,
    **kwargs: Dict[str, Any],
):
    if split_train_test:
        train_dataset = load_dataset(root, "train", resolution, **kwargs)
        test_dataset = load_dataset(root, "test", resolution, **kwargs)
    else:
        train_dataset = load_dataset(root, "none", resolution, **kwargs)
        test_dataset = None
    return SceneInfo(train_dataset=train_dataset, test_dataset=test_dataset)
