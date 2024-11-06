import inspect
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Optional, Union

from absl import logging

from ..types import BasicPointCloud, CameraInfo, TensorSpace, getNerfppNorm
from .base import Dataset
from .blender import BlenderDataset
from .colmap import Colmap4DDataset, ColmapDataset
from .panoptic import PanopticDataset
from .vru import VRUDataset

__all__ = ["KNOWN_DATASETS", "Dataset", "load_dataset"]

KNOWN_DATASETS = {
    "panoptic_sports": PanopticDataset,
    "vru": VRUDataset,
    "colmap4d": Colmap4DDataset,
    "colmap": ColmapDataset,
    "blender": BlenderDataset,
}


def load_dataset(
    root: Path,
    split: str,
    resolution: Union[int, float],
    **kwargs: Dict[str, Any],
) -> Dataset:
    # TODO custom loader.py
    if root.parent.name in KNOWN_DATASETS:
        dataset_name = root.parent.name
    elif (root / "sparse").exists():
        dataset_name = "colmap4d" if (root / "frames").exists() else "colmap"
    elif (root / "transforms_train.json").exists():
        dataset_name = "blender"
    else:
        raise ValueError("Could not recognize dataset type!")

    dataset_class = KNOWN_DATASETS[dataset_name]
    init_params = inspect.signature(dataset_class.__init__).parameters
    valid_keys = set(init_params.keys()) - {"self"}
    kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    if ignored_keys := [k for k in kwargs if k not in valid_keys]:
        logging.warning(f"Ignore dataset kwargs: {ignored_keys}")

    logging.info(f"Load {dataset_name} dataset from {root}")
    return KNOWN_DATASETS[dataset_name](root, split, resolution, **kwargs)
