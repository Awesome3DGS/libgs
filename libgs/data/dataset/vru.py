from pathlib import Path
from typing import List, Literal

from .colmap import Colmap4DDataset


class VRUDataset(Colmap4DDataset):
    def __init__(
        self,
        root: Path,
        split: Literal["train", "test", "none"] = "none",
        resolution: float = -1.0,
        resolution_scale: float = 1.0,
        num_frames: int = 250,
        eval_pose_indices: List[int] = [0, 10, 20, 30],
    ):
        super().__init__(
            root, split, resolution, resolution_scale, num_frames, eval_pose_indices
        )
