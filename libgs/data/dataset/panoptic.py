from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np

from ..types import CameraInfo
from ..utils.graphics import focal2fov
from .colmap import Colmap4DDataset


class PanopticDataset(Colmap4DDataset):
    def __init__(
        self,
        root: Path,
        split: Literal["train", "test", "none"] = "none",
        resolution: float = -1.0,
        resolution_scale: float = 1.0,
        num_frames: int = 150,
        eval_pose_indices: List[int] = [0, 10, 15, 30],
    ):
        super().__init__(
            root, split, resolution, resolution_scale, num_frames, eval_pose_indices
        )

    def parse_pose(self, extr, intr) -> Tuple[np.array, np.array, int, int, int, int]:
        R, T, focal_length_x, focal_length_y = super().parse_pose(extr, intr)
        cx, cy = intr.params[-2], intr.params[-1]
        return R, T, focal_length_x, focal_length_y, cx, cy

    def parse_items(self, extr, intr) -> List[CameraInfo]:
        R, T, focal_length_x, focal_length_y, cx, cy = self.parse_pose(extr, intr)
        items, frame_paths = [], sorted((self.root / "frames").iterdir())
        for frame_index, frame_path in enumerate(frame_paths[: self.num_frames]):
            item = CameraInfo(
                uid=intr.id,
                R=R,
                T=T,
                fovx=focal2fov(focal_length_x, intr.width),
                fovy=focal2fov(focal_length_y, intr.height),
                width=intr.width,
                height=intr.height,
                cx=cx,
                cy=cy,
                path=frame_path / extr.name,
                frame=frame_index,
                znear=1.0,
                zfar=100.0,
            )
            items.append(item)
        return items
