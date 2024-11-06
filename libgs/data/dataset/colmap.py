from concurrent.futures import ThreadPoolExecutor
from copy import copy
from functools import cached_property
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch

from ..types import BasicPointCloud, CameraInfo, TensorSpace, fetchPly, storePly
from ..utils.colmap import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from ..utils.graphics import focal2fov
from .base import Dataset, to_tensor


class BaseColmapDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: Literal["train", "test", "none"] = "none",
        resolution: float = 1.0,
        resolution_scale: float = 1.0,
    ):
        super().__init__(root, split, resolution, resolution_scale)

    @property
    def sparse_path(self) -> Path:
        return self.root / "sparse/0"

    @property
    def ply_path(self) -> Path:
        return self.sparse_path / "points3D.ply"

    @cached_property
    def point_cloud(self) -> Optional[BasicPointCloud]:
        root = self.sparse_path
        if not self.ply_path.exists():
            print(
                "Converting point3d.bin to .ply, "
                "will happen only the first time you open the scene."
            )
            try:
                xyz, rgb, _ = read_points3D_binary(root / "points3D.bin")
            except:
                xyz, rgb, _ = read_points3D_text(root / "points3D.txt")
            storePly(self.ply_path, xyz, rgb)
        try:
            return fetchPly(self.ply_path)
        except:
            return None

    def get_test_split(self, extrinsics) -> List[str]:
        if (test_split_path := self.root / "test_split.txt").exists():
            return test_split_path.read_text().split("\n")
        return []

    def parse_pose(self, extr, intr) -> Tuple[np.array, np.array, int, int]:
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = focal_length_y = intr.params[0]
        elif intr.model in ["PINHOLE", "OPENCV"]:
            focal_length_x, focal_length_y = intr.params[0], intr.params[1]
        else:
            raise ValueError(
                "Colmap camera model not handled: only undistorted datasets "
                "(PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            )
        return R, T, focal_length_x, focal_length_y

    def parse_items(self, extr, intr) -> List[CameraInfo]:
        raise NotImplementedError

    def parse_item(self, extr, intr) -> CameraInfo:
        raise NotImplementedError

    def setup(self):
        root = self.sparse_path
        try:
            extrinsics = read_extrinsics_binary(root / "images.bin")
            intrinsics = read_intrinsics_binary(root / "cameras.bin")
        except:
            extrinsics = read_extrinsics_text(root / "images.txt")
            intrinsics = read_intrinsics_text(root / "cameras.txt")
        extrinsics = sorted(extrinsics.values(), key=lambda e: e.name)

        items, test_split = [], self.get_test_split(extrinsics)
        for idx, extr in enumerate(extrinsics):
            is_test = extr.name in test_split
            if self.split == "train" and is_test:
                continue
            if self.split == "test" and not is_test:
                continue
            intr = intrinsics[extr.camera_id]
            try:
                items.extend(self.parse_items(extr, intr))
            except NotImplementedError:
                items.append(self.parse_item(extr, intr))
        self.items = items


class ColmapDataset(BaseColmapDataset):
    def __init__(
        self,
        root: Path,
        split: Literal["train", "test", "none"] = "none",
        resolution: float = 1.0,
        resolution_scale: float = 1.0,
        split_step: int = 8,
        image_path: str = "images",
    ):
        super().__init__(root, split, resolution, resolution_scale)
        self.split_step = split_step
        self.image_path = image_path

    def get_test_split(self, extrinsics) -> List[str]:
        if test_split := super().get_test_split(extrinsics):
            return test_split
        return [e.name for i, e in enumerate(extrinsics) if i % self.split_step == 0]

    def parse_item(self, extr, intr) -> CameraInfo:
        R, T, focal_length_x, focal_length_y = self.parse_pose(extr, intr)
        return CameraInfo(
            uid=intr.id,
            R=R,
            T=T,
            fovx=focal2fov(focal_length_x, intr.width),
            fovy=focal2fov(focal_length_y, intr.height),
            width=intr.width,
            height=intr.height,
            path=self.root / self.image_path / extr.name,
        )


class Colmap4DDataset(BaseColmapDataset):
    def __init__(
        self,
        root: Path,
        split: Literal["train", "test", "none"] = "none",
        resolution: float = -1.0,
        resolution_scale: float = 1.0,
        num_frames: int = 300,
        eval_pose_indices: List[int] = [0],
        offsets: Optional[List[int]] = None,
        compress_offsets: bool = True,
        standalone_offsets: bool = False,
    ):
        super().__init__(root, split, resolution, resolution_scale)
        self.num_frames = num_frames
        self.eval_pose_indices = eval_pose_indices
        self.offsets = offsets
        self.compress_offsets = compress_offsets
        self.standalone_offsets = standalone_offsets

    def get_test_split(self, extrinsics) -> List[str]:
        if test_split := super().get_test_split(extrinsics):
            return test_split
        return [extrinsics[i].name for i in self.eval_pose_indices]

    def parse_items(self, extr, intr) -> List[CameraInfo]:
        R, T, focal_length_x, focal_length_y = self.parse_pose(extr, intr)
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
                path=frame_path / extr.name,
                frame=frame_index,
            )
            items.append(item)
        return items

    def get_frame_dataset(self, *frames: List[int]) -> BaseColmapDataset:
        dataset = copy(self)
        dataset.items = [item for item in self.items if item.frame in frames]
        return dataset

    def get_clip_dataset(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> BaseColmapDataset:
        assert start is not None and end is not None

        if start is None:
            filter_fn = lambda f: f < end
        elif end is None:
            filter_fn = lambda f: f >= start
        else:
            filter_fn = lambda f: f >= start and f < end
        dataset = copy(self)
        dataset.items = [item for item in self.items if filter_fn(item.frame)]
        return dataset

    def __getitem__(self, index: int) -> TensorSpace:
        item = super().__getitem__(index)
        if self.offsets is None:
            return item

        def fn(frame):
            n = len(item.path.parts[-2])
            path = item.path.parents[1] / f"{frame:0{n}d}" / item.path.name
            return to_tensor(self.open_image(path), item.image.shape[1:][::-1])

        frames = [f for o in self.offsets if (f := item.frame + o) < self.num_frames]
        if frames:
            with ThreadPoolExecutor(max_workers=8) as executor:
                offset_image = list(executor.map(fn, frames))
            offset_image = torch.stack(offset_image)
            if self.compress_offsets:
                offset_image = offset_image.mean(dim=0)
            if self.standalone_offsets:
                item.offset_image = offset_image
            else:
                item.image = offset_image
        return item
