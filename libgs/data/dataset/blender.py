import json
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
from PIL import Image

from libgs.utils.sh import SH2RGB

from ..types import BasicPointCloud, CameraInfo, fetchPly, storePly
from ..utils.graphics import focal2fov, fov2focal
from .base import Dataset


class BlenderDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: Literal["train", "test", "none"] = "none",
        resolution: float = -1.0,
        resolution_scale: float = 1.0,
        white_background: bool = False,
        extension: str = ".png",
    ):
        super().__init__(root, split, resolution, resolution_scale)
        self.white_background = white_background
        self.extension = extension

    @property
    def ply_path(self) -> Path:
        return self.root / "points3d.ply"

    @cached_property
    def point_cloud(self) -> Optional[BasicPointCloud]:
        if not self.ply_path.exists():
            # Since this data set has no colmap data, we start with random points
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")

            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            storePly(self.ply_path, xyz, SH2RGB(shs) * 255)

        try:
            return fetchPly(self.ply_path)
        except:
            return None

    def setup(self):
        if self.split == "none":
            items = self.process_split("train") + self.process_split("test")
        else:
            items = self.process_split(self.split)
        self.items = items

    def process_split(self, split: Literal["train", "test"]) -> List[CameraInfo]:
        file = self.root / f"transforms_{split}.json"
        contents = json.loads(file.read_text(encoding="utf-8"))
        fovx = contents["camera_angle_x"]
        with ThreadPoolExecutor(max_workers=8) as executor:
            fn = lambda args: self.parse_item(args[1], fovx, args[0])
            items = list(executor.map(fn, enumerate(contents["frames"])))
        return items

    def parse_item(self, frame, fovx, uid=0) -> CameraInfo:
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        # R is stored transposed due to 'glm' in CUDA code
        R, T = np.transpose(w2c[:3, :3]), w2c[:3, 3]

        image_path = self.root / (frame["file_path"] + self.extension)
        image = self.open_image(image_path)

        return CameraInfo(
            uid,
            R=R,
            T=T,
            fovx=fovx,
            fovy=focal2fov(fov2focal(fovx, image.width), image.height),
            width=image.width,
            height=image.height,
            path=image_path,
            image=image,
        )

    def open_image(self, image_path) -> Image.Image:
        bg = np.array([1, 1, 1]) if self.white_background else np.array([0, 0, 0])

        image = Image.open(image_path)
        image_data = np.array(image.convert("RGBA")) / 255.0
        alpha = image_data[:, :, 3:4]
        image_data = (image_data[:, :, :3] * alpha + bg * (1 - alpha)) * 255
        image = Image.fromarray(np.array(image_data, dtype=np.byte), "RGB")
        return image
