from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from jaxtyping import Array, Float
from PIL import Image
from torch import Tensor, from_numpy

from ..utils.graphics import fov2focal, getProjectionMatrix, getWorld2View2


@dataclass
class CameraInfo:
    uid: int  # Unique camera ID, typically one across the entire dataset
    R: Float[Array, "3 3"]
    T: Float[Array, "3"]
    fovx: float
    fovy: float
    width: int
    height: int
    path: Path
    znear: float = 0.01
    zfar: float = 100.0
    cx: Optional[int] = None
    cy: Optional[int] = None
    frame: Optional[int] = None
    image: Optional[Image.Image] = None

    @property
    def c2w(self):
        R, T, bottom = self.R, self.T, [0, 0, 0, 1]
        return np.vstack((np.hstack((R.T, T.reshape(-1, 1))), bottom))

    @property
    def fx(self):
        return fov2focal(self.fovx, self.width)

    @property
    def fy(self):
        return fov2focal(self.fovy, self.height)

    def get_world2view2(
        self,
        trans: Float[Array, "3"] = np.zeros(3),
        scale: float = 1.0,
    ) -> Float[Tensor, "4 4"]:
        return from_numpy(getWorld2View2(self.R, self.T, trans, scale)).T

    def get_projection(self) -> Float[Tensor, "4 4"]:
        nd_cx = None if self.cx is None else self.cx / self.width
        nd_cy = None if self.cy is None else self.cy / self.height
        args = (self.znear, self.zfar, self.fovx, self.fovy, nd_cx, nd_cy)
        return getProjectionMatrix(*args).T

    def to_json(self, inverse: bool = True, **kwargs):
        if self.frame is not None:
            kwargs.setdefault("frame", self.frame)

        w2c = np.linalg.inv(self.c2w)
        return {
            "img_name": self.path.stem,
            "width": self.width,
            "height": self.height,
            "position": w2c[:3, 3].tolist(),
            "rotation": w2c[:3, :3].tolist(),
            "fx": self.fx,
            "fy": self.fy,
            **kwargs,
        }


def getNerfppNorm(cameras: List[CameraInfo]):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    camera_centers = []

    for camera in cameras:
        w2c = getWorld2View2(camera.R, camera.T)
        c2w = np.linalg.inv(w2c)
        camera_centers.append(c2w[:3, 3:4])

    center, diagonal = get_center_and_diag(camera_centers)
    radius = diagonal * 1.1
    translate = -center

    return {"translate": translate, "radius": radius}
