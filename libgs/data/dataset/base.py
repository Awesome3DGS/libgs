import random
from functools import lru_cache
from multiprocessing import Value
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import get_worker_info
from torchvision import transforms as T

from ..types import BasicPointCloud, CameraInfo, TensorSpace

WARNED = Value("i", 0)


@lru_cache(maxsize=1)
def warning():
    if get_worker_info().id == 0 and WARNED.value == 0:
        with WARNED.get_lock():
            WARNED.value += 1
        print(
            "[WARNING] Rescale large image to 1.6K. "
            "Set resolution=1 to use original size."
        )


def to_tensor(pil_image: Image.Image, size: Tuple[int, int]):
    transform = T.Compose([T.Resize(size[::-1]), T.ToTensor()])
    return transform(pil_image)


class Dataset(TorchDataset):
    items: List[CameraInfo]
    ply_path: Path
    point_cloud: Optional[BasicPointCloud]
    image_size: Optional[Tuple[int, int]]

    def __init__(
        self,
        root: Path,
        split: Literal["train", "test", "none"] = "none",
        resolution: float = -1.0,
        resolution_scale: float = 1.0,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.resolution = resolution
        self.resolution_scale = resolution_scale

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> TensorSpace:
        item = self.items[index]
        size = self.resolve_size(item)
        image = item.image if item.image else self.open_image(item.path)
        image = to_tensor(image, size)
        image = (image[:3] * image[3:4]) if image.shape[0] == 4 else image
        world_view_transform = item.get_world2view2()
        projection_matrix = item.get_projection()

        return TensorSpace(
            uid=item.uid,
            image=image,
            fovx=item.fovx,
            fovy=item.fovy,
            world_view_transform=world_view_transform,
            full_proj_transform=world_view_transform @ projection_matrix,
            camera_center=world_view_transform.inverse()[3, :3],
            path=item.path,
            frame=item.frame,
        )

    def setup(self):
        pass

    def open_image(self, image_path) -> Image.Image:
        return Image.open(image_path)

    def resolve_size(self, item: CameraInfo) -> Tuple[int, int]:
        size = getattr(self, "image_size", (item.width, item.height))

        if self.resolution in [1, 2, 4, 8]:  # downsampling times
            base = self.resolution_scale * self.resolution
            return round(size[0] / base), round(size[1] / base)

        base = self.resolution_scale
        if self.resolution == -1:
            if size[0] > 1600:
                base = self.resolution_scale * (size[0] / 1600)
                warning()
        else:
            base = self.resolution_scale * (size[0] / self.resolution)
        return int(size[0] / base), int(size[1] / base)

    def shuffle(self):
        random.shuffle(self.items)
