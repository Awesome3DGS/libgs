from pathlib import Path
from typing import Tuple, Union

import cv2
from PIL import Image


def extract_images(
    video_path: Union[str, Path],
    images_root: Union[str, Path],
    image_size: Tuple[int, int],
    num_frames: int = 300,
):
    images_root = Path(images_root)
    images_root.mkdir(parents=True, exist_ok=True)
    video_frames = cv2.VideoCapture(video_path)

    count, max_len = 0, max(len(f"{num_frames}"), 4)
    while video_frames.isOpened() and count < num_frames:
        ret, video_frame = video_frames.read()
        if not ret:
            break
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        video_frame = Image.fromarray(video_frame)
        image = video_frame.resize(image_size, Image.LANCZOS)
        image.save(images_root / f"{count:0{max_len}d}.png")
        count += 1
