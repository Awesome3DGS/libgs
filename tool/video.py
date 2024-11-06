import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
from PIL import Image


def extract_frames(
    video_path: Path,
    frames_root: Path,
    image_size: Optional[Tuple[int, int]] = None,
    num_frames: Optional[int] = None,
):
    cam_id = int(re.sub(r"^\D+", "", video_path.stem))
    video_stream = cv2.VideoCapture(str(video_path))

    count, max_len = 0, max(len(f"{num_frames}"), 4)
    while video_stream.isOpened():
        ret, image = video_stream.read()
        if not ret:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if image_size is not None:
            image = image.resize(image_size, Image.LANCZOS)
        save_path = frames_root / f"{count:0{max_len}d}" / f"cam{cam_id:02}.png"
        print("==> image:", count, image.size, save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        image.save(save_path)
        count += 1
        if num_frames is not None and count >= num_frames:
            break


if __name__ == "__main__":
    root = Path(sys.argv[1])
    video_root, frames_root = root / "videos", root / "frames"
    for video_path in video_root.iterdir():
        print("Extract frames from video:", video_path)
        extract_frames(video_path, frames_root, num_frames=300)
