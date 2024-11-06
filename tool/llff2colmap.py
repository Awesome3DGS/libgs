import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def load_poses(root_dir: Path):
    poses_arr = np.load(root_dir / "poses_bounds.npy")
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
    near_fars = poses_arr[:, -2:]
    H, W, focal = poses[0, :, -1]
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    return poses, H, W, focal, near_fars


def save_colmap_images(colmap_dir, poses, image_paths):
    object_images_file = open(colmap_dir / "images.txt", "w")
    for idx, pose in enumerate(poses):
        R = pose[:3, :3]
        R = -R
        R[:, 0] = -R[:, 0]
        T = pose[:3, 3]

        R = np.linalg.inv(R)
        T = -np.matmul(R, T)
        T = [str(i) for i in T]
        qevc = [str(i) for i in rotmat2qvec(R)]

        print(
            idx + 1,
            " ".join(qevc),
            " ".join(T),
            1,
            image_paths[idx].name,
            "\n",
            file=object_images_file,
        )
    object_images_file.close()


def save_colmap_cameras(colmap_dir, image_size, focal):
    # write camera infomation.
    object_cameras_file = open(colmap_dir / "cameras.txt", "w")
    print(
        1,
        "SIMPLE_PINHOLE",
        image_size[0],
        image_size[1],
        focal,
        image_size[0] / 2,
        image_size[1] / 2,
        file=object_cameras_file,
    )  #
    object_cameras_file.close()


def rebuild_colmap(root_dir: Path, images_dir: Path):
    colmap_dir = root_dir / "colmap"
    if colmap_dir.exists():
        shutil.rmtree(colmap_dir)
    os.makedirs(colmap_dir)

    shutil.copytree(images_dir, colmap_dir / "images")
    shutil.move(root_dir / "sparse_", colmap_dir / "sparse_custom")

    exit_code = os.system(f"bash {Path(__file__).parent}/rebuild_colmap.sh {root_dir}")
    if exit_code != 0:
        print("rebuild colmap failed")


def main(root_dir: Path):
    poses, H, W, focal, near_fars = load_poses(root_dir)
    print(poses)

    images_dir = root_dir / "images"
    if (root_dir / "frames").exists():
        images_dir = sorted((root_dir / "frames").iterdir())[0]
    image_paths = sorted(images_dir.iterdir())
    image_size = Image.open(image_paths[0]).size

    colmap_dir = root_dir / "sparse_"
    os.makedirs(colmap_dir, exist_ok=True)
    print("Save colmap images.")
    save_colmap_images(colmap_dir, poses, image_paths)
    print("Save colmap cameras.")
    save_colmap_cameras(colmap_dir, image_size, focal / (W / image_size[0]))
    print("Save colmap points.")
    open(colmap_dir / "points3D.txt", "w").close()
    print("Rebuild colmap.")
    rebuild_colmap(root_dir, images_dir)
    print("Done!")


if __name__ == "__main__":
    root_dir = Path(sys.argv[1])
    main(root_dir)
