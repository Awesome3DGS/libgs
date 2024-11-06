import os
from pathlib import Path
from typing import Optional


def find_git_root(path: Path) -> Optional[Path]:
    if path.joinpath(".git").exists():
        return path
    if path.parent == path:
        return None
    return find_git_root(path.parent)


def get_git_commit_hash(path: Optional[Path] = None) -> Optional[str]:
    path = Path(os.getcwd()) if path is None else path

    try:
        git_root = find_git_root(path)
        if git_root is None:
            raise FileNotFoundError

        with open(git_root / ".git" / "HEAD", "r") as head_file:
            ref = head_file.readline().strip()

        if not ref.startswith("ref:"):
            return ref

        ref_path = git_root / ".git" / Path(*ref.split()[1].split("/"))
        with open(ref_path, "r") as ref_file:
            return ref_file.readline().strip()
    except FileNotFoundError:
        print("[Git] not a valid git repository.")
    except Exception as e:
        print(f"[Git] {e}")
