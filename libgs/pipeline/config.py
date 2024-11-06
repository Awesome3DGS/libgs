from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional


@dataclass
class DataConfig:
    pass


@dataclass
class ModuleConfig:
    pass


@dataclass
class TrainerConfig:
    pass


@dataclass
class Config:
    output_dir: Path = Path("output")
    experiment_name: str = datetime.now().strftime("%Y-%m-%d")
    mode: Literal["train", "validate"] = "train"
    ckpt_path: Optional[Path] = None
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    module: ModuleConfig = field(default_factory=ModuleConfig)
