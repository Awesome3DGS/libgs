import os
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from absl import logging
from torch import DeviceObjType, Tensor
from torch.utils.data import DataLoader

from libgs.utils.config import to_yaml
from libgs.utils.dist import is_global_zero

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    SummaryWriter = Any
    TENSORBOARD_FOUND = False


from .config import Config, DataConfig, ModuleConfig, TrainerConfig


class DataModule:
    def __init__(self, config: DataConfig):
        self.config = config

    def setup(self, *args, **kwargs):
        raise NotImplementedError

    def on_save_checkpoint(self, ckpt: Dict[str, Any]):
        pass

    def on_load_checkpoint(self, ckpt: Dict[str, Any]):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        raise NotImplementedError

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        raise NotImplementedError

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        raise NotImplementedError


class Module(nn.Module):
    datamodule: DataModule
    trainer: "Trainer"

    def __init__(self, config: ModuleConfig, datamodule: Optional[DataModule] = None):
        super().__init__()

        self.config = config
        self.datamodule = datamodule

    @property
    def output_dir(self) -> Path:
        return self.trainer.output_dir

    @property
    def global_step(self) -> int:
        return self.trainer.global_step

    @property
    def device(self) -> DeviceObjType:
        return self.trainer.device

    def log(self, name: str, value: Union[Tensor, float, int]):
        return self.trainer.log(name, value)

    def log_dict(self, scalars: Dict[str, Union[Tensor, float, int]]):
        return self.trainer.log_dict(scalars)

    def log_histogram(self, name: str, histogram: Tensor):
        return self.trainer.log_histogram(name, histogram)

    def log_images(self, name: str, images: Tensor):
        return self.trainer.log_images(name, images)

    def setup(self, *args, **kwargs):
        pass

    def on_save_checkpoint(self, ckpt: Dict[str, Any]):
        pass

    def on_load_checkpoint(self, ckpt: Dict[str, Any]):
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def pre_training_step(self, *args, **kwargs):
        pass

    def training_step(self, *args, **kwargs):
        raise NotImplementedError

    def post_training_step(self, *args, **kwargs):
        pass

    def pre_validation_step(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        raise NotImplementedError

    def post_validation_step(self, *args, **kwargs):
        pass

    def validation_end(self, *args, **kwargs):
        pass


class Trainer:
    output_dir: Path
    module: Module
    tb_writer: Optional[SummaryWriter] = None
    global_step: int = 0

    def __init__(self, config: TrainerConfig, output_dir: Path):
        self.config = config

        current_time = datetime.now().strftime("%Y%m%dT%H%M%S")
        unique_str = os.getenv("LIBGS_JOB_ID", str(uuid.uuid4()))[0:6]
        self.output_dir = output_dir / f"{current_time}_{unique_str}"
        logging.info(f"Output directory: {self.output_dir}")

        # An empty tensor holding the device
        self._dummy = torch.empty(0, device="cuda")

    @property
    def datamodule(self) -> DataModule:
        return self.module.datamodule

    @property
    def is_global_zero(self) -> bool:
        return is_global_zero()

    @property
    def ckpt_save_path(self) -> Optional[Path]:
        return self.output_dir / f"chkpnt{self.global_step}.pth"

    @property
    def device(self) -> DeviceObjType:
        return self._dummy.device

    def to(self, device: Union[DeviceObjType, str]):
        self._dummy.to(device)
        if self.module:
            self.module.to(self.device)

    def log(self, name: str, value: Union[Tensor, float, int]):
        if self.tb_writer:
            self.tb_writer.add_scalar(name, value, self.global_step)

    def log_dict(self, scalars: Dict[str, Union[Tensor, float, int]]):
        if self.tb_writer:
            for name, value in scalars.items():
                self.tb_writer.add_scalar(name, value, self.global_step)

    def log_histogram(self, name: str, histogram: Tensor):
        if self.tb_writer:
            self.tb_writer.add_histogram(name, histogram, self.global_step)

    def log_images(self, name: str, images: Tensor):
        if self.tb_writer:
            self.tb_writer.add_images(name, images, global_step=self.global_step)

    def fit(self, module: Module, ckpt_path: Optional[Path] = None):
        self.setup(module, ckpt_path)
        return self.training_loop()

    def validate(self, module: Module, ckpt_path: Optional[Path] = None):
        self.setup(module, ckpt_path)
        return self.validation_loop()

    def init_logger(self) -> Optional[SummaryWriter]:
        if TENSORBOARD_FOUND:
            self.tb_writer = SummaryWriter(self.output_dir)
        else:
            logging.warn("Tensorboard not available: not logging progress")

    def setup(self, module: Module, ckpt_path: Optional[Path] = None):
        self.module, module.trainer = module, self
        self.init_logger()
        self.datamodule.setup(self.output_dir)
        self.module.to(self.device).setup()
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            self.global_step = ckpt["global_step"]
            self.datamodule.on_load_checkpoint(ckpt)
            self.module.on_load_checkpoint(ckpt)

    def training_loop(self):
        raise NotImplementedError

    @torch.no_grad()
    def validation_loop(self):
        raise NotImplementedError

    @torch.no_grad()
    def save_checkpoint(self):
        save_path = self.ckpt_save_path
        if save_path is None:
            logging.warning(f"Skip checkpoint saving due to save_path is None")
            return
        logging.info(f"\nSaving checkpoint to {save_path}\n")
        ckpt = dict(global_step=self.global_step)
        self.module.on_save_checkpoint(ckpt)
        self.datamodule.on_save_checkpoint(ckpt)
        torch.save(ckpt, save_path)


class Pipeline:
    datamodule: DataModule
    module: Module
    trainer: Trainer

    def __init__(self, config: Config, **kwargs):
        self.config = config
        self.kwargs = kwargs

        self.setup()

    @property
    def output_dir(self) -> Path:
        return self.trainer.output_dir

    @property
    def is_global_zero(self) -> bool:
        return self.trainer.is_global_zero

    def setup(self):
        if self.config.mode == "validate" and self.config.ckpt_path is None:
            raise ValueError("You need to provide a @ckpt_path for validation!")
        self.setup_trainer_and_modules()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_config()

    def setup_trainer_and_modules(self):
        raise NotImplementedError

    def save_config(self) -> None:
        if self.is_global_zero:
            config, path = asdict(self.config), self.output_dir / "config.yaml"
            with path.open("w") as f:
                to_yaml(config, stream=f)
            logging.info(f"Config saved at {str(path)}")

    def run(self, **kwargs) -> Any:
        _ = torch.cuda.empty_cache() if torch.cuda.is_available() else None
        kwargs.setdefault("ckpt_path", self.config.ckpt_path)
        if self.config.mode == "train":
            return self.trainer.fit(self.module, **kwargs)
        return self.trainer.validate(self.module, **kwargs)
