from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
from absl import logging
from tqdm import tqdm

from libgs.data.utils.fetch import Fetcher
from libgs.pipeline import Trainer as BaseTrainer
from libgs.pipeline import TrainerConfig as BaseTrainerConfig


@dataclass
class TrainerConfig(BaseTrainerConfig):
    cache_data_on_device: bool = True
    testing_steps: List[int] = field(default_factory=lambda: [7000, 15000])
    saving_ckpt_steps: List[int] = field(default_factory=lambda: [7000, 15000])
    num_steps: int = 30000


class Trainer(BaseTrainer):
    @property
    def num_steps(self) -> int:
        return self.config.num_steps

    @property
    def is_first_step(self) -> bool:
        return self.global_step == 1  # range from 1

    @property
    def is_last_step(self) -> bool:
        return self.global_step == self.num_steps  # range from 1

    @property
    def ckpt_save_path(self) -> Path:
        return self.output_dir / f"ckpt-{self.global_step}.pth"

    def training_loop(self):
        train_dataloader = self.datamodule.train_dataloader()
        logging.info(f"Number of views in training dataset: {len(train_dataloader)}")
        cache = self.config.cache_data_on_device
        move_to_device = lambda ts: ts.to(self.device, True)
        fetcher = Fetcher(train_dataloader, cache, move_to_device)

        metrics_ema = {"loss": 0.0, "psnr": 0.0}

        def update_metric(name, value):
            metrics_ema[name] *= 0.6
            metrics_ema[name] += 0.4 * value

        progress_bar = tqdm(range(self.num_steps), "Training", initial=self.global_step)

        for self.global_step in range(self.global_step + 1, self.num_steps + 1):
            self.module.pre_training_step()

            args = (fetcher.next(), self.global_step)
            loss, metrics, render_pkg = self.module.training_step(*args)
            loss.backward()

            update_metric("loss", loss.item())
            update_metric("psnr", metrics["psnr"].item())
            if self.global_step % 10 == 0:
                metrics_pbar = {k: f"{v:.4f}" for k, v in metrics_ema.items()}
                metrics_pbar["gs"] = metrics["gs"]
                progress_bar.set_postfix(metrics_pbar)
                progress_bar.update(10)
            if self.global_step == self.num_steps:
                progress_bar.close()

            self.module.post_training_step(render_pkg)

            saving_ckpt_steps = self.config.saving_ckpt_steps + [self.num_steps]
            if self.global_step in saving_ckpt_steps:
                self.save_checkpoint()

            if self.global_step in self.config.testing_steps + [self.num_steps]:
                self.validation_loop()
        self.global_step = 0  # !!!Important

    @torch.no_grad()
    def validation_loop(self):
        torch.cuda.empty_cache()

        loaders = self.datamodule.val_dataloader()
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]

        results_list = []
        for loader_idx, loader in enumerate(loaders):
            name = self.datamodule.eval_names[loader_idx]
            logging.info(f"Number of views in {name} dataset: {len(loader)}")
            results = []
            for idx, viewpoint in enumerate(loader):
                viewpoint = viewpoint.to(self.device)
                metrics = self.module.validation_step(viewpoint, idx, loader_idx)
                results.append(metrics)
            results_list.append(results)

        num_loaders = len(loaders)
        if num_loaders == 1:
            results_list = results_list[0]
        self.module.validation_end(results_list, num_loaders)

        torch.cuda.empty_cache()
