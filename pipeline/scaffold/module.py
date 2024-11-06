from dataclasses import asdict, dataclass, field
from typing import List

import torch
from absl import logging
from torch.nn.functional import l1_loss
from torchvision.utils import save_image

from libgs.metric import psnr, ssim
from libgs.metric.lpips import LPIPS
from libgs.pipeline import Module as BaseModule
from libgs.pipeline import ModuleConfig as BaseModuleConfig
from libgs.renderer import RendererConfig
from libgs.renderer.network_gui import interact_with_gui

from .data import DataModule
from .model.gaussian import GaussianModel
from .renderer import Renderer


@dataclass
class GaussianConfig:
    feat_dim: int = 32
    n_offsets: int = 10
    voxel_size: float = 0.001
    update_depth: int = 3
    update_init_factor: int = 16
    update_hierachy_factor: int = 4
    use_feat_bank: bool = False
    appearance_dim: int = 0
    ratio: int = 1
    add_opacity_dist: bool = False
    add_cov_dist: bool = False
    add_color_dist: bool = False


@dataclass
class OptimizeConfig:
    # position
    position_lr_init: float = 0.0
    position_lr_final: float = 0.0
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    # offset
    offset_lr_init: float = 0.01
    offset_lr_final: float = 0.0001
    offset_lr_delay_mult: float = 0.01
    offset_lr_max_steps: int = 30_000
    # anchor attributes
    feature_lr: float = 0.0075
    opacity_lr: float = 0.02
    scaling_lr: float = 0.007
    rotation_lr: float = 0.002
    # opacity mlp
    mlp_opacity_lr_init: float = 0.002
    mlp_opacity_lr_final: float = 0.00002
    mlp_opacity_lr_delay_mult: float = 0.01
    mlp_opacity_lr_max_steps: int = 30_000
    # cov mlp
    mlp_cov_lr_init: float = 0.004
    mlp_cov_lr_final: float = 0.004
    mlp_cov_lr_delay_mult: float = 0.01
    mlp_cov_lr_max_steps: bool = 30_000
    # color mlp
    mlp_color_lr_init: float = 0.008
    mlp_color_lr_final: float = 0.00005
    mlp_color_lr_delay_mult: float = 0.01
    mlp_color_lr_max_steps: int = 30_000
    # feature bank mlp
    mlp_featurebank_lr_init: float = 0.01
    mlp_featurebank_lr_final: float = 0.00001
    mlp_featurebank_lr_delay_mult: float = 0.01
    mlp_featurebank_lr_max_steps: int = 30_000
    # appearance
    appearance_lr_init: float = 0.05
    appearance_lr_final: float = 0.0005
    appearance_lr_delay_mult: float = 0.01
    appearance_lr_max_steps: bool = 30_000
    # dense percent
    percent_dense: float = 0.01


@dataclass
class DensifyConfig:
    start_stat: int = 500
    update_from: int = 1500
    update_interval: int = 100
    update_until: int = 15_000
    min_opacity: float = 0.005
    success_threshold: float = 0.8
    densify_grad_threshold: float = 0.0002


@dataclass
class ModuleConfig(BaseModuleConfig):
    random_background: bool = False  # for training
    lambda_dssim: float = 0.2
    saving_gs_steps: List[int] = field(default_factory=lambda: [7000, 15000])
    num_saving_images: int = 5
    full_eval: bool = False
    gaussian: GaussianConfig = field(default_factory=GaussianConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    optimize: OptimizeConfig = field(default_factory=OptimizeConfig)
    densify: DensifyConfig = field(default_factory=DensifyConfig)


class Module(BaseModule):
    datamodule: DataModule
    trainer: "Trainer"
    gaussians: GaussianModel
    background: torch.Tensor  # TODO remove

    @property
    def num_steps(self):
        return self.trainer.num_steps

    def setup(self):
        white_background = self.datamodule.config.white_background
        bg_color_fn = torch.ones if white_background else torch.zeros
        self.register_buffer("background", bg_color_fn(3, device=self.device))

        if self.config.full_eval:
            self.lpips_fn = LPIPS("vgg").to(self.device)

        self.gaussians = GaussianModel(**asdict(self.config.gaussian))
        self.gaussians.set_appearance(len(self.datamodule.scene.train_dataset))
        self.gaussians.create_from_pcd(
            self.datamodule.scene.point_cloud, self.datamodule.cameras_extent
        )
        self.gaussians.training_setup(self.config.optimize)
        self.renderer = Renderer(self.config.renderer, self.gaussians)

    def on_save_checkpoint(self, ckpt):
        ckpt["gaussians"] = self.gaussians.capture()
        mlps = ["mlp_feature_bank", "mlp_opacity", "mlp_cov", "mlp_color"]
        embds = ["embedding_appearance"]
        for name in mlps + embds:
            if (m := getattr(self.gaussians, name, None)) is not None:
                ckpt[name] = m.state_dict()

    def on_load_checkpoint(self, ckpt):
        self.gaussians.restore(ckpt["gaussians"], self.config.optimize)
        mlps = ["mlp_feature_bank", "mlp_opacity", "mlp_cov", "mlp_color"]
        embds = ["embedding_appearance"]
        for name in mlps + embds:
            if (m := getattr(self.gaussians, name, None)) is not None:
                m.load_state_dict(ckpt[name])

    def forward(self, viewpoint, training=False, **kwargs):
        bg_color = self.background
        if training and self.config.random_background:
            bg_color = torch.rand(3, device=bg_color.device)
        return self.renderer(viewpoint, bg_color, **kwargs)

    def pre_training_step(self):
        interact_with_gui(
            self.global_step,
            self.renderer.config,  # can be modified
            self,
            self.datamodule.config.root,
            self.num_steps,
        )

    def training_step(self, viewpoint, current_step):
        self.gaussians.update_learning_rate(current_step)
        if (current_step - 1) == self.renderer.config.debug_from:
            self.renderer.config.debug = True

        retain_grad = current_step < self.config.densify.update_until
        render_pkg = self.forward(viewpoint, training=True, retain_grad=retain_grad)
        image, gt_image = render_pkg["render"], viewpoint.image
        loss_rgb = l1_loss(image, gt_image)
        loss_dssim = 1.0 - ssim(image, gt_image)
        loss_reg = render_pkg["scaling"].prod(dim=1).mean()

        lambda_dssim = self.config.lambda_dssim
        loss = (1.0 - lambda_dssim) * loss_rgb + lambda_dssim * loss_dssim
        loss = loss + 0.01 * loss_reg
        metrics = {
            "loss/rgb": loss_rgb,
            "loss/dssim": loss_dssim,
            "loss/reg": loss_reg,
            "psnr": psnr(image, gt_image),
            "gs": self.gaussians.get_anchor.shape[0],
        }
        self.log_dict(metrics)

        return loss, metrics, render_pkg

    @torch.no_grad()
    def post_training_step(self, render_pkg):
        if self.global_step < self.config.densify.update_until:
            self.densify_gaussians(render_pkg)
        elif self.global_step == self.config.densify.update_until:
            del self.gaussians.opacity_accum
            del self.gaussians.offset_gradient_accum
            del self.gaussians.offset_denom
            torch.cuda.empty_cache()

        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.save_point_cloud()

    def densify_gaussians(self, render_pkg):
        opt = self.config.densify
        if self.global_step > opt.start_stat:
            self.gaussians.training_statis(
                render_pkg["viewspace_points"],
                render_pkg["neural_opacity"],
                render_pkg["visibility_filter"],
                render_pkg["selection_mask"],
                render_pkg["voxel_visible_mask"],
            )

        if (
            self.global_step > opt.update_from
            and self.global_step % opt.update_interval == 0
        ):
            self.gaussians.adjust_anchor(
                check_interval=opt.update_interval,
                success_threshold=opt.success_threshold,
                grad_threshold=opt.densify_grad_threshold,
                min_opacity=opt.min_opacity,
            )

    def validation_step(self, viewpoint, idx, loader_idx):
        if idx == 0:
            self.gaussians.eval()

        image = self.forward(viewpoint)["render"].clamp(0.0, 1.0)
        gt_image = viewpoint.image
        if idx <= self.config.num_saving_images:
            root = self.datamodule.config.root
            path = self.output_dir / f"image" / viewpoint.path.relative_to(root)
            save_dir, name, ext = path.parent, path.stem, path.suffix
            save_dir.mkdir(parents=True, exist_ok=True)
            save_image(image, save_dir / f"{name}_step{self.global_step}{ext}")

            testing_steps = self.trainer.config.testing_steps + [self.num_steps]
            if self.global_step == testing_steps[0]:
                save_image(gt_image, save_dir / f"{name}_gt{ext}")

        metrics = dict(l1=l1_loss(image, gt_image), psnr=psnr(image, gt_image))
        if self.config.full_eval:
            metrics["ssim"] = ssim(image, gt_image)
            metrics["lpips"] = self.lpips_fn(image, gt_image).mean()
        return metrics

    def validation_end(self, results, num_loaders=1):
        def fn(metrics_list, idx):
            mean = lambda xs: sum(xs) / len(xs)
            metrics = {k: mean([ms[k] for ms in metrics_list]) for k in metrics_list[0]}
            name = self.datamodule.eval_names[idx]
            self.log_dict({f"eval-{name}/{k}": v for k, v in metrics.items()})
            logging.info(
                f"Evaluate {name} dataset at step {self.global_step}:\n\t"
                + " | ".join([f"{k.upper()}: {v:.4f}" for k, v in metrics.items()])
            )

        results = [results] if num_loaders == 1 else results
        for idx, result in enumerate(results):
            fn(result, idx)

        self.log_histogram("scene/opacity_histogram", self.gaussians.get_opacity)
        self.log("total_points", self.gaussians.get_anchor.shape[0])

        self.gaussians.train()

    def save_point_cloud(self):
        skiping = self.global_step not in self.config.saving_gs_steps
        if not self.trainer.is_last_step and skiping:
            return

        filename = f"gaussians-{self.global_step}.ply"
        save_path = self.output_dir / "point_cloud" / filename
        self.gaussians.save_ply(save_path)
        logging.info(f"\nSave gaussians to {save_path}")
