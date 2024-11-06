from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Union

import torch
from absl import logging
from torch.nn.functional import l1_loss
from torchvision.utils import save_image

from libgs.data.types import TensorSpace
from libgs.metric import psnr, ssim
from libgs.metric.lpips import LPIPS
from libgs.pipeline import Module as BaseModule
from libgs.pipeline import ModuleConfig as BaseModuleConfig
from libgs.renderer import RendererConfig
from libgs.renderer.network_gui import interact_with_gui

from .data import DataModule
from .model.deformation import Deformation
from .model.gaussian import GaussianModel
from .renderer import Renderer


@dataclass
class OptimizeConfig:
    position_lr_init: float = 1.6e-4
    position_lr_final: float = 1.6e-6
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 20000
    deformation_lr_init: float = 1.6e-4
    deformation_lr_final: float = 1.6e-5
    deformation_lr_delay_mult: float = 0.01
    grid_lr_init: float = 1.6e-3
    grid_lr_final: float = 1.6e-4
    grid_lr_delay_mult: float = 0.01
    feature_lr: float = 2.5e-3
    opacity_lr: float = 0.05
    scaling_lr: float = 5e-3
    rotation_lr: float = 1e-3
    percent_dense: float = 0.01


@dataclass
class DensifyConfig:
    from_step: int = 500
    until_step: int = 10000
    interval: int = 100
    opacity_reset_interval: int = 60000
    opacity_threshold_coarse: float = 0.005
    opacity_threshold_fine_init: float = 0.005
    opacity_threshold_fine_after: float = 0.005
    grad_threshold_coarse: float = 0.0002
    grad_threshold_fine_init: float = 0.0002
    grad_threshold_fine_after: float = 0.0002
    pruning_from_step: int = 500
    pruning_interval: int = 100
    max_gaussians: int = 10000000
    min_gaussians: int = 200000


@dataclass
class DeformConfig:
    net_width: int = 128
    timebase_pe: int = 4
    defor_depth: int = 0
    posebase_pe: int = 10
    scale_rotation_pe: int = 2
    opacity_pe: int = 2
    timenet_output: int = 32
    grid_pe: int = 0
    no_grid: bool = False
    bounds: float = 1.6
    kplanes_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "grid_dimensions": 2,
            "input_coordinate_dim": 4,
            "output_coordinate_dim": 16,
            "resolution": [64, 64, 64, 150],
        }
    )
    multires: List[int] = field(default_factory=lambda: [1, 2])
    no_dx: bool = False
    no_ds: bool = False
    no_dr: bool = False
    no_do: bool = True
    no_dshs: bool = True
    apply_rotation: bool = False


@dataclass
class ModuleConfig(BaseModuleConfig):
    sh_degree: int = 3
    random_background: bool = False  # for training
    lambda_dssim: float = 0.0
    time_smoothness_weight: float = 0.001
    l1_time_planes: float = 0.0001
    plane_tv_weight: float = 0.0002
    saving_gs_steps: List[int] = field(default_factory=lambda: [7000, 15000])
    num_saving_images: int = 5
    full_eval: bool = False
    deform: DeformConfig = field(default_factory=DeformConfig)
    optimize: OptimizeConfig = field(default_factory=OptimizeConfig)
    densify: DensifyConfig = field(default_factory=DensifyConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)


class Module(BaseModule):
    datamodule: DataModule
    trainer: "Trainer"
    gaussians: GaussianModel
    background: torch.Tensor  # TODO remove
    stage: Literal["coarse", "fine"] = "coarse"

    @property
    def num_steps(self):
        return self.trainer.num_steps

    def setup(self):
        white_background = self.datamodule.config.white_background
        bg_color_fn = torch.ones if white_background else torch.zeros
        self.register_buffer("background", bg_color_fn(3, device=self.device))

        if self.config.full_eval:
            self.lpips_fn = LPIPS("vgg").to(self.device)

        point_cloud = self.datamodule.scene.point_cloud
        xyz_max = point_cloud.points.max(axis=0)
        xyz_min = point_cloud.points.min(axis=0)

        self.gaussians = GaussianModel(self.config.sh_degree)
        self.gaussians.create_from_pcd(point_cloud, self.datamodule.cameras_extent)
        self.deformation = Deformation(self.config.deform, self.gaussians)
        self.deformation.set_aabb(xyz_max, xyz_min)
        self.deformation = self.deformation.to(self.device)

        self.gaussians.training_setup(self.config.optimize)
        self.deformation.training_setup(self.config.optimize)

        self.renderer = Renderer(self.config.renderer, self.gaussians, self.deformation)

    def on_save_checkpoint(self, ckpt):
        ckpt["stage"] = self.stage
        ckpt["gaussians"] = self.gaussians.capture()
        ckpt["deformation"] = self.deformation.capture()

    def on_load_checkpoint(self, ckpt):
        self.stage = ckpt["stage"]
        self.gaussians.restore(ckpt["gaussians"], self.config.optimize)
        self.deformation.restore(ckpt["deformation"])

    def forward(self, viewpoint, training=False, **kwargs):
        bg_color = self.background
        if training and self.config.random_background:
            bg_color = torch.rand(3, device=bg_color.device)
        if self.stage == "fine":
            kwargs["time"] = viewpoint.frame / self.datamodule.num_frames
        return self.renderer(viewpoint, bg_color, **kwargs)

    def pre_training_step(self):
        interact_with_gui(
            self.global_step,
            self.renderer.config,  # can be modified
            self,
            self.datamodule.config.root,
            self.num_steps,
        )

    def training_step(self, viewpoints, current_step):
        self.gaussians.update_learning_rate(current_step)
        self.deformation.update_learning_rate(current_step)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if current_step % 1000 == 0:
            self.gaussians.oneupSHdegree()

        if (current_step - 1) == self.renderer.config.debug_from:
            self.renderer.config.debug = True

        images, gt_images, render_pkgs = [], [], []
        for viewpoint in viewpoints:
            render_pkg = self.forward(viewpoint, training=True)
            images.append(render_pkg["render"])
            gt_images.append(viewpoint.image)
            render_pkgs.append(render_pkg)

        images, gt_images = torch.stack(images), torch.stack(gt_images)

        loss_rgb = l1_loss(images, gt_images)
        loss, loss_dssim = loss_rgb, 0
        lambda_dssim = self.config.lambda_dssim
        if lambda_dssim > 0:
            loss_dssim = 1.0 - ssim(images, gt_images)
            loss = (1.0 - lambda_dssim) * loss_rgb + lambda_dssim * loss_dssim

        metrics = {
            "loss/rgb": loss_rgb,
            "loss/dssim": loss_dssim,
            "psnr": psnr(images, gt_images),
            "gs": self.gaussians.get_xyz.shape[0],
        }

        if self.stage == "fine" and self.config.time_smoothness_weight > 0:
            tv_loss = self.deformation.compute_regulation(
                self.config.time_smoothness_weight,
                self.config.l1_time_planes,
                self.config.plane_tv_weight,
            )
            loss = loss + tv_loss
            metrics["loss/tv"] = tv_loss

        self.log_dict(metrics)

        return loss, metrics, render_pkgs

    @torch.no_grad()
    def post_training_step(self, render_pkgs):
        if self.global_step < self.config.densify.until_step:
            # Densification, depend on grad
            self.densify_gaussians(render_pkgs)

        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        if self.stage == "fine":
            self.deformation.optimizer.step()
            self.deformation.optimizer.zero_grad(set_to_none=True)

        self.save_point_cloud()

    def densify_gaussians(self, render_pkgs):
        visibility_filters, viewspace_point_grads, radiis = [], [], []
        for render_pkg in render_pkgs:
            visibility_filters.append(render_pkg["visibility_filter"])
            viewspace_point_grads.append(render_pkg["viewspace_points"].grad)
            radiis.append(render_pkg["radii"])
        visibility_filter = torch.stack(visibility_filters).any(dim=0)
        viewspace_point_grad = torch.stack(viewspace_point_grads).sum(dim=0)
        radii = torch.stack(radiis).max(dim=0).values

        # Keep track of max radii in image-space for pruning
        self.gaussians.max_radii2D[visibility_filter] = torch.max(
            self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
        )
        self.gaussians.add_densification_stats(viewspace_point_grad, visibility_filter)

        opt = self.config.densify
        if self.stage == "coarse":
            opacity_threshold = opt.opacity_threshold_coarse
            grad_threshold = opt.grad_threshold_coarse
        else:

            def schedule(init, after):
                return init - self.global_step / opt.until_step * (init - after)

            opacity_threshold = schedule(
                opt.opacity_threshold_fine_init, opt.opacity_threshold_fine_after
            )
            grad_threshold = schedule(
                opt.grad_threshold_fine_init, opt.grad_threshold_fine_after
            )
        exceed_reset_inverval = self.global_step > opt.opacity_reset_interval
        size_threshold = 20 if exceed_reset_inverval else None

        densify_started = self.global_step > opt.from_step
        at_interval = self.global_step % opt.interval == 0
        within_limit = self.gaussians.get_xyz.shape[0] < opt.max_gaussians
        if densify_started and at_interval and within_limit:
            self.gaussians.densify(grad_threshold, self.datamodule.cameras_extent)

        pruning_started = self.global_step > opt.pruning_from_step
        at_interval = self.global_step % opt.pruning_interval == 0
        exceed_limit = self.gaussians.get_xyz.shape[0] > opt.min_gaussians
        if pruning_started and at_interval and exceed_limit:
            self.gaussians.prune(
                opacity_threshold,
                self.datamodule.cameras_extent,
                size_threshold,
            )

        white_background = self.datamodule.config.white_background
        at_reset_interval = self.global_step % opt.opacity_reset_interval == 0
        at_densify_start = self.global_step == opt.from_step
        if at_reset_interval or (white_background and at_densify_start):
            self.gaussians.reset_opacity()

    def validation_step(self, viewpoint: TensorSpace, idx: int, loader_idx: int):
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

    def validation_end(self, results: Union[dict, List[dict]], num_loaders: int = 1):
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
        self.log("total_points", self.gaussians.get_xyz.shape[0])

    def save_point_cloud(self):
        skiping = self.global_step not in self.config.saving_gs_steps
        if not self.trainer.is_last_step and skiping:
            return

        filename = f"gaussians-{self.global_step}.ply"
        save_path = self.output_dir / "point_cloud" / filename
        self.gaussians.save_ply(save_path)
        logging.info(f"\nSave gaussians to {save_path}")
