import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from jaxtyping import Float
from torch import Tensor

from libgs.data.types import TensorSpace
from libgs.model.gaussian import GaussianModel
from libgs.utils.general import quaternion_multiply
from libgs.utils.sh import eval_sh


@dataclass
class RendererConfig:
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False
    debug_from: int = -1
    debug: bool = False


class Renderer(nn.Module):
    def __init__(
        self,
        config: RendererConfig,
        gaussians: GaussianModel,
    ):
        super().__init__()
        self.config = config
        self.gaussians = gaussians

    def create_screenspace_points(self):
        xyz = self.gaussians.get_xyz
        screenspace_points = torch.zeros_like(xyz, requires_grad=True) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        return screenspace_points

    def create_rasterizer(self, viewpoint, bg_color, scaling_modifier):
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint.image.shape[1]),
            image_width=int(viewpoint.image.shape[2]),
            tanfovx=math.tan(viewpoint.fovx * 0.5),
            tanfovy=math.tan(viewpoint.fovy * 0.5),
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint.world_view_transform,
            projmatrix=viewpoint.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint.camera_center,
            prefiltered=False,
            debug=self.config.debug,
        )
        return GaussianRasterizer(raster_settings=raster_settings)

    def apply_color(self, kwargs, viewpoint, override_color=None):
        if override_color is not None:
            return {**kwargs, "colors_precomp": override_color}

        if not self.config.convert_SHs_python:
            return {**kwargs, "shs": self.gaussians.get_features}

        shs_view = self.gaussians.get_features.transpose(1, 2).view(
            -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
        )
        dir_pp = self.gaussians.get_xyz - viewpoint.camera_center.repeat(
            self.gaussians.get_features.shape[0], 1
        )
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(self.gaussians.active_sh_degree, shs_view, dir_pp_normalized)
        return {**kwargs, "colors_precomp": torch.clamp_min(sh2rgb + 0.5, 0.0)}

    def forward(
        self,
        viewpoint: TensorSpace,
        bg_color: Float[Tensor, "3"],
        scaling_modifier: float = 1.0,
        override_color: Optional[Float[Tensor, "n 3"]] = None,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients
        # of the 2D (screen-space) means
        screenspace_points = self.create_screenspace_points()
        # Set up rasterization configuration
        rasterizer = self.create_rasterizer(viewpoint, bg_color, scaling_modifier)

        kwargs = dict(
            means3D=self.gaussians.get_xyz,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=None,
            opacities=self.gaussians.get_opacity,
            scales=None,
            rotations=None,
            cov3D_precomp=None,
        )

        # If precomputed 3d covariance is provided, use it. If not,
        # then it will be computed from scaling / rotation by the rasterizer.
        if self.config.compute_cov3D_python:
            kwargs["cov3D_precomp"] = self.gaussians.get_covariance(scaling_modifier)
        else:
            kwargs["scales"] = self.gaussians.get_scaling
            kwargs["rotations"] = self.gaussians.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is
        # desired to precompute colors from SHs in Python, do it.
        # If not, then SH -> RGB conversion will be done by rasterizer.
        kwargs = self.apply_color(kwargs, viewpoint, override_color)

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(**kwargs)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
