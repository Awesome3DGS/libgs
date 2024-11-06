from typing import Optional

from jaxtyping import Float
from torch import Tensor, zeros_like

from libgs.data.types import TensorSpace
from libgs.renderer import Renderer as BaseRenderer
from libgs.renderer import RendererConfig

from .model.deformation import Deformation
from .model.gaussian import GaussianModel


class Renderer(BaseRenderer):
    gaussians: GaussianModel
    deformation: Deformation

    def __init__(
        self,
        config: RendererConfig,
        gaussians: GaussianModel,
        deformation: Deformation,
    ):
        super().__init__(config, gaussians)
        self.deformation = deformation

    def apply_deform(self, kwargs: dict, viewpoint: TensorSpace, time: float):
        (
            kwargs["means3D"],
            scales,
            rotations,
            opacities,
            shs,
        ) = self.deformation(
            self.gaussians.get_xyz,
            self.gaussians._scaling,
            self.gaussians._rotation,
            self.gaussians._opacity,
            self.gaussians.get_features,
            zeros_like(kwargs["means3D"][:, :1]) + time,
        )
        if not self.deformation.args.no_ds:
            kwargs["scales"] = self.gaussians.scaling_activation(scales)
        if not self.deformation.args.no_dr:
            kwargs["rotations"] = self.gaussians.rotation_activation(rotations)
        if not self.deformation.args.no_do:
            kwargs["opacities"] = self.gaussians.opacity_activation(opacities)
        if not self.deformation.args.no_dshs:
            kwargs["shs"] = shs
        return kwargs

    def forward(
        self,
        viewpoint: TensorSpace,
        bg_color: Float[Tensor, "3"],
        scaling_modifier: float = 1.0,
        override_color: Optional[Float[Tensor, "n 3"]] = None,
        time: Optional[float] = None,
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

        if time is not None:
            kwargs = self.apply_deform(kwargs, viewpoint, time)

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
