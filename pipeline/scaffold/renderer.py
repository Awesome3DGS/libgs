import math
from typing import List

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import repeat
from jaxtyping import Bool, Float
from torch import Tensor

from libgs.data.types import TensorSpace
from libgs.renderer import Renderer as BaseRenderer

from .model.gaussian import GaussianModel


class Renderer(BaseRenderer):
    gaussians: GaussianModel  # FIXME

    @property
    def is_training(self) -> bool:
        return self.gaussians.get_color_mlp.training

    def create_screenspace_points(
        self, xyz: Float[Tensor, "n d"], retain_grad: bool = False
    ) -> Float[Tensor, "n d"]:
        screenspace_points = torch.zeros_like(xyz, requires_grad=True) + 0
        if retain_grad:
            try:
                screenspace_points.retain_grad()
            except:
                pass
        return screenspace_points

    def create_rasterizer(
        self, viewpoint, bg_color, scaling_modifier
    ) -> GaussianRasterizer:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint.image.shape[1]),
            image_width=int(viewpoint.image.shape[2]),
            tanfovx=math.tan(viewpoint.fovx * 0.5),
            tanfovy=math.tan(viewpoint.fovy * 0.5),
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint.world_view_transform,
            projmatrix=viewpoint.full_proj_transform,
            sh_degree=1,
            campos=viewpoint.camera_center,
            prefiltered=False,
            debug=self.config.debug,
        )
        return GaussianRasterizer(raster_settings=raster_settings)

    def prefilter_voxel(
        self,
        viewpoint: TensorSpace,
        bg_color: Float[Tensor, "3"],
        scaling_modifier: float = 1.0,
    ) -> Bool[Tensor, "n 3"]:
        means3D = self.gaussians.get_anchor
        self.create_screenspace_points(means3D, True)
        rasterizer = self.create_rasterizer(viewpoint, bg_color, scaling_modifier)

        scales, rotations, cov3D_precomp = None, None, None
        if self.config.compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

        radii_pure = rasterizer.visible_filter(
            means3D=means3D,
            scales=scales[:, :3],
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        return radii_pure > 0

    def generate_neural_gaussians(
        self,
        viewpoint: TensorSpace,
        visible_mask=None,
        is_training=False,
    ) -> List[Tensor]:
        ## view frustum filtering for acceleration
        if visible_mask is None:
            anchor = self.gaussians.get_anchor
            visible_mask = torch.ones_like(anchor[:, 0], dtype=torch.bool)

        feat = self.gaussians._anchor_feat[visible_mask]
        anchor = self.gaussians.get_anchor[visible_mask]
        grid_offsets = self.gaussians._offset[visible_mask]
        grid_scaling = self.gaussians.get_scaling[visible_mask]

        ## get view properties for anchor
        ob_view = anchor - viewpoint.camera_center
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        ob_view = ob_view / ob_dist

        ## view-adaptive feature
        if self.gaussians.use_feat_bank:
            cat_view = torch.cat([ob_view, ob_dist], dim=1)
            # shape of bank_weight: [n, 1, 3]
            bank_weight = self.gaussians.get_featurebank_mlp(cat_view).unsqueeze(dim=1)
            ## multi-resolution feat
            feat = feat.unsqueeze(dim=-1)
            feat = (
                feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1]
                + feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2]
                + feat[:, ::1, :1] * bank_weight[:, :, 2:]
            )
            feat = feat.squeeze(dim=-1)  # [n, c]

        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N, c+3+1]
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)  # [N, c+3]
        if self.gaussians.appearance_dim > 0:
            camera_indicies = torch.ones_like(cat_local_view[:, 0], dtype=torch.long)
            camera_indicies = camera_indicies * viewpoint.uid
            appearance = self.gaussians.get_appearance(camera_indicies)

        # get offset's opacity
        if self.gaussians.add_opacity_dist:
            neural_opacity = self.gaussians.get_opacity_mlp(cat_local_view)  # [N, k]
        else:
            neural_opacity = self.gaussians.get_opacity_mlp(cat_local_view_wodist)

        # opacity mask generation
        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity > 0.0).view(-1)

        # select opacity
        opacity = neural_opacity[mask]

        # get offset's color
        if self.gaussians.add_color_dist:
            local_view_for_color = cat_local_view
        else:
            local_view_for_color = cat_local_view_wodist
        if self.gaussians.appearance_dim > 0:
            local_view_for_color = torch.cat([local_view_for_color, appearance], dim=1)
        color = self.gaussians.get_color_mlp(local_view_for_color)
        color = color.reshape([anchor.shape[0] * self.gaussians.n_offsets, 3])

        # get offset's cov
        if self.gaussians.add_cov_dist:
            scale_rot = self.gaussians.get_cov_mlp(cat_local_view)
        else:
            scale_rot = self.gaussians.get_cov_mlp(cat_local_view_wodist)
        scale_rot = scale_rot.reshape([anchor.shape[0] * self.gaussians.n_offsets, 7])

        # offsets
        offsets = grid_offsets.view([-1, 3])

        # combine for parallel masking
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(
            concatenated, "n (c) -> (n k) (c)", k=self.gaussians.n_offsets
        )
        concatenated_all = torch.cat(
            [concatenated_repeated, color, scale_rot, offsets], dim=-1
        )
        masked = concatenated_all[mask]
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split(
            [6, 3, 3, 7, 3], dim=-1
        )

        # post-process cov
        scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
        rot = self.gaussians.rotation_activation(scale_rot[:, 3:7])

        # post-process offsets to get centers for gaussians
        offsets = offsets * scaling_repeat[:, :3]
        xyz = repeat_anchor + offsets

        if is_training:
            return xyz, color, opacity, scaling, rot, neural_opacity, mask
        else:
            return xyz, color, opacity, scaling, rot

    def forward(
        self,
        viewpoint: TensorSpace,
        bg_color: Float[Tensor, "3"],
        scaling_modifier: float = 1,
        retain_grad: bool = False,
    ) -> dict:
        visible_mask = self.prefilter_voxel(viewpoint, bg_color, scaling_modifier)
        neural_gaussians_result = self.generate_neural_gaussians(
            viewpoint, visible_mask, self.is_training
        )
        xyz, color, opacity, scaling, rot = neural_gaussians_result[:5]
        if self.is_training:
            neural_opacity, mask = neural_gaussians_result[5:]

        screenspace_points = self.create_screenspace_points(xyz, retain_grad)
        rasterizer = self.create_rasterizer(viewpoint, bg_color, scaling_modifier)
        rendered_image, radii = rasterizer(
            means3D=xyz,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=color,
            opacities=opacity,
            scales=scaling,
            rotations=rot,
            cov3D_precomp=None,
        )

        result = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
        if self.is_training:
            result.update(
                {
                    "selection_mask": mask,
                    "neural_opacity": neural_opacity,
                    "scaling": scaling,
                    "voxel_visible_mask": visible_mask,
                }
            )
        return result
