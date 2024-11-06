# Credit: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py


import torch

from libgs.model.gaussian import GaussianModel as BaseGaussianModel


class GaussianModel(BaseGaussianModel):
    def prune(self, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densify(self, max_grad, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

    def add_densification_stats(self, viewspace_point_grad, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
