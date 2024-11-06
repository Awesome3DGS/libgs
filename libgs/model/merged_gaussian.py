#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import List

import torch

from .gaussian import GaussianModel


class MergedGaussianModel:
    gaussian_models: List[GaussianModel]

    def __init__(self, gaussian_models: List[GaussianModel]):
        self.gaussian_models = gaussian_models

    @property
    def get_scaling(self):
        return torch.cat([m.get_scaling for m in self.gaussian_models])

    @property
    def get_rotation(self):
        return torch.cat([m.get_rotation for m in self.gaussian_models])

    @property
    def get_xyz(self):
        return torch.cat([m.get_xyz for m in self.gaussian_models])

    @property
    def get_features(self):
        return torch.cat([m.get_features for m in self.gaussian_models])

    @property
    def get_opacity(self):
        return torch.cat([m.get_opacity for m in self.gaussian_models])

    @property
    def active_sh_degree(self):
        return self.gaussian_models[0].active_sh_degree

    @active_sh_degree.setter
    def active_sh_degree(self, sh_degree):
        raise NotImplemented

    @property
    def max_sh_degree(self):
        return self.gaussian_models[0].max_sh_degree

    @max_sh_degree.setter
    def max_sh_degree(self, sh_degree):
        raise NotImplemented

    def get_covariance(self, scaling_modifier=1):
        return torch.cat(
            [m.get_covariance(scaling_modifier) for m in self.gaussian_models]
        )
