import torch
import torch.nn as nn
import torch.nn.init as init

from libgs.utils.general import batch_quaternion_multiply, get_expon_lr_func

from .hexplane import HexPlaneField
from .regulation import compute_plane_smoothness


class Deformation(nn.Module):
    def __init__(self, args, gaussians):
        super().__init__()

        self.D = args.defor_depth
        self.W = args.net_width
        self.grid_pe = args.grid_pe
        self.no_grid = args.no_grid
        self.args = args

        self.register_buffer(
            "pos_poc", torch.FloatTensor([(2**i) for i in range(args.posebase_pe)])
        )

        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.create_net()
        self.apply(initialize_weights)

        self.gaussians = gaussians

    @property
    def get_aabb(self):
        return self.grid.get_aabb

    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb", xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)

    def capture(self):
        return self.state_dict(), self.optimizer.state_dict()

    def restore(self, model_args):
        state, opt_state = model_args
        self.load_state_dict(state)
        self.optimizer.load_state_dict(opt_state)

    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe != 0:
            grid_out_dim = self.grid.feat_dim + (self.grid.feat_dim) * 2
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4, self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim, self.W)]

        for i in range(self.D - 1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W, self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3)
        )
        self.scales_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3)
        )
        self.rotations_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 4)
        )
        self.opacity_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1)
        )
        self.shs_deform = nn.Sequential(
            nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 16 * 3)
        )

    def query_time(self, xyz_emb, time_emb):
        if self.no_grid:
            hidden = torch.cat([xyz_emb[:, :3], time_emb[:, :1]], -1)
        else:
            grid_feature = self.grid(xyz_emb[:, :3], time_emb[:, :1])
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature, self.grid_pe)
            hidden = torch.cat([grid_feature], -1)

        hidden = self.feature_out(hidden)

        return hidden

    def forward(
        self, xyz, scaling=None, rotation=None, opacity=None, shs=None, times_sel=None
    ):
        xyz_emb = poc_fre(xyz, self.pos_poc)
        hidden = self.query_time(xyz_emb, times_sel)

        if not self.args.no_dx:
            dx = self.pos_deform(hidden)
            xyz = xyz + dx

        if not self.args.no_ds:
            ds = self.scales_deform(hidden)
            scaling = scaling + ds

        if not self.args.no_dr:
            dr = self.rotations_deform(hidden)
            if self.args.apply_rotation:
                rotation = batch_quaternion_multiply(rotation, dr)
            else:
                rotation = rotation + dr

        if not self.args.no_do:
            do = self.opacity_deform(hidden)
            opacity = opacity + do

        if not self.args.no_dshs:
            dshs = self.shs_deform(hidden).reshape([shs.shape[0], 16, 3])
            shs = shs + dshs

        return xyz, scaling, rotation, opacity, shs

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" not in name:
                parameter_list.append(param)
        return parameter_list

    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" in name:
                parameter_list.append(param)
        return parameter_list

    def training_setup(self, training_args):
        spatial_lr_scale = self.gaussians.spatial_lr_scale
        l = [
            {
                "params": list(self.get_mlp_parameters()),
                "lr": training_args.deformation_lr_init * spatial_lr_scale,
                "name": "deformation",
            },
            {
                "params": list(self.get_grid_parameters()),
                "lr": training_args.grid_lr_init * spatial_lr_scale,
                "name": "grid",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deformation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.deformation_lr_init * spatial_lr_scale,
            lr_final=training_args.deformation_lr_final * spatial_lr_scale,
            lr_delay_mult=training_args.deformation_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.grid_scheduler_args = get_expon_lr_func(
            lr_init=training_args.grid_lr_init * spatial_lr_scale,
            lr_final=training_args.grid_lr_final * spatial_lr_scale,
            lr_delay_mult=training_args.deformation_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group["lr"] = lr

    def _plane_regulation(self):
        multi_res_grids = self.grid.grids
        total = 0
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids = [0, 1, 3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total

    def _time_regulation(self):
        multi_res_grids = self.grid.grids
        total = 0
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids = [2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total

    def _l1_regulation(self):
        multi_res_grids = self.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total

    def compute_regulation(
        self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight
    ):
        return (
            plane_tv_weight * self._plane_regulation()
            + time_smoothness_weight * self._time_regulation()
            + l1_time_planes_weight * self._l1_regulation()
        )


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight, gain=1)


def poc_fre(input_data, poc_buf):
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb
