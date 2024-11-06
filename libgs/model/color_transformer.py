from typing import Union

import torch
import torch.nn as nn


class ColorTransformer(nn.Module):
    def __init__(self, num_cameras: int, num_channels: int = 3):
        super().__init__()

        self.num_cameras = num_cameras
        self.num_channels = num_channels
        self.camera_key2ids = {}

        scale = torch.zeros(num_cameras, num_channels)
        offset = torch.zeros(num_cameras, num_channels)

        self.scale = nn.Parameter(scale)
        self.offset = nn.Parameter(offset)

    def forward(self, image: torch.Tensor, camera_id: Union[int, str]):
        if not isinstance(camera_id, int):
            if camera_id not in self.camera_key2ids:
                self.camera_key2ids[camera_id] = len(self.camera_key2ids)
            camera_id = self.camera_key2ids[camera_id]
        scale, offset = self.scale[camera_id], self.offset[camera_id]
        return image * scale.exp().view(-1, 1, 1) + offset.view(-1, 1, 1)

    def training_setup(self, lr: float = 1e-4):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-15)

    def capture(self):
        opt_dict = self.optimizer.state_dict()
        return self.scale, self.offset, self.camera_key2ids, opt_dict

    def restore(self, model_args, lr):
        self.scale, self.offset, self.camera_key2ids, opt_dict = model_args
        self.training_setup(lr)
        self.optimizer.load_state_dict(opt_dict)
