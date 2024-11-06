# Credit: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/image_utils.py

from typing import Literal

import torch


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2, reduction: Literal["mean", "sum", "none"] = "mean"):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    result = 20 * torch.log10(1.0 / torch.sqrt(mse))
    if reduction == "mean":
        return result.mean()
    if reduction == "sum":
        return result.sum()
    return result
