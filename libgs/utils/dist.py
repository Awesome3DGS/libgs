import torch.distributed as dist


def is_global_zero():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0
