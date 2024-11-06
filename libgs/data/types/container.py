from types import SimpleNamespace
from typing import Union

from torch import DeviceObjType, is_tensor


class TensorSpace(SimpleNamespace):
    """A flexible container for batch managing PyTorch Tensors."""

    def device(self) -> DeviceObjType:
        for v in vars(self).values():
            if is_tensor(v):
                return v.device
        return torch.empty(0).device

    def to(
        self,
        device: Union[DeviceObjType, str],
        inplace: bool = False,
        **kwargs,
    ) -> "TensorSpace":
        fn = lambda v: v.to(device, **kwargs) if is_tensor(v) else v
        if inplace:
            for k, v in vars(self).items():
                setattr(self, k, fn(v))
            return self
        return self.__class__(**{k: fn(v) for k, v in vars(self).items()})
