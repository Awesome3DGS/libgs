# Copy and modified from https://github.com/vict0rsch/torch_simple_timing

from time import time
from typing import Dict, List, Optional, Union

import torch

try:
    CUDA_AVALIABLE: bool = torch.cuda.is_available()
except:
    CUDA_AVALIABLE: bool = False


class Clock:
    def __init__(self, store: Optional[List[float]] = None):
        self.duration = None
        self.store = store

    def start(self) -> "Clock":
        self.start_time = time()
        return self

    def stop(self) -> None:
        self.end_time = time()
        self.duration = self.end_time - self.start_time
        if self.store is not None:
            self.store.append(self.duration)


class CudaClock(Clock):
    def start(self) -> "Clock":
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        self.start_event.record()

        return self

    def stop(self) -> None:
        self.end_event.record()
        torch.cuda.synchronize()
        self.duration = self.start_event.elapsed_time(self.end_event) / 1000
        if self.store is not None:
            self.store.append(self.duration)


class Timer:
    def __init__(self):
        self.reset()

    def reset(self, keys: Optional[Union[str, List[str]]] = None) -> None:
        if isinstance(keys, str):
            keys = [keys]

        if keys is None:
            self.times, self.clocks = {}, {}
        else:
            for k in keys:
                self.times.pop(k, None)
                self.clocks.pop(k, None)

    def clock(self, name: str, gpu: bool = True) -> Clock:
        if name not in self.clocks:
            self.times[name] = []
            clock_cls = CudaClock if gpu and CUDA_AVALIABLE else Clock
            self.clocks[name] = clock_cls(store=self.times[name])
        return self.clocks[name]

    def stats(
        self, names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        if names is None:
            names = self.times.keys()
        names = set(names)
        stats = {}
        for k, v in self.times.items():
            if k in names and len(v) > 0:
                t = torch.tensor(v).float()
                a = t.sum().item()
                m = torch.mean(t).item()
                s = torch.std(t).item() if len(v) > 1 else 0.0
                n = len(v)
                stats[k] = {"sum": a, "mean": m, "std": s, "n": n}
        return stats

    def display(
        self,
        names: Optional[List[str]] = None,
        stats: Dict[str, Dict[str, Union[int, float]]] = None,
        precision: int = 3,
        left: str = "\t",
        delimiter: str = "\n",
        align: bool = True,
    ):
        if stats is None:
            stats = self.stats(names)

        if align:
            max_key_len = max([len(k) for k in stats])

        def format(key, stat):
            prefix = f"{left}{key:>{max_key_len}}: " if align else f"{left}{key}: "
            total, mean, std, n = stat["sum"], stat["mean"], stat["std"], stat["n"]
            return f"{prefix}{mean:.{precision}f} ± {std:.{precision}f} (n={n}, sum={total:.{precision}f})"

        return delimiter.join([format(k, v) for k, v in stats.items()])
