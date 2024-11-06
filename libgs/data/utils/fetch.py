import random
from itertools import cycle
from typing import Any, Callable, Iterable, Optional

from torch.utils.data import DataLoader, RandomSampler


def repeat_iterable(iterable: Iterable):
    while True:
        yield from iterable


class Fetcher:
    def __init__(
        self,
        loader: DataLoader,
        cache: bool = True,
        move_to_device: Optional[Callable[[Any], Any]] = None,
    ):
        self.loader = loader
        self.cache = cache
        self.move_to_device = move_to_device
        self.num_items = len(loader)

        self.reset()

    def reset(self):
        if self.cache:
            self.cached_items = []
            self.loader_iter = cycle(self.loader)
        else:
            self.loader_iter = repeat_iterable(self.loader)

    def next(self):
        item = next(self.loader_iter)
        if self.move_to_device:
            item = self.move_to_device(item)
        if self.cache and len(self.cached_items) < self.num_items:
            self.cached_items.append(item)
            if len(self.cached_items) == self.num_items:
                if isinstance(self.loader.sampler, RandomSampler):
                    random.shuffle(self.cached_items)
                self.loader_iter = cycle(self.cached_items)
                self.cached_items = []
        return item
