from dataclasses import is_dataclass
from importlib import import_module
from inspect import isclass
from typing import Dict, Tuple, Type

try:
    from typing import ModuleType  # py >= 3.8
except ImportError:
    from types import ModuleType  # py < 3.8

LOADED_PIPELINES: Dict[str, ModuleType] = {}


def load_module(name: str) -> ModuleType:
    if name in LOADED_PIPELINES:
        return LOADED_PIPELINES[name]

    try:
        LOADED_PIPELINES[name] = import_module(f"pipeline.{name}")
    except ModuleNotFoundError as e:
        if e.name == f"pipeline.{name}":
            LOADED_PIPELINES[name] = import_module(name)
        else:
            raise
    return LOADED_PIPELINES[name]


def load_pipeline(name: str) -> Tuple[Type, Type]:
    try:
        module = load_module(name)
        if not hasattr(module, "Config") or not is_dataclass(module.Config):
            raise AttributeError("Missing or invalid 'Config' class")
        if not hasattr(module, "Pipeline") or not isclass(module.Pipeline):
            raise AttributeError("Missing or invalid 'Pipeline' class")
        return module.Config, module.Pipeline
    except (ImportError, AttributeError) as e:
        print(f"[Pipeline] {e}")
        raise
