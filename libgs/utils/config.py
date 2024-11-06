import inspect
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import MISSING, asdict, fields, is_dataclass
from pathlib import Path
from typing import Literal, Tuple, Type, get_args, get_origin

import yaml
from omegaconf import OmegaConf as oc


class MissingFieldError(Exception):
    def __init__(self, field_name):
        message = f"Field '{field_name}' is required but not found"
        self.field_name = field_name
        super().__init__(message)


SPECIAL_TYPE_CONVERTERS = {
    Path: Path,
}


def convert_str_to_type(field_type, field_value):
    if get_origin(field_type) is Literal:
        if field_value not in get_args(field_type):
            warnings.warn(f"'{field_value}' is not valid value for {field_type}")
        return field_value

    if field_type in SPECIAL_TYPE_CONVERTERS:
        return SPECIAL_TYPE_CONVERTERS[field_type](field_value)

    if callable(getattr(field_type, "from_str", None)):
        return field_type.from_str(field_value)

    try:
        warnings.warn(f"Try to cast '{field_value}' to {field_type}")
        return field_type(field_value)
    except (TypeError, ValueError):
        warnings.warn(f"Failed to cast '{field_value}' to {field_type}")
        return field_value


def to_structured(config: oc, dataclass_type: Type, default=None) -> Tuple[Type, oc]:
    """
    Convert OmegaConf to a structured dataclass instance.

    Parameters:
    - config: OmegaConf configuration
    - dataclass_type: The dataclass type to which the configuration will be converted

    Returns:
    - An instance of the specified dataclass with values populated from the OmegaConf configuration
    """
    if not dataclass_type or not hasattr(dataclass_type, "__dataclass_fields__"):
        raise ValueError("Invalid dataclass type")

    if not inspect.isclass(dataclass_type):
        default = dataclass_type if default is None else default
        dataclass_type = dataclass_type.__class__

    if default is not None and not isinstance(default, dataclass_type):
        raise ValueError("Default value is not an instance of dataclass_type")

    dataclass_fields = fields(dataclass_type)
    dataclass_args = {}

    for field in dataclass_fields:
        field_name, field_type = field.name, field.type
        if field_name.startswith("_"):  # skip private field
            continue

        if is_dataclass(field_type):
            nested_config, nested_default = config.get(field_name, {}), None
            if hasattr(default, field_name):
                nested_default = getattr(default, field_name)
            elif field.default is not MISSING:
                nested_default = field.default
            try:
                dataclass_args[field_name], _ = to_structured(
                    nested_config, field_type, nested_default
                )
            except MissingFieldError as e:
                raise MissingFieldError(f"{field_name}.{e.field_name}")
        elif field_name in config:
            field_value = config[field_name]
            if isinstance(field_value, str) and field_type != str:
                field_value = convert_str_to_type(field_type, field_value)
            dataclass_args[field_name] = field_value
        elif default and hasattr(default, field_name):
            dataclass_args[field_name] = getattr(default, field_name)
        elif field.default is not MISSING:
            dataclass_args[field_name] = field.default
        elif field.default_factory is not MISSING:
            dataclass_args[field_name] = field.default_factory()
        else:
            raise MissingFieldError(field_name)

    remaining_fields = {
        key: value for key, value in config.items() if key not in dataclass_args
    }
    remaining = oc.create(remaining_fields)

    return dataclass_type(**dataclass_args), remaining


def to_yaml(obj, default_flow_style=False, sort_keys=False, **kwargs):
    def _convert_to_yaml(obj):
        if oc.is_config(obj):
            return oc.to_container(obj)
        if is_dataclass(obj):
            return _convert_to_yaml(asdict(obj))
        if isinstance(obj, Mapping):
            return {k: _convert_to_yaml(v) for k, v in obj.items()}
        if isinstance(obj, Iterable) and not isinstance(obj, str):
            return [_convert_to_yaml(item) for item in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    kwargs["sort_keys"] = sort_keys
    kwargs["default_flow_style"] = default_flow_style
    return yaml.dump(_convert_to_yaml(obj), **kwargs)
