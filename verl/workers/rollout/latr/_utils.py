# Utility functions extracted from LATR's tools/utils.py.
# Only the functions needed by the LATR rollout code are included here.

from __future__ import annotations

import dataclasses
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin

import torch

T = TypeVar("T")


def init_dataclass_from_dict(class_type: type[T], args: dict, auto_cast: bool = True) -> T:
    """
    Initialize a dataclass from a dictionary.
    Non-existing fields will be ignored.
    """
    fields = dataclasses.fields(class_type)  # type: ignore
    d = {}
    for field in fields:
        if field.name not in args:
            continue

        name = field.name
        value = args[name]
        if not auto_cast:
            d[name] = value
            continue

        _type = field.type
        if isinstance(_type, str):
            _type = eval(_type)
        # Handle Union types (e.g., int | None). Cast to the first argument type.
        origin = get_origin(_type)
        if origin in (Union, UnionType):
            type_candidates = get_args(_type)
        elif origin is not None:
            type_candidates = [origin]
        else:
            type_candidates = [_type]
        if value is not None:
            if isinstance(value, str):
                try:
                    _value = eval(value)
                    if any(isinstance(_value, _type) for _type in type_candidates):
                        value = _value
                except Exception:
                    pass
            else:
                try:
                    value = type_candidates[0](value)
                except Exception:
                    pass
        d[name] = value

    return class_type(**d)


def update_additive_stats(stats: dict[str, Any], new_stats: dict[str, Any]):
    """Add new_stats to stats for all keys. They must have identical structure (can be nested)."""
    for k, v in new_stats.items():
        if k not in stats:
            stats[k] = v
            continue
        if isinstance(v, dict):
            update_additive_stats(stats[k], v)
        else:
            stats[k] += v


def get_repeat_interleave(x: torch.Tensor) -> int:
    """
    Detect the repeat-interleave factor of a tensor's first dimension.
    Reverse operation of torch.repeat_interleave.
    Returns 1 if not interleaved.
    """
    if x.numel() == 0:
        return 1

    first_dim_size = x.size(0)

    if first_dim_size == 1:
        return 1

    first_element = x[0]
    interval = None

    for i in range(1, first_dim_size):
        if not torch.equal(x[i], first_element):
            interval = i
            break

    if interval is None:
        return first_dim_size

    if first_dim_size % interval != 0:
        raise ValueError("Input tensor is not interleaved")

    original_size = first_dim_size // interval

    for i in range(original_size):
        start_idx = i * interval
        end_idx = start_idx + interval
        group = x[start_idx:end_idx]
        if not torch.all(group == group[0]):
            raise ValueError("Input tensor is not interleaved")

    return interval
