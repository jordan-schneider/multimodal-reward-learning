from typing import List, TypeVar, get_origin

from cattrs import Converter


class StrictConverter(Converter):
    T = TypeVar("T")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_structure_hook(int, StrictConverter._strict_structure_int)
        self.register_structure_hook(str, StrictConverter._strict_structure_str)
        self.register_structure_hook_func(
            lambda t: get_origin(t) == list, StrictConverter._strict_structure_list
        )

    @staticmethod
    def _strict_structure_int(val, _):
        if not isinstance(val, int):
            raise ValueError(f"Expected int, got {val} of type {type(val)}")
        return val

    @staticmethod
    def _strict_structure_str(val, _):
        if not isinstance(val, str):
            raise ValueError(f"Expected str, got {val} of type {type(val)}")
        return val

    @staticmethod
    def _strict_structure_list(v: List[T], _) -> List[T]:
        if not isinstance(v, list):
            raise ValueError(f"Expected list, got {v} of type {type(v)}")
        return v
