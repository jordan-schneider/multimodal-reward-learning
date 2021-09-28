from typing import Dict, Literal, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

Value = Union[float, torch.Tensor, np.ndarray]


class SequentialWriter:
    def __init__(self, writer: SummaryWriter) -> None:
        self.writer = writer
        self.counters: Dict[str, int] = {}

    def add_scalar(self, tag: str, scalar_value: Value) -> None:
        global_step = self.counters.get(tag, 0)
        self.writer.add_scalar(tag, scalar_value, global_step)
        self.counters[tag] = global_step + 1

    BinMethod = Literal[
        "tensorflow", "auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"
    ]

    def add_histogram(self, tag: str, values: Value, bins: BinMethod = "tensorflow") -> None:
        global_step = self.counters.get(tag, 0)
        self.writer.add_histogram(tag, values, global_step, bins=bins)
        self.counters[tag] = global_step + 1
