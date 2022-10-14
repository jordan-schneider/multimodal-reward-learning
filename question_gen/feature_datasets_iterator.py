import pickle as pkl
from pathlib import Path
from typing import Iterable, Optional

from question_gen.trajectory_db import FeatureDataset


class FeatureDatasetsIterator(Iterable[FeatureDataset]):
    def __init__(self, paths: Iterable[Path], max_length: Optional[int] = None) -> None:
        self.paths = paths
        self.max_length = max_length
        if self.max_length is not None and self.max_length < 0:
            raise ValueError(f"max_length={self.max_length} must be non-negative")

    def __iter__(self):
        for path in self.paths:
            data: FeatureDataset = pkl.load(path.open("rb"))
            if not hasattr(data, "clip"):
                data = FeatureDataset.from_pickled(data)
            if self.max_length is not None:
                data = data.clip(
                    max_length=self.max_length, keep_last_action=self.max_length > 1
                )
            yield data
