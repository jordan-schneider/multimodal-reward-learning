import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd  # type: ignore
from mrl.util import dump, load


class Results:
    current_experiment: Optional[str]

    def __init__(self, outdir: Path, load_contents: bool = False):
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.experiments: Dict[str, Dict[str, Any]] = {}

        if load_contents:
            for experiment_dir in self.outdir.iterdir():
                if experiment_dir.is_dir():
                    experiment_name = experiment_dir.parts[-1]
                    self.start(experiment_name)
                    for file in experiment_dir.iterdir():
                        obj_name = file.stem
                        self.experiments[experiment_name][obj_name] = load(file)

    def start(self, experiment_name: str):
        self.experiments[experiment_name] = self.experiments.get(experiment_name, {})
        self.current_experiment = experiment_name
        (self.outdir / self.current_experiment).mkdir(exist_ok=True)

    def update(self, name: str, value: Any, save: bool = True) -> None:
        assert self.current_experiment is not None, "No current experiment"
        self.experiments[self.current_experiment][name] = value
        if save:
            dump(value, self.outdir / self.current_experiment / name)

    def has(self, name: str) -> bool:
        return any(name in exp.keys() for exp in self.experiments.values())

    def get(self, name: str) -> Any:
        assert self.current_experiment is not None, "No current experiment"
        return self.experiments[self.current_experiment].get(name)

    def getall(self, name: str) -> pd.DataFrame:
        if not self.has(name):
            raise ValueError(f"No {name} values in any experiment")

        out = pd.DataFrame(columns=["trial", "time", name])
        for exp_name, exp in self.experiments.items():
            if name not in exp.keys():
                logging.warning(
                    f"{name} not present in experiment {exp_name}, skipping"
                )
                continue

            value = exp[name]
            if isinstance(value, np.ndarray):
                if len(value.shape) > 1:
                    raise NotImplementedError(
                        f"Underlying array with {len(value.shape)} dims > 1 not supported"
                    )
                df = self.__make_df(name, value)
                df["trial"] = exp_name
                out = out.append(df.copy())
            elif isinstance(value, dict):
                for modality, v in value.items():
                    df = self.__make_df([name], v)
                    df["trial"] = exp_name
                    df["modality"] = modality
                    out = out.append(df.copy())

        out = out.reset_index(drop=True)
        return out

    def experiment_names(self) -> List[str]:
        return list(self.experiments.keys())

    @staticmethod
    def __make_df(columns: Sequence[str], value: np.ndarray) -> pd.DataFrame:
        assert value.ndim == 1, "Only 1D arrays supported"
        df = pd.DataFrame(value, columns=columns)
        df.index.name = "time"
        df = df.reset_index()
        return df

    def close(self):
        self.current_experiment = None
