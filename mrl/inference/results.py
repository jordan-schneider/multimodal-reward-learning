import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd  # type: ignore

from mrl.util import dump, load


class Results:
    current_experiment_name: Optional[str]

    def __init__(self, outdir: Path):
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.experiment: Dict[str, Any] = {}

    def start(self, experiment_name: str):
        self.current_experiment_name = experiment_name
        (self.outdir / self.current_experiment_name).mkdir(exist_ok=True)
        self._load(self.current_experiment_name)

    def _load(self, experiment_name: str):
        experiment_dir = self.outdir / experiment_name
        self.experiment = {}
        if experiment_dir.is_dir():
            for file in experiment_dir.iterdir():
                if file.suffix not in [".pkl", ".npy"]:
                    continue
                obj_name = file.stem
                self.experiment[obj_name] = load(file)

    def update(self, name: str, value: Any, save: bool = True) -> None:
        assert self.current_experiment_name is not None, "No current experiment"
        self.experiment[name] = value
        if save:
            dump(value, self.outdir / self.current_experiment_name / name)

    def update_dict(self, name: str, key: Any, value: Any, save: bool = True) -> None:
        assert self.current_experiment_name is not None, "No current experiment"
        d = self.experiment.get(name, {})
        d[key] = value
        self.experiment[name] = d
        if save:
            dump(d, self.outdir / self.current_experiment_name / name)

    def has(self, name: str) -> bool:
        """Returns true if the current experiment has a key with the given name."""
        return name in self.experiment.keys()

    def any_has(self, name: str) -> bool:
        current_experiment_name = self.current_experiment_name
        for experiment_name in self.experiment_names():
            self._load(experiment_name)
            if self.has(name):
                self._load(current_experiment_name)
                return True
        self._load(current_experiment_name)
        return False

    def get(self, name: str) -> Any:
        assert self.current_experiment_name is not None, "No current experiment"
        return self.experiment.get(name)

    def getall(self, name: str) -> pd.DataFrame:
        """Produces a dataframe of the given value for all trials.

        Supports 1D numpy array global underlying values, or 1D or 2D underlying values per trial. 2D underlying values
        must have the first axis be time, and the second axis will be spread out over multiple columns in the dataframe.

        For example if name="mean_reward" is a (T, 6) array of reward values, the output dataframe will have columns
        ["trial", "time", "mean_reward_0", "mean_reward_1", "mean_reward_2", "mean_reward_3", "mean_reward_4",
        "mean_reward_5"].

        Args:
            name (str): The name of the value to colelct.

        Raises:
            ValueError: If name is not the name of any value.
            NotImplementedError: Handling global values of more than 1D.
            NotImplementedError: Handling per-trial values of more than 2D.

        Returns:
            pd.DataFrame: Dataframe with trial, time, and 1 or more value columns with data across all trials.
        """
        if not self.any_has(name):
            raise ValueError(f"No {name} values in any experiment")

        current_experiment_name = self.current_experiment_name

        out = pd.DataFrame(columns=["trial", "time"])
        for exp_name in self.experiment_names():
            self._load(exp_name)
            if name not in self.experiment.keys():
                logging.warning(
                    f"{name} not present in experiment {exp_name}, skipping"
                )
                continue

            value = self.experiment[name]
            if isinstance(value, np.ndarray):
                if len(value.shape) > 1:
                    raise NotImplementedError(
                        f"Underlying array with {len(value.shape)} dims > 1 not supported"
                    )
                df = self.__make_df([name], value)
                df["trial"] = exp_name
                out = out.append(df.copy())
            elif isinstance(value, dict):
                for modality, v in value.items():
                    assert isinstance(v, np.ndarray)
                    if len(v.shape) == 1:
                        df = self.__make_df([name], v)
                        df["trial"] = exp_name
                        df["modality"] = modality
                        out = pd.concat((out, df.copy()))
                    elif len(v.shape) == 2:
                        df = self.__make_df(
                            [f"{name}_{dim}" for dim in range(v.shape[1])], v
                        )
                        df["trial"] = exp_name
                        df["modality"] = modality
                        out = pd.concat((out, df.copy()))
                    else:
                        raise NotImplementedError(
                            f"Underlying array with {len(v.shape)} dims > 2 not supported"
                        )

        out = out.reset_index(drop=True)
        if current_experiment_name is not None:
            self._load(current_experiment_name)
        return out

    def experiment_names(self) -> List[str]:
        return list([d.name for d in self.outdir.iterdir() if d.is_dir()])

    @staticmethod
    def __make_df(columns: Sequence[str], value: np.ndarray) -> pd.DataFrame:
        assert value.ndim == 1 or value.ndim == 2, "Only 1D and 2D arrays supported"
        df = pd.DataFrame(value, columns=columns)
        df.index.name = "time"
        df = df.reset_index()
        return df

    def close(self):
        self.current_experiment_name = None
        self.experiment = {}
