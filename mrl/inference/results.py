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
                        if file.suffix not in [".pkl", ".npy"]:
                            continue
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

    def update_dict(self, name: str, key: Any, value: Any, save: bool = True) -> None:
        assert self.current_experiment is not None, "No current experiment"
        d = self.experiments[self.current_experiment].get(name, {})
        d[key] = value
        self.experiments[self.current_experiment][name] = d
        if save:
            dump(value, self.outdir / self.current_experiment / name)

    def has(self, name: str) -> bool:
        return any(name in exp.keys() for exp in self.experiments.values())

    def get(self, name: str) -> Any:
        assert self.current_experiment is not None, "No current experiment"
        return self.experiments[self.current_experiment].get(name)

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
        if not self.has(name):
            raise ValueError(f"No {name} values in any experiment")

        out = pd.DataFrame(columns=["trial", "time"])
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
        return out

    def experiment_names(self) -> List[str]:
        return list(self.experiments.keys())

    @staticmethod
    def __make_df(columns: Sequence[str], value: np.ndarray) -> pd.DataFrame:
        assert value.ndim == 1 or value.ndim == 2, "Only 1D and 2D arrays supported"
        df = pd.DataFrame(value, columns=columns)
        df.index.name = "time"
        df = df.reset_index()
        return df

    def close(self):
        self.current_experiment = None
