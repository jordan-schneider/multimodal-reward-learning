import pickle as pkl
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from mrl.dataset.trajectory_db import FeatureDataset


def main(dir: Path) -> None:
    dir = Path(dir)
    for path in dir.glob("*.pkl"):
        print(f"Fixing {path}")
        save = False
        data = pkl.load(path.open("rb"))
        if isinstance(data, pd.DataFrame):
            dataset = FeatureDataset(np.random.default_rng())
            dataset.df = data
            save = True
        else:
            dataset = data
        print(type(dataset))
        if "grid_shape" not in dataset.df.columns:
            dataset.df = dataset.df.assign(grid_shape=[(20, 20)] * len(dataset.df))
            save = True
        else:
            print("Already fixed")

        if save:
            pkl.dump(dataset, path.open("wb"))


if __name__ == "__main__":
    fire.Fire(main)
