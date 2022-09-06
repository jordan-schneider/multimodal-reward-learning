import pickle as pkl
from pathlib import Path

import fire
import numpy as np


def main(dir: Path) -> None:
    dir = Path(dir)
    for path in dir.glob("*.pkl"):
        print(f"Fixing {path}")
        data = pkl.load(path.open("rb"))
        if "grid_shape" not in data.df.columns:
            data.df.assign(grid_shape=[(20, 20)] * len(data.df))
            pkl.dump(data, path.open("wb"))


if __name__ == "__main__":
    fire.Fire(main)
