import os
from pathlib import Path

import fire  # type: ignore
import numpy as np

# Some data was concatenated instead of stacked in a way that flattened the array. This script tries
# to find all of those files and fixes their shape


def main(rootdir: Path, name: str) -> None:
    rootdir = Path(rootdir)
    for file in rootdir.rglob(f"{name}.*[0-9]*.npy"):
        arr = np.load(file)
        if len(arr.shape) != 2:
            print(file)
            print(arr.shape)
            arr = arr.reshape((-1, 5))
            np.save(file, arr)


if __name__ == "__main__":
    fire.Fire(main)
