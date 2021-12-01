import os
import re
from pathlib import Path

import fire  # type: ignore
import numpy as np

# Some data was concatenated instead of stacked in a way that flattened the array. This script tries
# to find all of those files and fixes their shape


def main(rootdir: Path, name: str) -> None:
    rootdir = Path(rootdir)
    for file in rootdir.rglob(f"{name}*"):
        if re.search(f"{name}(.[0-9]+)?.npy", str(file)) is None:
            continue
        arr = np.load(file)

        if len(arr.shape) != 2:
            print(file)
            print(arr.shape)
            arr = arr.reshape((-1, 5))
            np.save(file, arr)

        if np.any(np.all(arr == 0, axis=1)):
            arr = arr[np.any(arr != 0, axis=1)]
            np.save(file, arr)


if __name__ == "__main__":
    fire.Fire(main)
