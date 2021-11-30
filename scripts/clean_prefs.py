import os
from pathlib import Path

import fire  # type: ignore
import numpy as np


def main(rootdir: Path, name: str) -> None:
    rootdir = Path(rootdir)
    for file in rootdir.rglob(f"{name}.[0-9]*.npy"):
        arr = np.load(file)
        if len(arr.shape) != 2:
            os.remove(file)


if __name__ == "__main__":
    fire.Fire(main)
