from pathlib import Path

import fire  # type: ignore
import numpy as np


def main(ars_path: Path, reward_path: Path) -> None:
    ars = np.load(ars_path)
    reward = np.load(reward_path)

    assert np.all(ars @ reward >= 0)


if __name__ == "__main__":
    fire.Fire(main)
