# How distinct are the normal vectors from different modalities?

from pathlib import Path
from typing import Dict, List, Optional

import fire  # type: ignore
import numpy as np
from matplotlib import pyplot as plt  # type: ignore


def overlap(
    outdir: Path,
    state_path: Optional[Path] = None,
    traj_path: Optional[Path] = None,
    n_normals: int = 1000,
    min_ball_size: float = 1e-5,
    max_ball_size: float = 1,
    normalize: bool = False,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    normals: Dict[str, np.ndarray] = {}
    if state_path is not None:
        state_path = Path(state_path)
        normals["state"] = np.load(state_path)
    if traj_path is not None:
        traj_path = Path(traj_path)
        normals["traj"] = np.load(traj_path)

    if normalize:
        for name, data in normals.items():
            normals[name] = (data.T / np.linalg.norm(data, axis=1)).T

    # Find the fraction of normals in each set that are not within ball_size of any normal in any other set
    ball_sizes = np.logspace(
        np.log10(min_ball_size),
        np.log10(max_ball_size),
        num=int(np.log10(max_ball_size) - np.log10(min_ball_size)) + 1,
        base=10,
    )

    unique_per_size = [find_unique(normals, ball_size) for ball_size in ball_sizes]
    uniques_by_name = {
        name: np.array([unique_per_size[i][name] for i in range(len(unique_per_size))])
        for name in normals.keys()
    }

    for name, uniques in uniques_by_name.items():
        plt.plot(ball_sizes, uniques, label=name)
    plt.xscale("log")
    plt.title("How many normals aren't covered by any other method?")
    plt.xlabel("Distance to be considered the same")
    plt.ylabel("Number of unqiue normals")
    plt.legend()
    plt.savefig(outdir / "overlap.png")


def find_unique(
    normal_groups: Dict[str, np.ndarray], ball_size: float
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name, normals in normal_groups.items():
        n_unique = 0
        for other_name, other_normals in normal_groups.items():
            if name == other_name:
                continue
            dists = np.array(
                [
                    np.min(np.linalg.norm(normal - other_normals, axis=1))
                    for normal in normals
                ]
            )
            n_unique += np.sum(dists > ball_size)
        out[name] = n_unique
    return out


if __name__ == "__main__":
    fire.Fire(overlap)
