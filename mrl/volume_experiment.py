from pathlib import Path
from typing import Optional

import fire  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from numpy.testing import assert_allclose
from tqdm import tqdm  # type: ignore


def make_sphere_cover(
    n_samples: int, rng: np.random.Generator, dims: int = 5
) -> np.ndarray:
    samples = rng.standard_normal(size=(n_samples, dims))
    samples = (samples.T / np.linalg.norm(samples, axis=1)).T
    assert samples.shape == (n_samples, dims)
    assert_allclose(np.linalg.norm(samples, axis=1), 1)
    return samples


def find_volumes(
    diffs: np.ndarray, reward_samples: int, seed: Optional[int], outdir: Path
) -> None:
    rng = np.random.default_rng(seed)
    rewards = make_sphere_cover(reward_samples, rng)

    internal_rewards = np.ones(len(rewards), dtype=bool)
    volumes = np.empty(len(diffs))
    for i in tqdm(range(len(diffs))):
        internal_rewards &= rewards @ diffs[i] > 0
        volumes[i] = np.mean(internal_rewards)

    print(f"Final volume is {volumes[-1]}")

    final_index = np.argmin(volumes)

    print(f"Last meaningful preference at {final_index} with volume {volumes[-1]}")

    plt.plot(volumes[: int(1.1 * final_index)])
    plt.vlines(final_index, 0, volumes[0], linestyles="dashed")
    plt.xlabel("Human Samples")
    plt.ylabel("Fraction of rewards remaining")
    plt.title("Convergence of ARS")
    plt.savefig(outdir / "volume_experiment.png")


def collect_diffs(path: Path) -> np.ndarray:
    diffs = []
    for f in path.parent.glob(f"{path.name}.[0-9]*.npy"):
        diffs.append(np.load(f))

    return np.concatenate(diffs)


def volume(
    data_name: Path,
    outdir: Path,
    n_prefs: int = -1,
    reward_samples: int = 10_000,
    seed: Optional[int] = None,
    reward_path: Optional[Path] = None,
) -> None:
    data_name = Path(data_name)
    diffs = collect_diffs(data_name)[:n_prefs]

    if reward_path is not None:
        reward = np.load(reward_path)
        inside = np.all(reward.T @ diffs.T > 0)
        if inside:
            print("True reward in volume")
        else:
            print("True reward outside volume")

    find_volumes(diffs, reward_samples, seed, Path(outdir))


if __name__ == "__main__":
    fire.Fire(volume)
