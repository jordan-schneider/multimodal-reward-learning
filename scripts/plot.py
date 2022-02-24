from pathlib import Path

import fire  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore
from mrl.inference.posterior import Results


def prob_aligned(rootdir: Path, max_comparisons: int = 100) -> None:
    rootdir = Path(rootdir)
    results = Results(rootdir / "trials", load_contents=True)
    prob_aligned = results.getall("prob_aligned")
    prob_aligned = prob_aligned[prob_aligned["time"] < max_comparisons]
    sns.relplot(
        data=prob_aligned,
        x="time",
        y="prob_aligned",
        hue="modality",
        kind="line",
    ).savefig(rootdir / f"prob_aligned.first_{max_comparisons}.png")


def plot_trajs(rootdir: Path, max_comparisons: int = 1000) -> None:
    rootdir = Path(rootdir)
    results = Results(rootdir / "trials", load_contents=True)
    prob_aligned = results.getall("prob_aligned")
    prob_aligned = prob_aligned[prob_aligned["modality"] == "traj"]
    prob_aligned = prob_aligned[prob_aligned["time"] < max_comparisons]
    sns.relplot(
        data=prob_aligned,
        x="time",
        y="prob_aligned",
        hue="modality",
        kind="line",
    ).savefig(rootdir / f"prob_aligned.trajs.first_{max_comparisons}.png")


def flip_prob(flip_probs_path: Path, nbins: int = 100) -> None:
    flip_probs_path = Path(flip_probs_path)
    flip_probs = np.load(flip_probs_path)

    temp = float(flip_probs_path.parts[-2])

    plt.hist(flip_probs, bins=nbins)
    plt.xlim((0, 1))
    plt.title(f"Flip probabilities for temp={temp:0.5f}")
    plt.savefig(flip_probs_path.with_suffix(".png"))


if __name__ == "__main__":
    fire.Fire(
        {"prob-aligned": prob_aligned, "flip-prob": flip_prob, "trajs": plot_trajs}
    )
