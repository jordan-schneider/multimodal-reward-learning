import logging
from math import ceil
from pathlib import Path
from typing import Dict, Union

import fire  # type: ignore
import joypy  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from mrl.inference.results import Results
from mrl.util import setup_logging


def plot_comparisons(results: Results, outdir: Path) -> None:
    """Plot multiple comparison experiments"""
    if results.has("dispersion_gt"):
        logging.info("Plotting dispersion_gt")
        dispersion_gt = results.getall("dispersion_gt")
        sns.relplot(
            data=dispersion_gt, x="time", y="dispersion_gt", hue="modality", kind="line"
        ).savefig(outdir / "dispersion_gt.png")
        plt.close()
    else:
        logging.warning("Results did not have dispersion gt")

    if results.has("prob_aligned"):
        logging.info("Plotting prob_aligned")
        prob_aligned = results.getall("prob_aligned")
        sns.relplot(
            data=prob_aligned,
            x="time",
            y="prob_aligned",
            hue="modality",
            kind="line",
        ).savefig(outdir / "prob_aligned.png")
        plt.close()
    else:
        logging.warning("Results did not have prob aligned")

    if results.has("likelihood"):
        logging.info("Plotting likelihood")
        likelihoods_gt = results.getall_gt_likelihood()
        sns.relplot(
            data=likelihoods_gt,
            x="time",
            y="likelihood_gt",
            hue="modality",
            kind="line",
        ).savefig(outdir / "likelihood_gt.png")
        plt.close()
    else:
        logging.warning("Results did not have likelihood gt")

    if results.has("entropy"):
        logging.info("Plotting entropy")
        entropies = results.getall("entropy")
        sns.relplot(
            data=entropies, x="time", y="entropy", hue="modality", kind="line"
        ).savefig(outdir / "entropy.png")
    else:
        logging.warning("Results does not have entropies")


def plot_comparison(results: Results, use_gt: bool = False) -> None:
    """Plot single comparison experiment"""
    assert results.current_experiment is not None, "No current experiment"
    outdir = results.outdir / results.current_experiment
    if likelihoods := results.get("likelihood"):
        plot_liklihoods(likelihoods, outdir)

        if use_gt:
            plot_gt_likelihood(likelihoods, outdir)

    if entropies := results.get("entropy"):
        plot_entropies(entropies, outdir)
    if counts := results.get("count"):
        plot_counts(counts, outdir)

    if dispersion_mean := results.get("dispersion_mean"):
        plot_dispersions(dispersion_mean, outdir, outname="dispersion_mean")

    if centroid_per_modality := results.get("centroid_per_modality"):
        plot_rewards(rewards=centroid_per_modality, outdir=outdir, outname="centroids")

    if mean_rewards := results.get("mean_reward"):
        plot_rewards(rewards=mean_rewards, outdir=outdir, outname="mean_reward")

    if dispersion_centroid_per_modality := results.get("dispersion_centroid"):
        plot_dispersions(
            dispersion_centroid_per_modality, outdir, outname="dispersion_centroid"
        )

    if dispersions_gt := results.get("dispersion_gt"):
        assert use_gt
        plot_dispersions(dispersions_gt, outdir, outname="dispersion_gt")


def plot_gt_likelihood(
    likelihoods: Union[Dict[str, np.ndarray], np.ndarray], outdir: Path
) -> None:
    logging.info("Plotting likelihood")
    if isinstance(likelihoods, dict):
        for name, l in likelihoods.items():
            plt.plot(l[-1], label=name)
        plt.legend()
    else:
        plt.plot(likelihoods[-1])
    plt.title("Ground truth posterior likelihood")
    plt.xlabel("Human preferences")
    plt.ylabel("Likelihood of true reward")
    plt.savefig(outdir / "gt_likelihood.png")
    plt.close()


def plot_dispersions(
    dispersions: Union[Dict[str, np.ndarray], np.ndarray], outdir: Path, outname: str
) -> None:
    if isinstance(dispersions, dict):
        for name, d in dispersions.items():
            plt.plot(d, label=name)
        plt.legend()
    else:
        plt.plot(dispersions)
    plt.xlabel("Human preferences")
    plt.ylabel("Mean dispersion")
    plt.title("Concentration of posterior with data")
    plt.savefig(outdir / f"{outname}.png")
    plt.close()

    if isinstance(dispersions, dict):
        for name, d in dispersions.items():
            log_dispersion = np.log(d)
            log_dispersion[log_dispersion == -np.inf] = None
            plt.plot(log_dispersion, label=outname)
        plt.legend()
    else:
        log_dispersion = np.log(dispersions)
        log_dispersion[log_dispersion == -np.inf] = None
        plt.plot(log_dispersion)
    plt.xlabel("Human preferences")
    plt.ylabel("log(mean dispersion)")
    plt.title("Log-concentration of posterior with data")
    plt.savefig(outdir / f"log_{outname}.png")
    plt.close()


def plot_counts(
    counts: Union[Dict[str, np.ndarray], np.ndarray], outdir: Path, threshold: int = 200
) -> None:
    max_count = (
        max(np.max(c) for c in counts.values())
        if isinstance(counts, dict)
        else np.max(counts)
    )
    if isinstance(counts, dict):
        for name, c in counts.items():
            plt.plot(c, label=name)
        plt.legend()
    else:
        plt.plot(counts)
    plt.title("Number of rewards with nonzero likelihood")
    plt.xlabel("Number of preferences")
    plt.ylabel("Count")
    plt.ylim((0, max_count * 1.05))
    plt.savefig(outdir / "counts.png")
    plt.close()

    if isinstance(counts, dict):
        plot_small = any(np.any(c < threshold) for c in counts.values())
    else:
        plot_small = bool(np.any(counts < threshold))
    if plot_small:
        if isinstance(counts, dict):
            for name, c in counts.items():
                plt.plot(c[c < threshold], label=name)
            plt.legend()
        else:
            plt.plot(counts[counts < threshold])
        plt.title("Number of rewards with nonzero likelihood")
        plt.xlabel("Number of preferences")
        plt.ylabel("Count")
        plt.savefig(outdir / "small_counts.png")
        plt.close()


def plot_rewards(
    rewards: Union[Dict[str, np.ndarray], np.ndarray], outdir: Path, outname: str
) -> None:
    ndims = (
        list(rewards.values())[0].shape[1]
        if isinstance(rewards, dict)
        else rewards.shape[1]
    )
    for dim in range(ndims):
        if isinstance(rewards, dict):
            for name, r in rewards.items():
                plt.plot(r[:, dim], label=name)
            plt.legend()
        else:
            plt.plot(rewards[:, dim])
        plt.ylim(-1, 1)
        plt.xlabel("Preferences")
        plt.ylabel(f"{dim}-th dimension of reward")
        plt.title(f"{dim}-th dimension of reward")
        plt.savefig(outdir / f"{dim}.{outname}.png")
        plt.close()


def plot_liklihoods(
    likelihoods: Union[Dict[str, np.ndarray], np.ndarray], outdir: Path
) -> None:
    def plot(likelihoods: pd.DataFrame, outdir: Path, name: str) -> None:
        df = pd.DataFrame(likelihoods, dtype=np.float128)
        assert df.notnull().all().all()
        assert (df < np.inf).all().all()
        df = df.melt(
            value_vars=range(likelihoods.shape[1]),
            var_name="timestep",
            value_name="likelihood",
        )

        n_plots = min(10, likelihoods.shape[1])
        timesteps = np.arange(
            0, likelihoods.shape[1], ceil(likelihoods.shape[1] / n_plots)
        )
        assert len(timesteps) == n_plots, f"{len(timesteps)} != {n_plots}"

        df = df.loc[df["timestep"].isin(timesteps)]
        df = df.loc[df.timestep > 1e-3]

        small_df = df.loc[df.likelihood < 0.1]
        large_df = df.loc[df.likelihood >= 0.1]

        assert large_df.notnull().all().all()
        assert (large_df.abs() < np.inf).all().all()

        if len(small_df) > 0:
            fig, _ = joypy.joyplot(
                small_df, hist=True, by="timestep", overlap=0, bins=100
            )
            fig.savefig(outdir / f"{name}.small.png")
            plt.close(fig)

        if len(large_df) > 0:
            fig, _ = joypy.joyplot(
                large_df, hist=True, by="timestep", overlap=0, bins=100
            )
            fig.savefig(outdir / f"{name}.large.png")
            plt.close(fig)

    if isinstance(likelihoods, dict):
        for name, l in likelihoods.items():
            plot(l, outdir, name=f"likelihood_hist.{name}")
    else:
        plot(likelihoods, outdir, name="likelihood_hist")


def plot_entropies(entropies: Dict[str, np.ndarray], outdir: Path) -> None:
    for name, e in entropies.items():
        plt.plot(e, label=name)
    plt.legend()
    plt.title("Posterior entropy")
    plt.ylabel("Entropy")
    plt.xlabel("Human preferences")
    plt.savefig(outdir / "entropy.png")
    plt.close()


def post_hoc_plot_comparisons(outdir: Path) -> None:
    outdir = Path(outdir)
    setup_logging(level="INFO", outdir=outdir, name="post_hoc_plot_comparisons.log")
    results = Results(outdir=outdir / "trials", load_contents=True)
    plot_comparisons(results, outdir)


if __name__ == "__main__":
    fire.Fire(post_hoc_plot_comparisons)
