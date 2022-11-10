import logging
from math import ceil
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import fire  # type: ignore
import joypy  # type: ignore
import matplotlib  # type: ignore
import matplotlib.axes  # type: ignore
import matplotlib.figure  # type: ignore
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

    if results.has("gt_likelihood"):
        logging.info("Plotting likelihood")
        likelihoods_gt = results.getall("gt_likelihood")
        sns.relplot(
            data=likelihoods_gt,
            x="time",
            y="gt_likelihood",
            hue="modality",
            kind="line",
        ).savefig(outdir / "gt_likelihood.png")
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


def plot_comparison(results: Results, outdir: Path, use_gt: bool = False) -> None:
    """Plot single comparison experiment"""
    assert results.current_experiment is not None, "No current experiment"

    if (likelihoods := results.get("likelihood")) is not None:
        plot_likelihoods(likelihoods, outdir)

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


def plot_gt_likelihood(likelihoods: Dict[str, np.ndarray], outdir: Path) -> None:
    logging.info("Plotting likelihood")
    gt_likelhood = {name: l[-1] for name, l in likelihoods.items()}
    plot_dict(
        data=gt_likelhood,
        title="Ground truth posterior likelihood",
        xlabel="Human preferences",
        ylabel="Likelihood of true reward",
        outpath=outdir / "gt_likelihood.png",
    )


def plot_dispersions(
    dispersions: Dict[str, np.ndarray], outdir: Path, outname: str
) -> None:
    plot_dict(
        data=dispersions,
        title="Concentration of posterior with data",
        xlabel="Human preferences",
        ylabel="Mean dispersion",
        outpath=outdir / f"{outname}.png",
    )

    def _safe_log(arr: np.ndarray) -> np.ndarray:
        log = np.log(arr)
        log[log == -np.inf] = None
        return log

    log_dispersions = {k: _safe_log(v) for k, v in dispersions.items()}
    plot_dict(
        data=log_dispersions,
        title="Log-concentration of posterior with data",
        xlabel="Human preferences",
        ylabel="log(mean dispersion)",
        outpath=outdir / f"log_{outname}.png",
    )


def plot_counts(
    counts: Dict[str, np.ndarray], outdir: Path, threshold: int = 200
) -> None:
    max_count = max(np.max(c) for c in counts.values())
    if isinstance(counts, dict):
        plot_dict(
            data=counts,
            title="Number of rewards with nonzero likelihood",
            xlabel="Number of preferences",
            ylabel="Count",
            outpath=outdir / "counts.png",
            customization=lambda fig, ax: ax.set_ylim((0, max_count * 1.05)),
        )

    if any(np.any(c < threshold) for c in counts.values()):
        small_counts = {
            name: c[c < threshold]
            for name, c in counts.items()
            if np.any(c < threshold)
        }
        if isinstance(counts, dict):
            plot_dict(
                data=small_counts,
                title="Number of rewards with nonzero likelihood",
                xlabel="Number of preferences",
                ylabel="Count",
                outpath=outdir / "small_counts.png",
                customization=lambda fig, ax: ax.set_ylim((0, max_count * 1.05)),
            )


def plot_rewards(rewards: Dict[str, np.ndarray], outdir: Path, outname: str) -> None:
    ndims = list(rewards.values())[0].shape[1]
    for dim in range(ndims):
        nth_dim = {name: r[:, dim] for name, r in rewards.items()}
        plot_dict(
            data=nth_dim,
            title=f"{dim}-th dimension of reward",
            xlabel="Preferences",
            ylabel=f"{dim}-th dimension of reward",
            outpath=outdir / f"{dim}.{outname}.png",
            customization=lambda fig, ax: ax.set_ylim((-1, 1)),
        )


def _plot_likelihood(likelihoods: np.ndarray, outdir: Path, name: str) -> None:
    df = pd.DataFrame(likelihoods, dtype=np.float128)
    assert df.notnull().all().all()
    assert (df < np.inf).all().all()
    df = df.melt(
        value_vars=range(likelihoods.shape[1]),
        var_name="timestep",
        value_name="likelihood",
    )

    n_plots = min(10, likelihoods.shape[1])
    timesteps = np.arange(0, likelihoods.shape[1], ceil(likelihoods.shape[1] / n_plots))
    assert len(timesteps) == n_plots, f"{len(timesteps)} != {n_plots}"

    df = df.loc[df["timestep"].isin(timesteps)]
    df = df.loc[df.timestep > 1e-3]

    small_df = df.loc[df.likelihood < 0.1]
    large_df = df.loc[df.likelihood >= 0.1]

    assert large_df.notnull().all().all()
    assert (large_df.abs() < np.inf).all().all()

    if len(small_df) > 0:
        fig, axes = joypy.joyplot(
            small_df, hist=True, by="timestep", overlap=0, bins=100
        )
        axes[-1].set_xlabel("Likelihood")
        axes[-1].yaxis.set_label_coords(-0.07, 0.5)
        axes[-1].set_ylabel("# Preferences")
        axes[-1].yaxis.set_visible(True)
        axes[-1].yaxis.set_ticks([])
        fig.savefig(outdir / f"{name}.small.png")
        plt.close(fig)

    if len(large_df) > 0:
        fig, axes = joypy.joyplot(
            large_df, hist=True, by="timestep", overlap=0, bins=100
        )
        axes[-1].set_xlabel("Likelihood")
        axes[-1].yaxis.set_label_coords(-0.07, 0.5)
        axes[-1].set_ylabel("# Preferences")
        axes[-1].yaxis.set_visible(True)
        axes[-1].yaxis.set_ticks([])
        fig.savefig(outdir / f"{name}.large.png")
        plt.close(fig)

    fig, axes = joypy.joyplot(df, hist=True, by="timestep", overlap=0, bins=100)
    axes[-1].set_xlabel("Likelihood")
    axes[-1].yaxis.set_label_coords(-0.07, 0.5)
    axes[-1].set_ylabel("# Preferences")
    axes[-1].yaxis.set_visible(True)
    axes[-1].yaxis.set_ticks([])
    fig.savefig(outdir / f"{name}.png")
    plt.close(fig)


def plot_likelihoods(
    likelihoods: Union[Dict[str, np.ndarray], np.ndarray], outdir: Path
) -> None:
    if isinstance(likelihoods, dict):
        for name, l in likelihoods.items():
            _plot_likelihood(l, outdir, name=f"likelihood_hist.{name}")
    else:
        _plot_likelihood(likelihoods, outdir, name="likelihood_hist")


def plot_entropies(entropies: Dict[str, np.ndarray], outdir: Path) -> None:
    plot_dict(
        data=entropies,
        title="Posterior entropy",
        xlabel="Human preferences",
        ylabel="Entropy",
        outpath=outdir / "entropy.png",
    )


def plot_dict(
    data: Dict[str, np.ndarray],
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Optional[Path] = None,
    show: bool = False,
    customization: Optional[
        Callable[[matplotlib.figure.Figure, matplotlib.axes.Axes], None]
    ] = None,
) -> None:
    fig = plt.figure()
    axes = fig.subplots()
    for name, d in data.items():
        axes.plot(d, label=name)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if customization is not None:
        customization(fig, axes)
    if outpath is not None:
        fig.savefig(outpath, bbox_inches="tight")
    if show:
        fig.show()


def post_hoc_plot_comparisons(outdir: Path, experiment: Optional[str]) -> None:
    outdir = Path(outdir)
    setup_logging(level="INFO", outdir=outdir, name="post_hoc_plot_comparisons.log")
    try:
        results = Results(outdir=outdir / "trials", load_contents=True)
        if experiment is None:
            plot_comparisons(results, outdir)
        else:
            results.start(experiment)
            plotdir = outdir / "trials" / experiment / "plots"
            plotdir.mkdir(parents=True, exist_ok=True)
            plot_comparison(results, plotdir)
    except BaseException as e:
        logging.exception(e)
        raise e


if __name__ == "__main__":
    fire.Fire(post_hoc_plot_comparisons)
