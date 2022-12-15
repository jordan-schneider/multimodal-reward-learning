import logging
from math import ceil
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Tuple, Union

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


def plot_comparisons(
    results: Results,
    outdir: Path,
    fast_error_bars: bool = False,
) -> None:
    """Plot multiple comparison experiments. Tries to plot dispersion_gt, prob_aligned, gt_likelihood, entropy, and mean_reward."""

    errorbar = "ci" if not fast_error_bars else ("se", 2)
    if results.any_has("dispersion_gt"):
        logging.info("Plotting dispersion_gt")
        dispersion_gt = results.getall("dispersion_gt")
        sns.relplot(
            data=dispersion_gt,
            x="time",
            y="dispersion_gt",
            hue="modality",
            kind="line",
            errorbar=errorbar,
        ).savefig(outdir / "dispersion_gt.png")
        plt.close()
    else:
        logging.warning("Results did not have dispersion gt")

    if results.any_has("prob_aligned"):
        logging.info("Plotting prob_aligned")
        prob_aligned = results.getall("prob_aligned")
        sns.relplot(
            data=prob_aligned,
            x="time",
            y="prob_aligned",
            hue="modality",
            kind="line",
            errorbar=errorbar,
        ).savefig(outdir / "prob_aligned.png")
        plt.close()
    else:
        logging.warning("Results did not have prob aligned")

    if results.any_has("gt_likelihood"):
        logging.info("Plotting likelihood")
        likelihoods_gt = results.getall("gt_likelihood")
        sns.relplot(
            data=likelihoods_gt,
            x="time",
            y="gt_likelihood",
            hue="modality",
            kind="line",
            errorbar=errorbar,
        ).savefig(outdir / "gt_likelihood.png")
        plt.close()
    else:
        logging.warning("Results did not have likelihood gt")

    if results.any_has("entropy"):
        logging.info("Plotting entropy")
        logging.debug("Getting all entropies")
        entropies = results.getall("entropy")
        logging.debug("Plotting entropies")
        sns.relplot(
            data=entropies,
            x="time",
            y="entropy",
            hue="modality",
            kind="line",
            errorbar=errorbar,
        ).savefig(outdir / "entropy.png")
        plt.close()
        logging.debug("Done plotting entropies")
    else:
        logging.warning("Results does not have entropies")

    if results.any_has("mean_reward"):
        logging.info("Plotting mean reward")
        plot_multiple_rewards(
            rewards=results.getall("mean_reward"),
            outdir=outdir,
            outname="mean_reward",
            errorbar=errorbar,
        )
    else:
        logging.warning("Results does not have mean rewards")


def plot_comparison(results: Results, outdir: Path, use_gt: bool = False) -> None:
    """Plot single comparison experiment. Tries to plot liklihood, gt_likelihood, entropy, non-zero reward sample count, mean reward, dispersion from mean, centroid reward, dispersion from centroid, and dispersion from gt."""
    assert results.current_experiment_name is not None, "No current experiment"

    if (likelihoods := results.get("likelihood")) is not None:
        plot_likelihoods(likelihoods, outdir)

        if use_gt:
            plot_gt_likelihood(likelihoods, outdir)

    if entropies := results.get("entropy"):
        plot_entropies(entropies, outdir)
    if counts := results.get("count"):
        plot_counts(counts, outdir)

    if mean_rewards := results.get("mean_reward"):
        plot_rewards(rewards=mean_rewards, outdir=outdir, outname="mean_reward")

    if dispersion_mean := results.get("dispersion_mean"):
        plot_dispersions(dispersion_mean, outdir, outname="dispersion_mean")

    if centroid_per_modality := results.get("centroid_per_modality"):
        plot_rewards(rewards=centroid_per_modality, outdir=outdir, outname="centroids")

    if dispersion_centroid_per_modality := results.get("dispersion_centroid"):
        plot_dispersions(
            dispersion_centroid_per_modality, outdir, outname="dispersion_centroid"
        )

    if dispersions_gt := results.get("dispersion_gt"):
        assert use_gt
        plot_dispersions(dispersions_gt, outdir, outname="dispersion_gt")


def plot_multiple_rewards(
    rewards: pd.DataFrame,
    outdir: Path,
    outname: str,
    errorbar: Union[str, Tuple[str, int]],
) -> None:
    """Plot mean reward for multiple experiments. Each dimension of reward is plotted separately."""
    ndims = len([col for col in rewards.columns if "mean_reward" in col])
    for dim in range(ndims):
        logging.debug(f"Plotting reward dim {dim}")
        reward_dim_name = f"mean_reward_{dim}"
        assert reward_dim_name in rewards.columns
        sns.relplot(
            data=rewards,
            x="time",
            y=reward_dim_name,
            hue="modality",
            kind="line",
            errorbar=errorbar,
        ).savefig(outdir / f"{outname}_{dim}.png")
        plt.close()


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


def plot_rewards(
    rewards: Dict[str, np.ndarray],
    outdir: Optional[Path] = None,
    outname: Optional[str] = None,
    show: bool = False,
) -> None:
    ndims = list(rewards.values())[0].shape[1]
    for dim in range(ndims):
        nth_dim = {name: r[:, dim] for name, r in rewards.items()}
        plot_dict(
            data=nth_dim,
            title=f"{dim}-th dimension of reward",
            xlabel="Preferences",
            ylabel=f"{dim}-th dimension of reward",
            outpath=outdir / f"{dim}.{outname}.png"
            if outdir is not None and outname is not None
            else None,
            show=show,
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
    fig = plt.figure(figsize=(15, 5))
    axes = fig.subplots()
    axes.set_prop_cycle(color=plt.cm.tab20.colors)
    lines = []
    for name, d in data.items():
        lines.append(axes.plot(d, label=name)[0])
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    legend = axes.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if customization is not None:
        customization(fig, axes)
    if outpath is not None:
        fig.savefig(outpath, bbox_inches="tight")
    if show:
        legend_to_line = {k: v for k, v in zip(legend.get_lines(), lines)}
        for legend_entry in legend.get_lines():
            legend_entry.set_picker(True)

        def on_pick(event):
            legline = event.artist
            origline = legend_to_line[legline]
            visible = not origline.get_visible()
            origline.set_visible(visible)

            legline.set_alpha(1.0 if visible else 0.2)
            fig.canvas.draw()

        fig.canvas.mpl_connect("pick_event", on_pick)
        fig.show()


def post_hoc_plot_comparisons(
    datadir: Path,
    experiment: Optional[str] = None,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
    fast_error_bars: bool = False,
) -> None:
    """Plots experimental results for a previously run experiment.

    Args:
        datadir (Path): Location of the directory containing results. trials should be a subfolder. All plots will be saved in the plots subfolder, or a per-trial plots subfolder.
        experiment (Optional[str]): The name of a single trial to run. If None, plots aggregate results for all trials. Defualts to None.
        verbosity (Literal["INFO", "DEBUG"]): Logging level. Defaults to "INFO".
    """
    datadir = Path(datadir)
    setup_logging(level=verbosity, outdir=datadir, name="post_hoc_plot_comparisons.log")
    try:
        results = Results(outdir=datadir / "trials")
        if experiment is None:
            (datadir / "plots").mkdir(exist_ok=True)
            plot_comparisons(results, datadir / "plots", fast_error_bars)
        else:
            results.start(experiment)
            plotdir = datadir / "trials" / experiment / "plots"
            plotdir.mkdir(parents=True, exist_ok=True)
            plot_comparison(results, plotdir)
    except BaseException as e:
        logging.exception(e)
        raise e


if __name__ == "__main__":
    fire.Fire(post_hoc_plot_comparisons)
