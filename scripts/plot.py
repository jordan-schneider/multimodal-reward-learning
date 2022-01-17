from pathlib import Path

import fire  # type: ignore
import seaborn as sns  # type: ignore
from mrl.inference.posterior import Results


def prob_aligned(rootdir: Path, max_comparisons: int = 100) -> None:
    rootdir = Path(rootdir)
    results = Results(rootdir, load_contents=True)
    prob_aligned = results.getall("prob_aligned")
    prob_aligned = prob_aligned[prob_aligned["time"] < max_comparisons]
    sns.relplot(
        data=prob_aligned,
        x="time",
        y="prob_aligned",
        hue="modality",
        kind="line",
    ).savefig(rootdir / f"prob_aligned.first_{max_comparisons}.png")


if __name__ == "__main__":
    fire.Fire({"prob-aligned": prob_aligned})
