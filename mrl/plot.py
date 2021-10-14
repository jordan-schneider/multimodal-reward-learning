from pathlib import Path

import fire  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore


def plot_progress(rootdir: Path) -> None:
    rootdir = Path(rootdir)
    progress = pd.read_csv(rootdir / "progress.csv")
    for col in progress.columns:
        progress[col].plot()
        plot_path = rootdir / "plots" / f"{col}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.title(col)
        plt.savefig(plot_path)
        plt.close()


if __name__ == "__main__":
    fire.Fire({"progress": plot_progress})
