import pickle as pkl
from pathlib import Path

import fire
import numpy as np
from linear_procgen import make_env
from linear_procgen.util import get_root_env
from mrl.dataset.trajectory_db import FeatureDataset


def rebuild_env(name: str):
    env = make_env(name=name, num=1, reward=0)
    root_env = get_root_env(env)
    # Initial action needed to make sure whole env is loaded for get_state() even if set_state called right after
    env.act(np.array([0]))
    return env, root_env


def main(traj_dir: Path) -> None:
    paths = list(Path(traj_dir).glob("trajectories_*.pkl"))
    print(f"Found {paths}")
    env, root_env = rebuild_env("miner")
    reset = False
    for path in paths:
        print(f"Checking {path}")
        dataset: FeatureDataset = pkl.load(path.open("rb"))
        for index, row in dataset.df.iterrows():
            actions = row["actions"]
            cstates = row["cstates"]
            grids = row["grids"]
            for i in range(len(actions)):
                if reset:
                    reset = False
                    env, root_env = rebuild_env("miner")
                root_env.set_state([cstates[i]])
                actual_grid = env.get_info()[0]["grid"]
                expected_grid = grids[i]
                if not np.array_equal(actual_grid, expected_grid):
                    print(
                        f"Reconstructed grids not equal to recording {index} at t={i}"
                    )
                    import pdb

                    pdb.set_trace()

                # Loading into env right after the agent died causes problems.
                if np.any(actual_grid == 12):
                    reset = True

                if i + 1 < len(grids):
                    env.act(np.array([actions[i]]))
                    actual_grid = env.get_info()[0]["grid"]
                    expected_grid = grids[i + 1]
                    if not np.array_equal(actual_grid, expected_grid):
                        print(f"Step does not match recording {index} at t={i + 1}")
                        import pdb

                        pdb.set_trace()

                if np.any(actual_grid == 12):
                    reset = True


if __name__ == "__main__":
    fire.Fire(main)
