import pickle as pkl
import re
from pathlib import Path
from typing import Dict, Tuple

import fire  # type: ignore
import numpy as np
import torch
from linear_procgen.util import ENV_NAMES as FEATURE_ENV_NAMES
from linear_procgen.util import make_env
from mrl.dataset.roller import procgen_rollout_dataset
from mrl.util import find_best_gpu
from phasic_policy_gradient.train import make_model


def main(
    rootdir: Path,
    out: Path,
    env_name: FEATURE_ENV_NAMES,
    horizon: int = 10_000,
    seed: int = 0,
    overwrite: bool = False,
    use_only_last: bool = True,
    print_all: bool = False,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env(name=env_name, num=1, reward=1)

    policy = make_model(env, arch="shared")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    finished_trajs: Dict[Path, Tuple[int, int]] = {}
    if out.exists():
        finished_trajs = pkl.load(out.open("rb"))

    rootdir = Path(rootdir)
    model_paths = rootdir.rglob("*") if rootdir.is_dir() else [rootdir]
    for model_path in model_paths:
        if not overwrite and model_path in finished_trajs.keys():
            print(f"Skipping existing model {model_path}")
            continue
        if (
            use_only_last
            and re.search("model(9[0-9][059]|1000)\.jd$", str(model_path)) is None
        ):
            # print(f"Skipping non model file {model_path}")
            continue
        print(f"Loading model from {model_path}")
        device = find_best_gpu()
        policy.load_state_dict(torch.load(model_path, map_location=device))
        policy = policy.to(device=device)

        data = procgen_rollout_dataset(
            env, policy, timesteps=horizon, flags=["first"], tqdm=True
        )
        assert data.data["firsts"] is not None
        finished_trajs[model_path] = (np.sum(data.data["firsts"]), horizon)
        # TODO: Check this for bugs after feature change.
        if print_all:
            print(f"{model_path} finished {finished_trajs[model_path][0]}/{horizon}")

        pkl.dump(finished_trajs, out.open("wb"))

    best_path, finishes, _ = get_best(finished_trajs)
    print(
        f"Best model with {finishes}/{horizon} {finishes / horizon * 100:0.2f}% at {best_path}"
    )


def get_best(finished_trajs: Dict[Path, Tuple[int, int]]) -> Tuple[Path, int, int]:
    best_path = max(
        finished_trajs,
        key=lambda p: finished_trajs[p][0] / finished_trajs[p][1],
    )
    finishes, horizon = finished_trajs[best_path]
    return best_path, finishes, horizon


def print_current_best(finishes_path: Path) -> None:
    finished_trajs = pkl.load(open(finishes_path, "rb"))
    best_path, finishes, horizon = get_best(finished_trajs)
    print(
        f"Best model with {finishes}/{horizon} {finishes / horizon * 100:0.2f}% at {best_path}"
    )


def change_horizon(path: Path, horizon: int) -> None:
    finished_trajs = pkl.load(open(path, "rb"))
    for key, value in finished_trajs.items():
        if isinstance(value, tuple):
            value = value[0]
        finished_trajs[key] = (value, horizon)
    pkl.dump(finished_trajs, open(path, "wb"))


if __name__ == "__main__":
    fire.Fire(
        {"find": main, "print": print_current_best, "fix-horizon": change_horizon}
    )
