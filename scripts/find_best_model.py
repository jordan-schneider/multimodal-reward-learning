import pickle as pkl
import re
from pathlib import Path
from typing import Dict

import fire  # type: ignore
import numpy as np
import torch
from gym3 import ExtractDictObWrapper  # type: ignore
from mrl.envs import Miner
from mrl.util import procgen_rollout_features


def main(rootdir: Path, out: Path, horizon: int = 10_000) -> None:
    env = Miner(np.ones(5), 1)
    env = ExtractDictObWrapper(env, "rgb")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    finished_trajs: Dict[Path, int] = {}
    if out.exists():
        finished_trajs = pkl.load(out.open("rb"))

    rootdir = Path(rootdir)
    for model_path in rootdir.rglob("*"):
        # print(model_path)
        if model_path in finished_trajs.keys():
            continue
        if re.search("model(9[0-9][059]|1000)\.jd", str(model_path)) is None:
            continue
        print(f"Loading model from {model_path}")
        device = torch.device("cuda:0")
        policy = torch.load(model_path, map_location=device)
        policy.device = device

        data = procgen_rollout_features(env, policy, timesteps=horizon, tqdm=True)
        finished_trajs[model_path] = np.sum(data[:1] != 0)

        pkl.dump(finished_trajs, out.open("wb"))

    best_path = max(finished_trajs, key=finished_trajs.get)
    finishes = finished_trajs[best_path]
    print(
        f"Best model with {finishes}/{horizon} {finishes / horizon * 100:0.2f}% at {best_path}"
    )


if __name__ == "__main__":
    fire.Fire(main)
