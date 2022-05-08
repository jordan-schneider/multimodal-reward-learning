from pathlib import Path
from typing import cast

import fire  # type: ignore
import numpy as np
import torch
from gym3 import VideoRecorderWrapper  # type: ignore
from phasic_policy_gradient.roller import Roller
from phasic_policy_gradient.train import make_model
from PIL import Image  # type: ignore

from mrl.envs.util import ENV_NAMES, make_env
from mrl.util import find_best_gpu


def replay(
    env_name: ENV_NAMES,
    model_path: Path,
    n_videos: int,
    outdir: Path,
    horizon: int = 1000,
) -> None:
    env = make_env(name=env_name, reward=0, num=n_videos, render_mode="rgb_array")

    device = find_best_gpu()

    model = make_model(env, arch="shared").to(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    writer_kwargs = {
        "codec": "libx264rgb",
        "pixelformat": "bgr24",
        "output_params": ["-crf", "0"],
    }
    for i in range(n_videos):
        env = VideoRecorderWrapper(
            env,
            directory=outdir,
            env_index=i,
            writer_kwargs=writer_kwargs,
            info_key="rgb",
            prefix=str(i) + "_",
        )
    done = np.zeros((n_videos,), dtype=bool)

    roller = Roller(
        venv=env,
        act_fn=model.act,
        initial_state=model.initial_state(n_videos),
    )

    roller.single_step()

    t = 0
    while not np.all(done) and t < horizon:
        out = roller.single_step()
        t += 1
        firsts = out["first"]
        done = done | firsts.cpu().numpy()


def frames_to_gif(frames: np.ndarray, filename: Path) -> None:
    imgs = [Image.fromarray(frame) for frame in frames]
    imgs[0].save(filename, save_all=True, append_images=imgs[1:], duration=100, loop=0)


if __name__ == "__main__":
    fire.Fire(replay)
