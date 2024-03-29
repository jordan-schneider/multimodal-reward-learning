from pathlib import Path
from typing import List

import fire  # type: ignore
import numpy as np
from linear_procgen.feature_envs import FeatureEnv
from linear_procgen.util import ENV_NAMES as FEATURE_ENV_NAMES
from linear_procgen.util import make_env
from mpi4py import MPI  # type: ignore
from mrl.envs.rewards import make_reward_weights
from mrl.model_training.writer import SequentialWriter
from mrl.util import find_policy_path
from phasic_policy_gradient.train import train_fn
from torch.utils.tensorboard.writer import SummaryWriter


def setup_env_folder(
    env_dir: Path, env: FeatureEnv, n_rewards: int, overwrite: bool = False
):
    env_dir = Path(env_dir)
    env_dir.mkdir(parents=True, exist_ok=True)

    rewards = make_reward_weights(env, n_rewards)
    for i, reward in enumerate(rewards):
        reward_dir = env_dir / str(i + 1)
        reward_dir.mkdir(parents=True, exist_ok=True)
        reward_path = reward_dir / "reward.npy"
        if overwrite or not reward_path.exists():
            np.save(reward_path, reward)


def train(
    path: Path,
    env_name: FEATURE_ENV_NAMES,
    seed: int = 0,
    n_parallel_envs: int = 64,
    n_minibatch: int = 8,
    total_interacts: int = 100_000_000,
    replications: str = "1",
    overwrite: bool = False,
    port=29500,
) -> None:
    path = Path(path)
    path /= env_name
    path.mkdir(parents=True, exist_ok=True)

    comm = MPI.COMM_WORLD

    repls = parse_replications(replications)
    max_replication = max(repls)

    tmp_env = make_env(name=env_name, num=1, reward=1, extract_rgb=False)
    if comm.Get_rank() == 0:
        setup_env_folder(
            env_dir=path,
            env=tmp_env,
            n_rewards=max_replication,
            overwrite=overwrite,
        )
    comm.Barrier()  # Forcing other threads to wait for 0 thread to finish making rewards.

    for replication in repls:
        repl_path = path / str(replication)
        env = make_env(
            name=env_name,
            reward=np.load(repl_path / "reward.npy"),
            num=n_parallel_envs,
            rand_seed=seed + replication,
            log_writer=SequentialWriter(SummaryWriter(log_dir=repl_path / "logs/env")),
        )
        model_path, model_iter = find_policy_path(repl_path / "models", overwrite)

        start_time = model_iter * 100_000  # LogSaveHelper ic_per_save value

        model_save_dir = repl_path / "models"
        model_save_dir.mkdir(parents=True, exist_ok=True)

        train_fn(
            save_dir=model_save_dir,
            venv=env,
            n_minibatch=n_minibatch,
            model_path=model_path,
            start_time=start_time,
            arch="detach",
            interacts_total=total_interacts,
            n_epoch_vf=6,
            log_dir=repl_path / "logs/train",
            comm=comm,
            port=port,
        )


def parse_replications(replications: str) -> List[int]:
    if type(replications) == int:
        return [int(replications)]
    if "," in replications:
        return sum([parse_replications(g) for g in replications.split(",")], start=[])
    start, stop = replications.split("-")
    return list(range(int(start), int(stop) + 1))


if __name__ == "__main__":
    fire.Fire(train)
