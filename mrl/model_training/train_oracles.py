from pathlib import Path

import fire  # type: ignore
import gym3  # type: ignore
import numpy as np
from mpi4py import MPI  # type: ignore
from phasic_policy_gradient.train import train_fn

from mrl.envs.util import ENV_NAMES, make_env
from mrl.rewards import load_reward, make_rewards
from mrl.util import find_policy_path


def train(
    path: Path,
    env_name: ENV_NAMES,
    seed: int = 0,
    n_parallel_envs: int = 64,
    n_minibatch: int = 8,
    total_interacts: int = 100_000_000,
    fix_reward_sign: bool = False,
    use_original_reward: bool = False,
    use_near_original_reward: bool = False,
    replications: int = 1,
    overwrite: bool = False,
    port=29500,
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    comm = MPI.COMM_WORLD

    rng = np.random.default_rng(seed)
    make_rewards(
        path=path,
        env=env_name,
        rng=rng,
        comm=comm,
        overwrite=overwrite,
        fix_reward_sign=fix_reward_sign,
        use_original=use_original_reward,
        use_near_original=use_near_original_reward,
        replications=replications,
    )

    for replication in range(replications):
        repl_path = path / str(replication)
        env = make_env(
            kind=env_name,
            reward=load_reward(path=repl_path, comm=comm),
            num=n_parallel_envs,
            rand_seed=seed + replication,
        )
        env = gym3.ExtractDictObWrapper(env, "rgb")

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
            log_dir=repl_path / "logs",
            comm=comm,
            port=port,
        )


if __name__ == "__main__":
    fire.Fire({"train": train})
