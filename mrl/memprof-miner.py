import fire
import numpy as np
from procgen.env import ProcgenGym3Env

from mrl.envs import Miner


def run_miner(timesteps: int) -> None:
    env = Miner(np.zeros(5), 1)

    # Disabling in_danger() gets rid of the leak.

    for t in range(timesteps):
        env.act(np.array([5]))
        env.observe()
        env.get_info()


def make_miners(n_miners: int, timesteps: int) -> None:
    # This causes a leak. Thus the ProcgenGym3Env isn't destructing correctly.
    for _ in range(n_miners):
        env = ProcgenGym3Env(1, "miner")
        for t in range(timesteps):
            env.act(np.array([5]))
            env.observe()
            env.get_info()


if __name__ == "__main__":
    fire.Fire({"run_miner": run_miner, "make_miners": make_miners})
