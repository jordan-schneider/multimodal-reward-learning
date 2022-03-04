import fire
import numpy as np
from mrl.envs.util import get_root_env, make_env


def main() -> None:
    env = make_env(name="miner", num=1, reward=1)
    root_env = get_root_env(env)
    data = []

    for i in range(1001):
        reward, state, first = env.observe()

        action = np.random.random_integers(low=0, high=env.ac_space.eltype.n, size=(1,))

        env.act(action)

        data.append((reward, first, np.copy(root_env.features), action))

    for step in data:
        print(step)


if __name__ == "__main__":
    fire.Fire(main)
