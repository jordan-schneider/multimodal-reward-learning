import pdb
import pickle as pkl
import sqlite3
from pathlib import Path
from typing import Tuple

import fire  # type: ignore
import numpy as np
from linear_procgen import ENV_NAMES, make_env
from linear_procgen.feature_envs import FeatureEnv
from linear_procgen.util import get_root_env
from mrl.dataset.trajectory_db import FeatureDataset
from procgen import ProcgenGym3Env


def rebuild_env(name: ENV_NAMES) -> Tuple[FeatureEnv, ProcgenGym3Env]:
    env = make_env(name=name, num=1, reward=0)
    root_env = get_root_env(env)
    # Initial action needed to make sure whole env is loaded for get_state() even if set_state called right after
    env.act(np.array([0]))
    return env, root_env


def format_grid(grid: np.ndarray) -> np.ndarray:
    return grid[:400].reshape((20, 20))


def check_traj(
    actions: np.ndarray,
    cstates: np.ndarray,
    grids: np.ndarray,
    agent_pos: np.ndarray,
    exit_pos: np.ndarray,
    use_pdb: bool = False,
) -> None:
    env, root_env = rebuild_env("miner")

    # If the agent dies, we need to reset the env by reinitialzing the environment becuase the died variable isn't
    # set by set_state.
    reset = False
    for i in range(len(grids)):
        if reset:
            reset = False
            env, root_env = rebuild_env("miner")

        is_last = i + 1 == len(grids)

        root_env.set_state([cstates[i]])

        # Check if the cstate and the grid match
        actual_grid = env.get_info()[0]["grid"]
        expected_grid = grids[i]

        if not np.array_equal(actual_grid, expected_grid):
            print(f"Reconstructed grids not equal to recording at t={i}")
            if use_pdb:
                pdb.set_trace()

        if np.any(actual_grid == 12):
            reset = True
            if not is_last:
                print(f"Found nonterminal fire in actual grid at t={i}")

        if np.any(expected_grid == 12) and not is_last:
            print(f"Found nonterminal fire in expected grid at t={i}")

        if not is_last:
            env.act(np.array([actions[i]]))
            actual_grid = env.get_info()[0]["grid"]
            expected_grid = grids[i + 1]
            if not np.array_equal(actual_grid, expected_grid):
                print(f"Step does not match recording at t={i + 1}")
                print(f"actual_grid={actual_grid}")
                print(f"expected_grid={expected_grid}")
                print(f"diff={(actual_grid - expected_grid)[:400].reshape((20,20))}")
                print(
                    f"expected old_agent={agent_pos[i-1]}, new_agent={agent_pos[i]}, actual_agent={env.get_info()[0]['agent_pos']}"
                )
                print(
                    f"expected old_exit={exit_pos[i-1]}, new_exit={exit_pos[i]}, actual_exit={env.get_info()[0]['exit_pos']}"
                )
                print(f"action={actions[i]}")
                if use_pdb:
                    pdb.set_trace()

        if np.any(actual_grid == 12):
            reset = True


def from_df(traj_dir: Path, use_pdb: bool = False) -> None:
    paths = list(Path(traj_dir).glob("trajectories_*.pkl"))
    print(f"Found {paths}")
    env, _ = rebuild_env("miner")
    for path in paths:
        print(f"Checking {path}")
        dataset: FeatureDataset = pkl.load(path.open("rb"))
        for index, row in dataset.df.iterrows():
            print(f"Checking traj {index}")
            actions = row["actions"]
            cstates = row["cstates"]
            grids = row["grids"]
            agent_pos = row["agent_pos"]
            exit_pos = row["exit_pos"]
            check_traj(actions, cstates, grids, agent_pos, exit_pos, use_pdb)


def from_db(db_path: Path, use_pdb: bool = False) -> None:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, start_state, cstates, actions FROM trajectories")
    for (id, start_state, cstates, actions) in c.fetchall():
        print(f"Checking traj {id}")
        start_state = pkl.loads(start_state)
        actions = pkl.loads(actions)
        cstates = pkl.loads(cstates)

        env = make_env("miner", 1, 0)
        root_env = get_root_env(env)
        root_env.set_state([cstates[0]])

        grids = []
        agent_pos = []
        exit_pos = []

        for action in actions:
            info = env.get_info()[0]
            grids.append(info["grid"])
            agent_pos.append(info["agent_pos"])
            exit_pos.append(info["exit_pos"])

            env.act(np.array([action]))

        check_traj(
            actions,
            cstates,
            np.array(grids),
            np.array(agent_pos),
            np.array(exit_pos),
        )

    conn.close()


if __name__ == "__main__":
    fire.Fire({"df": from_df, "db": from_db})
