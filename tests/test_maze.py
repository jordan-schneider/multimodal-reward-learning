import logging
from typing import Dict, List, Tuple

import numpy as np
from mrl.envs import Maze


def test_step():
    num_envs = 5
    env = Maze(np.zeros(2), num_envs)
    obs, reward, first = env.observe()
    states = env.make_latent_states()
    features = env.make_features()

    env.act(np.zeros(num_envs))

    obs, reward, first = env.observe()
    states = env.make_latent_states()
    features = env.make_features()


def find_first_action(path: List[Tuple[int, int]], action_dict: Dict[str, int]) -> int:
    first_state, second_state = path[0], path[1]

    if first_state[0] > second_state[0]:
        action = action_dict["left"]
    elif first_state[0] < second_state[0]:
        action = action_dict["right"]
    elif first_state[1] > second_state[1]:
        action = action_dict["down"]
    else:
        action = action_dict["up"]

    return action


def test_correct_action():
    for _ in range(100):
        num_envs = 1
        env = Maze(np.zeros(2), num_envs)
        path = env.get_shortest_paths()[0]
        assert len(path) >= 2, f"path too short: {path}"
        start_features = env.make_features()[0]

        action = find_first_action(path, env.ACTION_DICT)

        env.act(np.array([action]))
        _, _, firsts = env.observe()

        end_features = env.make_features()[0]
        assert firsts[0] or end_features[0] < start_features[0]
        assert (
            firsts[0] or end_features[1] == 1
        ), f"init_dist={start_features[0]}, final_dist={end_features[0]} but end_features[1]={end_features[1]}"


def test_incorrect_action():
    for _ in range(100):
        num_envs = 1
        env = Maze(np.zeros(2), num_envs)
        path = env.get_shortest_paths()[0]
        assert len(path) >= 2, f"path too short: {path}"

        start_features = env.make_features()[0]

        good_action = find_first_action(path, env.ACTION_DICT)

        state = env.make_latent_states()[0]
        x, y = state.agent_pos
        if (
            x > 0
            and state.grid[x - 1, y] == 100
            and good_action != env.ACTION_DICT["left"]
        ):
            action = env.ACTION_DICT["left"]
        elif (
            x < state.grid.shape[0] - 1
            and state.grid[x + 1, y] == 100
            and good_action != env.ACTION_DICT["right"]
        ):
            action = env.ACTION_DICT["right"]
        elif (
            y > 0
            and state.grid[x, y - 1] == 100
            and good_action != env.ACTION_DICT["down"]
        ):
            action = env.ACTION_DICT["down"]
        elif (
            y < state.grid.shape[1] - 1
            and state.grid[x, y + 1] == 100
            and good_action != env.ACTION_DICT["up"]
        ):
            action = env.ACTION_DICT["up"]
        else:
            # No bad action, skip
            continue

        env.act(np.array([action]))

        _, _, first = env.observe()

        end_features = env.make_features()[0]
        assert end_features[0] > start_features[0]
        assert first or end_features[1] == -1


def test_bump_action():
    for _ in range(100):
        num_envs = 1
        env = Maze(np.zeros(2), num_envs)
        path = env.get_shortest_paths()[0]
        assert len(path) >= 2, f"path too short: {path}"

        start_features = env.make_features()[0]

        good_action = find_first_action(path, env.ACTION_DICT)

        state = env.make_latent_states()[0]
        x, y = state.agent_pos
        if (
            x > 0
            and state.grid[x - 1, y] != 100
            and good_action != env.ACTION_DICT["left"]
        ):
            action = env.ACTION_DICT["left"]
        elif (
            x < state.grid.shape[0] - 1
            and state.grid[x + 1, y] != 100
            and good_action != env.ACTION_DICT["right"]
        ):
            action = env.ACTION_DICT["right"]
        elif (
            y > 0
            and state.grid[x, y - 1] != 100
            and good_action != env.ACTION_DICT["down"]
        ):
            action = env.ACTION_DICT["down"]
        elif (
            y < state.grid.shape[1] - 1
            and state.grid[x, y + 1] != 100
            and good_action != env.ACTION_DICT["up"]
        ):
            action = env.ACTION_DICT["up"]
        else:
            # All directions open from here, skip
            continue

        env.act(np.array([action]))

        _, _, first = env.observe()

        end_features = env.make_features()[0]
        assert end_features[0] == start_features[0]
        assert first or end_features[1] == 0


def test_finish():
    for _ in range(100):
        num_envs = 1
        env = Maze(np.zeros(2), num_envs)

        path = env.get_shortest_paths()[0]
        while len(path) > 2:
            action = find_first_action(path, env.ACTION_DICT)

            env.act(np.array([action]))

            path = env.get_shortest_paths()[0]

        action = find_first_action(path, env.ACTION_DICT)
        env.act(np.array([action]))

        _, _, firsts = env.observe()
        assert firsts[0]


TIMEOUT = 500


def test_timeout():
    for _ in range(100):
        num_envs = 1
        env = Maze(np.zeros(2), num_envs)

        path = env.get_shortest_paths()[0]
        good_action = find_first_action(path, env.ACTION_DICT)

        bad_action = env.ACTION_DICT["up"]
        if bad_action == good_action:
            bad_action = env.ACTION_DICT["down"]

        for t in range(TIMEOUT):
            env.act(np.array([bad_action]))

        _, _, first = env.observe()
        assert first[0]
