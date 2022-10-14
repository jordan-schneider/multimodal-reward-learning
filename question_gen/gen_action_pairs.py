import logging
from typing import List, Optional, Tuple

import gym3  # type: ignore
import numpy as np
from experiment_server.type import FeatureTrajectory, State
from linear_procgen import ENV_NAMES as FEATURE_ENV_NAMES
from linear_procgen.util import get_root_env


def gen_action_pairs(
    env: gym3.Env,
    env_name: FEATURE_ENV_NAMES,
    n_pairs: int,
    n_steps: int,
    rng: np.random.Generator,
) -> List[Tuple[FeatureTrajectory, FeatureTrajectory]]:
    assert n_pairs <= n_steps
    # TODO: This is inefficient, we only need to branch every n_steps // n_pairs steps.
    pairs = get_branched_path(env, env_name, n_steps, n_steps, rng)
    logging.debug(f"Generated {len(pairs)} pairs")
    step_size = n_steps // n_pairs
    end = step_size * n_pairs
    logging.debug(f"Step size: {step_size}, end: {end}")
    return pairs[:end:step_size]


def get_branched_path(
    env: gym3.Env,
    env_name: FEATURE_ENV_NAMES,
    n_pairs: int,
    n_steps: int,
    rng: np.random.Generator,
) -> List[Tuple[FeatureTrajectory, FeatureTrajectory]]:
    assert n_pairs <= n_steps
    out: List[Tuple[FeatureTrajectory, FeatureTrajectory]] = []

    root_env = get_root_env(env)
    while len(out) < n_steps:
        start_cstate = root_env.get_state()
        start_state = build_state(env)
        start_feature = root_env.get_features()[0]

        first_action = get_action(env, [start_state], rng)

        env.act(first_action)
        first_features = root_env.get_features()[0]
        first_traj = FeatureTrajectory(
            start_state=start_state,
            actions=np.array([first_action]),
            env_name=env_name,
            modality="action",
            features=np.array([start_feature, first_features]),
        )

        second_state = build_state(env)

        root_env.set_state(start_cstate)
        second_action = get_action(env, [start_state, second_state], rng)

        if second_action is None:
            # If there is only one action, keep going and try again.
            env.act(first_action)
            continue

        env.act(second_action)
        second_features = root_env.get_features()[0]
        second_traj = FeatureTrajectory(
            start_state=start_state,
            actions=np.array([second_action]),
            env_name=env_name,
            modality="action",
            features=np.array([start_feature, second_features]),
        )

        out.append((first_traj, second_traj))
    return out


def get_action(
    env: gym3.Env, states_to_avoid: List[State], rng: np.random.Generator
) -> Optional[np.ndarray]:
    root_env = get_root_env(env)
    start_cstate = root_env.get_state()

    actions = np.array([1, 3, 5, 7]).reshape((4, 1))
    rng.shuffle(actions)

    for action in actions:
        env.act(action)
        current_state = build_state(env)
        root_env.set_state(start_cstate)
        if current_state not in states_to_avoid:
            return action

    return None


def build_state(env: gym3.Env) -> State:
    info = env.get_info()[0]
    return State(
        grid=info["grid"],
        grid_shape=tuple(info["grid_size"]),
        agent_pos=tuple(info["agent_pos"]),
        exit_pos=tuple(info["exit_pos"]),
    )
