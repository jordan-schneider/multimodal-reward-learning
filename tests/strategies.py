import numpy as np
from experiment_server.type import FeatureTrajectory, State, Trajectory
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import characters, composite, floats, integers, tuples

seeds = integers(0, 2**32 - 1)

small_int_strategy = integers(1, 5)

finite_floats = floats(allow_infinity=False, allow_nan=False, width=32)
floats_1000 = floats(min_value=-1000, max_value=1000, allow_nan=False, width=32)


@composite
def halfplanes_strategy(draw, n_halfplanes=small_int_strategy):
    return draw(
        arrays(
            np.float32,
            (draw(n_halfplanes), 4),
            elements=floats_1000,
        )
    )


reward_strategy = (
    arrays(np.float32, (4,), elements=floats_1000)
    .filter(lambda r: np.any(r != 0.0))
    .map(lambda r: r / np.linalg.norm(r))
)


@composite
def rewards_strategy(draw, n_rewards=small_int_strategy):
    return draw(
        arrays(np.float32, (draw(n_rewards), 4), elements=floats_1000)
        .filter(lambda r: np.linalg.norm(r) > 0.0)
        .map(lambda r: r / np.linalg.norm(r))
    )


@composite
def states(draw, max_grid_size: int = 10) -> State:
    grid_size = draw(integers(min_value=1, max_value=max_grid_size))
    grid = draw(
        arrays(dtype=np.int32, shape=(grid_size, grid_size), elements=integers(1, 100))
    )
    pos_val = integers(min_value=0, max_value=grid_size - 1)
    position = tuples(pos_val, pos_val)
    agent_pos = draw(position)
    exit_pos = draw(position)
    return State(grid, (grid_size, grid_size), agent_pos, exit_pos)


actions = arrays(
    dtype=np.int32, shape=array_shapes(max_dims=1), elements=integers(0, 4)
)


@composite
def trajs(draw, states, action_strategy):
    start_state = draw(states)
    actions = draw(action_strategy)
    env_name = draw(characters())
    modality = draw(characters())
    return Trajectory(
        start_state=start_state, actions=actions, env_name=env_name, modality=modality
    )


@composite
def feature_trajs(draw, states, action_strategy):
    start_state = draw(states)
    actions = draw(action_strategy)
    env_name = draw(characters())
    modality = draw(characters())
    features = draw(
        arrays(dtype=np.float32, shape=(actions.shape[0], 4), elements=floats_1000)
    )
    return FeatureTrajectory(
        start_state=start_state,
        actions=actions,
        env_name=env_name,
        modality=modality,
        features=features,
    )
