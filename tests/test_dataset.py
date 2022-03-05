import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import integers
from mrl.dataset.trajectories import TrajectoryDataset


@given(
    timesteps=integers(min_value=1, max_value=1000),
    n_envs=integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_trajs_done_every_step(timesteps: int, n_envs: int) -> None:
    states = np.empty((timesteps, n_envs, 64, 64, 3))
    actions = np.empty((timesteps, n_envs), dtype=np.int8)
    rewards = np.empty((timesteps, n_envs))
    firsts = np.ones((timesteps, n_envs), dtype=bool)

    data = TrajectoryDataset.from_gym3(
        states=states,
        actions=actions,
        rewards=rewards,
        firsts=firsts,
        remove_incomplete=False,
    )
    n_trajs = 0
    for traj in data.trajs():
        n_trajs += 1
        assert traj.states is not None
        assert traj.actions is not None
        assert traj.rewards is not None
        assert traj.states.shape == (1, 64, 64, 3)
        assert traj.actions.shape == (1,)
        assert traj.rewards.shape == (1,)

    assert n_trajs == timesteps * n_envs


@given(
    trajs_per_env=integers(min_value=2, max_value=1000),
    n_envs=integers(min_value=1, max_value=10),
    length=integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_trajs(trajs_per_env: int, n_envs: int, length: int) -> None:
    timesteps = trajs_per_env * length
    states = np.empty((timesteps, n_envs, 64, 64, 3))
    actions = np.empty((timesteps, n_envs), dtype=np.int8)
    rewards = np.empty((timesteps, n_envs))
    firsts = np.zeros((timesteps, n_envs), dtype=bool)
    firsts[::length] = True

    data = TrajectoryDataset.from_gym3(
        states=states,
        actions=actions,
        rewards=rewards,
        firsts=firsts,
        remove_incomplete=True,
    )
    n_trajs = 0
    for traj in data.trajs():
        n_trajs += 1
        assert traj.states is not None
        assert traj.actions is not None
        assert traj.rewards is not None
        assert traj.states.shape == (length, 64, 64, 3)
        assert traj.actions.shape == (length,)
        assert traj.rewards.shape == (length,)

    assert n_trajs == (trajs_per_env - 1) * n_envs
