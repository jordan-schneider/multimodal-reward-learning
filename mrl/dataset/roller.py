import gc
import logging
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from mrl.dataset.trajectories import TrajectoryDataset
from mrl.envs.feature_envs import FeatureEnv
from mrl.envs.util import get_root_env
from mrl.memprof import get_memory
from phasic_policy_gradient.ppg import PhasicValueModel
from procgen import ProcgenGym3Env
from tqdm import trange  # type: ignore


def procgen_rollout(
    env: ProcgenGym3Env,
    policy: PhasicValueModel,
    timesteps: int,
    *,
    tqdm: bool = False,
    check_occupancies: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_shape = env.ob_space.shape

    states = np.empty((timesteps, env.num, *state_shape))
    actions = np.empty((timesteps - 1, env.num), dtype=np.int64)
    rewards = np.empty((timesteps, env.num))
    firsts = np.empty((timesteps, env.num), dtype=bool)

    if check_occupancies:
        timesteps_written_to: Dict[int, int] = {}

    def record(t: int, env: ProcgenGym3Env, states, rewards, firsts) -> None:
        if check_occupancies:
            timesteps_written_to[t] = timesteps_written_to.get(t, 0) + 1
        reward, state, first = env.observe()
        state = cast(
            np.ndarray, state
        )  # env.observe typing dones't account for wrapper
        states[t] = state
        rewards[t] = reward
        firsts[t] = first

    times = trange(timesteps - 1) if tqdm else range(timesteps - 1)

    for t in times:
        record(t, env, states, rewards, firsts)
        state = torch.tensor(states[t], device=policy.device)
        first = firsts[t]
        init_state = policy.initial_state(env.num)
        action, _, _ = policy.act(state, first, init_state)
        action = cast(torch.Tensor, action)
        action = action.cpu().numpy()
        env.act(action)
        actions[t] = action
    record(timesteps - 1, env, states, rewards, firsts)

    if check_occupancies:
        assert timesteps_written_to == {key: 1 for key in range(timesteps)}

    return states, actions, rewards, firsts


class ArrayOrList:
    def __init__(
        self, val: Union[np.ndarray, List[np.ndarray]], max_list_len: int = 1000
    ) -> None:
        self.val = val
        self.list = isinstance(val, list)
        self.max_list_len = max_list_len
        self.make_array_on_set = False
        if self.list:
            self.offset = 0
            if len(self.val) > 0:
                self.array = np.empty((0, *self.val[0].shape))
            else:
                self.make_array_on_set = True

    def __setitem__(self, key: int, value: np.ndarray) -> None:
        if self.list:
            assert isinstance(self.val, list)
            effective_key = key - self.offset
            if effective_key == len(self.val):
                if self.make_array_on_set:
                    self.make_array_on_set = False
                    self.array = np.empty((0, *value.shape))
                if len(self.val) > self.max_list_len:
                    self.coalesce()
                self.val.append(value)
            elif effective_key > len(self.val):
                raise IndexError(f"Index {key} is out of range")
            elif effective_key < 0:
                self.array[key] = value
            else:
                self.val[effective_key] = value
        else:
            self.val[key] = value

    def coalesce(self) -> None:
        if self.list:
            self.array = np.concatenate((self.array, np.array(self.val)))
            self.val = []
            self.offset = self.array.shape[0]

    def __getitem__(self, key: Union[int, slice]) -> Any:
        if self.list:
            self.coalesce()
            self.array[key]
        else:
            return self.val[key]

    def numpy(self) -> np.ndarray:
        if self.list:
            self.coalesce()
            return self.array
        else:
            assert isinstance(self.val, np.ndarray)
            return self.val


def procgen_rollout_features(
    env: ProcgenGym3Env,
    policy: PhasicValueModel,
    timesteps: Optional[int] = None,
    n_trajs: Optional[int] = None,
    tqdm: bool = False,
) -> np.ndarray:
    root_env = get_root_env(env)
    assert isinstance(root_env, FeatureEnv)

    features = ArrayOrList(
        np.empty((timesteps, env.num, root_env._reward_weights.shape[0]))
        if timesteps is not None
        else []
    )

    def step(t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, state, first = env.observe()
        features[t] = root_env.features
        action_tensor, _, _ = policy.act(
            torch.tensor(state, device=policy.device),
            torch.tensor(first, device=policy.device),
            policy.initial_state(env.num),
        )
        action_tensor = cast(torch.Tensor, action_tensor)
        action = cast(np.ndarray, action_tensor.cpu().numpy())
        env.act(action)
        return state, action, first

    if n_trajs is not None:
        cur_trajs = 0
        t = 0
        while cur_trajs < n_trajs and (timesteps is None or t < timesteps):
            _, _, first = step(t)
            cur_trajs += np.sum(first)
            t += 1
        features[t] = root_env.features
    elif timesteps is not None:
        times = trange(timesteps - 1) if tqdm else range(timesteps - 1)
        for t in times:
            step(t)
        features[timesteps - 1] = root_env.features
    else:
        raise ValueError("Must specify either timesteps or n_trajs")
    return features.numpy()


class DatasetRoller:
    def __init__(
        self,
        env: ProcgenGym3Env,
        policy: PhasicValueModel,
        n_actions: int = -1,
        n_trajs: Optional[int] = None,
        flags: Sequence[Literal["state", "action", "reward", "first", "feature"]] = [
            "state",
            "action",
            "reward",
            "first",
            "feature",
        ],
        remove_incomplete: bool = False,
        tqdm: bool = False,
    ):
        self.env = env
        self.root_env = cast(FeatureEnv, get_root_env(env))
        assert isinstance(self.root_env, FeatureEnv)

        self.policy = policy

        self.n_actions = n_actions
        if self.n_actions == 0 or self.n_actions < -1:
            raise ValueError("n_actions must be positive or -1")
        self.run_until_done = self.n_actions == -1

        self.n_trajs = n_trajs
        if n_trajs is not None and n_trajs <= 0:
            raise ValueError("n_trajs must be positive")
        if self.run_until_done and self.n_trajs is None:
            raise ValueError("Must specify n_trajs if timesteps is -1")

        self.fixed_size_arrays = n_actions > -1 and self.n_trajs is None

        self.flags = flags
        self.tqdm = tqdm

        state_shape = env.ob_space.shape

        self.states = self.make_array((n_actions + 1, env.num, *state_shape), "state")
        self.actions = self.make_array((n_actions, env.num), "action", dtype=np.uint8)
        self.rewards = self.make_array((n_actions + 1, env.num), "reward")
        self.firsts = self.make_array((n_actions + 1, env.num), "first", dtype=bool)
        self.features = self.make_array(
            (n_actions + 1, env.num, self.root_env.n_features), "feature"
        )

        self.remove_incomplete = remove_incomplete

    def make_array(
        self, shape: Tuple[int, ...], name: str, dtype=np.float32
    ) -> Optional[ArrayOrList]:
        if name in self.flags:
            return ArrayOrList(
                np.empty(shape, dtype=dtype) if self.fixed_size_arrays else []
            )
        else:
            return None

    def record(
        self,
        t: int,
        feature: Optional[np.ndarray],
        state: Optional[np.ndarray],
        reward: Optional[np.ndarray],
        first: Optional[np.ndarray],
    ) -> None:
        if self.states is not None and state is not None:
            self.states[t] = state
        if self.rewards is not None and reward is not None:
            self.rewards[t] = reward
        if self.firsts is not None and first is not None:
            self.firsts[t] = first
        if self.features is not None and feature is not None:
            self.features[t] = feature

    def step(self, t: int) -> np.ndarray:
        with torch.no_grad():
            reward, state, first = self.env.observe()
            self.record(t, self.root_env.features, state, reward, first)
            state_tensor = torch.tensor(state)
            first_tensor = torch.tensor(first)
            del reward, state
            init_state = self.policy.initial_state(self.env.num)
            action_tensor, _, _ = self.policy.act(
                state_tensor, first_tensor, init_state
            )
            del state_tensor, first_tensor, init_state
            action = cast(np.ndarray, cast(torch.Tensor, action_tensor).cpu().numpy())
            self.env.act(action)
            if self.actions is not None:
                self.actions[t] = action
            return first

    def roll(self) -> TrajectoryDataset:
        if self.n_trajs is not None:
            self.roll_fixed_trajs()
        elif self.n_actions > 0:
            self.roll_fixed_actions()
        else:
            raise ValueError("Must speficy n_trajs if timesteps=-1")

        states_arr, actions_arr, rewards_arr, firsts_arr, features_arr = (
            arr.numpy() if arr is not None else None
            for arr in (
                self.states,
                self.actions,
                self.rewards,
                self.firsts,
                self.features,
            )
        )

        return TrajectoryDataset.from_gym3(
            states=states_arr,
            actions=actions_arr,
            rewards=rewards_arr,
            firsts=firsts_arr,
            features=features_arr,
            remove_incomplete=self.remove_incomplete,
        )

    def roll_fixed_actions(self):
        times = trange(self.n_actions) if self.tqdm else range(self.n_actions)

        for t in times:
            self.step(t)

        reward, state, first = self.env.observe()
        self.record(self.n_actions, self.root_env.features, state, reward, first)

    def roll_fixed_trajs(self, garbage_collection_period: int = 500):
        assert self.n_trajs is not None
        cur_trajs = -self.env.num  # -1 per env because the first step has first=True
        t = 0
        while cur_trajs < self.n_trajs and (self.run_until_done or t < self.n_actions):
            if t % garbage_collection_period == 0:
                gc.collect()
                logging.debug(
                    f"step {t} vm={get_memory()['VmSize']}, peak={get_memory()['VmPeak']}"
                )
            first = self.step(t)

            cur_trajs += np.sum(first)
            del first
            t += 1

        if t == self.n_actions:
            # If you ran out of time, record the last state
            # TODO: This duplicates states if you're appending to the output and reusing the roller.
            # But we're never reusing the roller right now.
            reward, state, first = self.env.observe()
            self.record(self.n_actions, self.root_env.features, state, reward, first)


def procgen_rollout_dataset(
    env: ProcgenGym3Env,
    policy: PhasicValueModel,
    timesteps: int = -1,
    n_trajs: Optional[int] = None,
    flags: Sequence[Literal["state", "action", "reward", "first", "feature"]] = [
        "state",
        "action",
        "reward",
        "first",
        "feature",
    ],
    remove_incomplete: bool = True,
    tqdm: bool = False,
) -> TrajectoryDataset:
    roller = DatasetRoller(
        env=env,
        policy=policy,
        n_actions=timesteps,
        n_trajs=n_trajs,
        flags=flags,
        remove_incomplete=remove_incomplete,
        tqdm=tqdm,
    )
    return roller.roll()
