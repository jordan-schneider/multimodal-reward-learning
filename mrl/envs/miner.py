import logging
from typing import Any, Dict, Final, List, Optional, Tuple, Type, Union, cast

import numpy as np
from mrl.envs.feature_envs import FeatureEnv, StateInterface
from mrl.envs.gym3_util import recover_grid

__DIST_ARRAY = np.array(
    [[np.abs(x) + np.abs(y) for x in range(-34, 35)] for y in range(-34, 35)]
)
DIAMOND_PERCENT = 12 / 400.0  # from miner.cpp


def get_dist_array(agent_x: int, agent_y: int, width: int, height: int) -> np.ndarray:
    return __DIST_ARRAY[
        34 - agent_x : 34 - agent_x + width, 34 - agent_y : 34 - agent_y + height
    ]


class MinerState(StateInterface):
    GRID_ITEM_NAMES = {
        1: "boulder",
        2: "diamond",
        3: "moving_boulder",
        4: "moving_diamond",
        5: "enemy",
        6: "exit",
        9: "dirt",
        10: "oob_wall",
        100: "space",
    }

    def __init__(
        self,
        grid_size: Tuple[int, int],
        grid: np.ndarray,
        agent_pos: Tuple[int, int],
        exit_pos: Tuple[int, int],
    ) -> None:
        self.grid = recover_grid(grid, grid_size)
        assert len(self.grid.shape) == 2
        self.agent_pos = agent_pos
        self.exit_pos = exit_pos

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MinerState):
            return False
        if not np.array_equal(self.grid, other.grid):
            return False
        if not self.agent_pos == other.agent_pos:
            return False
        if not self.exit_pos == other.exit_pos:
            return False
        return True


class Miner(FeatureEnv[MinerState]):
    ACTION_DICT: Final = {
        "up": 5,
        "down": 3,
        "left": 1,
        "right": 7,
        "stay": 4,
    }

    class State(MinerState):
        # Allows the typing for Miner.State to work.
        pass

    def __init__(
        self,
        reward_weights: np.ndarray,
        num: int,
        center_agent: bool = True,
        use_backgrounds: bool = True,
        use_monochrome_assets: bool = False,
        restrict_themes: bool = False,
        use_generated_assets: bool = False,
        paint_vel_info: bool = False,
        distribution_mode: str = "hard",
        normalize_features: bool = False,
        **kwargs,
    ) -> None:
        if reward_weights.shape[0] != 4:
            raise ValueError(f"Must supply 4 reward weights, {reward_weights=}")

        self._reward_weights = reward_weights
        self._n_features = reward_weights.shape[0]
        self.use_normalized_features = normalize_features
        super().__init__(
            num=num,
            env_name="miner",
            center_agent=center_agent,
            use_backgrounds=use_backgrounds,
            use_monochrome_assets=use_monochrome_assets,
            restrict_themes=restrict_themes,
            use_generated_assets=use_generated_assets,
            paint_vel_info=paint_vel_info,
            distribution_mode=distribution_mode,
            **kwargs,
        )
        self.states = self.make_latent_states()
        self.last_diamonds = np.ones(num, dtype=int) * -1
        self.diamonds = np.array(
            [Miner.diamonds_remaining(state) for state in self.states], dtype=int
        )
        self.firsts = [True] * num

        self.features = self.make_features()

    def act(self, action: np.ndarray) -> None:
        super().act(action)
        self.last_diamonds = self.diamonds
        self.states = self.make_latent_states()
        self.diamonds = np.array(
            [Miner.diamonds_remaining(state) for state in self.states], dtype=np.float32
        )

    def observe(self) -> Tuple[np.ndarray, Any, Any]:
        _, observations, self.firsts = super().observe()

        # compute features
        self.features = self.make_features()
        rewards = self.features @ self._reward_weights

        return rewards, observations, self.firsts

    def make_latent_states(self) -> List[MinerState]:
        infos = self.get_info()
        return [self.make_latent_state(info) for info in infos]

    @staticmethod
    def make_latent_state(info: Dict[str, Any]) -> MinerState:
        agent_pos = cast(Tuple[int, int], tuple(info["agent_pos"]))
        exit_pos = cast(Tuple[int, int], tuple(info["exit_pos"]))
        return Miner.State(info["grid_size"], info["grid"], agent_pos, exit_pos)

    def make_features(self) -> np.ndarray:
        dangers = np.array([self.in_danger(state) for state in self.states])
        dists = np.array(
            [
                Miner.dist_to_diamond(state, diamonds_remaining)
                for state, diamonds_remaining in zip(self.states, self.diamonds)
            ],
            dtype=np.float32,
        )
        pickup = np.array(
            [
                Miner.got_diamond(n_diamonds, last_n_diamonds, first)
                for n_diamonds, last_n_diamonds, first in zip(
                    self.diamonds, self.last_diamonds, self.firsts
                )
            ]
        )

        diamonds = np.array(self.diamonds, dtype=np.float32)

        assert len(pickup) == self.num
        assert len(dangers) == self.num
        assert len(dists) == self.num
        assert len(diamonds) == self.num

        if self.use_normalized_features:
            max_dist = float(self.states[0].grid.shape[0] * 2 - 1)
            dists /= max_dist

            max_diamonds = DIAMOND_PERCENT * self.states[0].grid.size
            diamonds /= max_diamonds

        features = np.array([pickup, dangers, dists, diamonds], dtype=np.float32).T
        assert features.shape == (self.num, self._n_features)

        return features

    @property
    def n_features(self) -> int:
        return self._n_features

    @staticmethod
    def in_danger(
        state: MinerState, return_time_to_die: bool = False, debug: bool = False
    ) -> Union[bool, Tuple[bool, int]]:
        agent_x, agent_y = state.agent_pos

        # You can't be in danger if there's nothing above you
        if agent_y + 1 >= state.grid.shape[1]:
            return (False, -1) if return_time_to_die else False

        # You are only in danger if the thing directly above you is moving
        if state.grid[agent_x, agent_y + 1] in {3, 4}:
            return (True, 1) if return_time_to_die else True
        elif state.grid[agent_x, agent_y + 1] in {1, 2, 9, 10}:
            return (False, -1) if return_time_to_die else False

        for y in range(agent_y + 2, state.grid.shape[1]):
            if state.grid[agent_x, y] in {1, 2, 3, 4}:
                t = y - agent_y
                return (True, t) if return_time_to_die else True
            elif state.grid[agent_x, y] in {9, 10}:
                return (False, -1) if return_time_to_die else False

        return (False, -1) if return_time_to_die else False

    @staticmethod
    def dist_to_diamond(
        state: MinerState, diamonds_remaining: int, return_pos: bool = False
    ) -> Union[int, Tuple[int, Tuple[int, int]]]:
        if diamonds_remaining == 0:
            if return_pos:
                return 0, (-1, -1)
            else:
                return 0

        agent_x, agent_y = state.agent_pos
        width, height = state.grid.shape
        diamonds = cast(np.ndarray, np.logical_or(state.grid == 2, state.grid == 4))
        dists = get_dist_array(agent_x, agent_y, width, height)
        diamond_dists = np.ma.array(dists, mask=np.logical_not(diamonds))
        pos_closest_diamond = cast(
            Tuple[int, int],
            np.unravel_index(diamond_dists.argmin(), diamond_dists.shape),
        )
        min_dist = diamond_dists[pos_closest_diamond]

        if return_pos:
            return min_dist, pos_closest_diamond
        else:
            return min_dist

    @staticmethod
    def diamonds_remaining(state: MinerState) -> int:
        return np.sum((state.grid == 2) | (state.grid == 4))

    @staticmethod
    def got_diamond(n_diamonds: int, last_n_diamonds: int, first: bool) -> bool:
        if first:
            return False

        if n_diamonds > last_n_diamonds:
            raise Exception(
                f"There are {n_diamonds} this step vs {last_n_diamonds} last step, and first={first}."
            )
        return n_diamonds != last_n_diamonds
