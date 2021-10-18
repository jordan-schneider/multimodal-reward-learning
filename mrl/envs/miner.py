from typing import Any, Dict, Final, List, Optional, Tuple, Union, cast

import numpy as np
from procgen import ProcgenGym3Env  # type: ignore

__DIST_ARRAY = np.array([[np.abs(x) + np.abs(y) for x in range(-34, 35)] for y in range(-34, 35)])
DIAMOND_PERCENT = 12 / 400.0  # from miner.cpp


def get_dist_array(agent_x: int, agent_y: int, width: int, height: int) -> np.ndarray:
    return __DIST_ARRAY[34 - agent_x : 34 - agent_x + width, 34 - agent_y : 34 - agent_y + height]


class Miner(ProcgenGym3Env):
    ACTION_DICT: Final = {
        "up": 5,
        "down": 3,
        "left": 1,
        "right": 7,
        "stay": 4,
    }
    N_FEATURES: Final = 5

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
        use_normalized_features: bool = False,
        **kwargs,
    ) -> None:
        self._reward_weights = reward_weights
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
        self.last_diamonds = np.ones(num) * -1
        self.diamonds = np.array(
            [Miner.diamonds_remaining(state) for state in self.states], dtype=np.float32
        )
        self.firsts = [True] * num

        self.features: Optional[np.ndarray] = None

        self.use_normalized_features = use_normalized_features

    def act(self, action: np.ndarray) -> None:
        super().act(action)
        self.last_diamonds = self.diamonds
        self.states = self.make_latent_states()
        self.diamonds = np.array(
            [Miner.diamonds_remaining(state) for state in self.states], dtype=np.float32
        )

    def observe(self) -> Tuple[Any, Any, Any]:
        _, observations, self.firsts = super().observe()

        # compute features
        self.features = self.make_features()
        assert self._reward_weights.shape == (
            self.N_FEATURES,
        ), f"reward weights={self._reward_weights}"

        rewards = self.features @ self._reward_weights

        return rewards, observations, self.firsts

    def get_last_features(self) -> np.ndarray:
        # This is only a function because gym3.Wrapper doens't pass attributres through
        if self.features is None:
            self.features = self.make_features()
        return self.features

    class MinerState:
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
            self.grid: np.ndarray = grid.transpose()[: grid_size[0], : grid_size[1]]
            assert len(self.grid.shape) == 2
            self.agent_pos = tuple(agent_pos)
            self.exit_pos = tuple(exit_pos)

        def __eq__(self, other: Any) -> bool:
            correct_class = isinstance(other, Miner.MinerState)
            grid_equal = np.array_equal(self.grid, other.grid)
            agent_pos_equal = self.agent_pos == other.agent_pos
            exit_pos_equal = self.exit_pos == other.exit_pos
            return correct_class and grid_equal and agent_pos_equal and exit_pos_equal

    def make_latent_states(self) -> List[MinerState]:
        infos = self.get_info()
        return [self.make_latent_state(info) for info in infos]

    @staticmethod
    def make_latent_state(info: Dict[str, Any]) -> MinerState:
        return Miner.MinerState(
            info["grid_size"], info["grid"], info["agent_pos"], info["exit_pos"]
        )

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
        exits = np.array(
            [
                Miner.reached_exit(state, n_diamonds)
                for state, n_diamonds in zip(self.states, self.diamonds)
            ]
        )

        assert len(pickup) == self.num
        assert len(exits) == self.num
        assert len(dangers) == self.num
        assert len(dists) == self.num
        assert len(self.diamonds) == self.num

        if self.use_normalized_features:
            max_dist = float(self.states[0].grid.shape[0] * 2 - 1)
            max_diamonds = DIAMOND_PERCENT * self.states[0].grid.size
            dists /= max_dist
            self.diamonds /= max_diamonds

        features = np.array([pickup, exits, dangers, dists, self.diamonds], dtype=np.float32).T
        assert features.shape == (self.num, self.N_FEATURES)

        return features

    def in_danger(
        self, state: MinerState, return_time_to_die: bool = False, debug: bool = False
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
        # TODO(joschnei): Instead of indexing into this, try creating it on the fly, might be faster
        dists = get_dist_array(agent_x, agent_y, width, height)
        diamond_dists = np.ma.array(dists, mask=np.logical_not(diamonds))
        pos_closest_diamond = cast(
            Tuple[int, int], np.unravel_index(diamond_dists.argmin(), diamond_dists.shape)
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

        assert (
            n_diamonds <= last_n_diamonds
        ), f"There are {n_diamonds} this step vs {last_n_diamonds} last step, and first={first}"
        return n_diamonds != last_n_diamonds

    @staticmethod
    def reached_exit(state: MinerState, n_diamonds: int) -> bool:
        return n_diamonds == 0 and state.agent_pos == state.exit_pos
