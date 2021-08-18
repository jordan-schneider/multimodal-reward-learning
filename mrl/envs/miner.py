from typing import Any, Dict, Final, List, Optional, Tuple, Union, cast

import numpy as np
from procgen import ProcgenGym3Env  # type: ignore


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
        self.last_diamonds = [-1] * num
        self.diamonds = [Miner.diamonds_remaining(state) for state in self.states]
        self.firsts = [True] * num

        self.danger_sim = ProcgenGym3Env(
            1,
            env_name="miner",
            distribution_mode=kwargs.get("distribution_mode", "hard"),
        )

    def act(self, action: np.ndarray) -> None:
        super().act(action)
        self.last_diamonds = self.diamonds
        self.states = self.make_latent_states()
        self.diamonds = [Miner.diamonds_remaining(state) for state in self.states]

    def observe(self) -> Tuple[Any, Any, Any]:
        _, observations, self.firsts = super().observe()

        # compute features
        features = self.make_features()
        assert self._reward_weights.shape == (
            self.N_FEATURES,
        ), f"reward weights={self._reward_weights}"

        rewards = features @ self._reward_weights

        return rewards, observations, self.firsts

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
            serialization: str,
        ) -> None:
            self.grid: np.ndarray = grid.transpose()[: grid_size[0], : grid_size[1]]
            assert len(self.grid.shape) == 2
            self.agent_pos = tuple(agent_pos)
            self.exit_pos = tuple(exit_pos)
            self.serialization = serialization

        def __eq__(self, other: Any) -> bool:
            correct_class = isinstance(other, Miner.MinerState)
            grid_equal = np.array_equal(self.grid, other.grid)
            agent_pos_equal = self.agent_pos == other.agent_pos
            exit_pos_equal = self.exit_pos == other.exit_pos
            serialization_equal = self.serialization == other.serialization
            return (
                correct_class
                and grid_equal
                and agent_pos_equal
                and exit_pos_equal
                and serialization_equal
            )

    def make_latent_states(self) -> List[MinerState]:
        infos = self.get_info()
        serializations = self.get_state()
        return [
            self.make_latent_state(info, serialization)
            for info, serialization in zip(infos, serializations)
        ]

    @staticmethod
    def make_latent_state(info: Dict[str, Any], serialization: str) -> MinerState:
        return Miner.MinerState(
            info["grid_size"], info["grid"], info["agent_pos"], info["exit_pos"], serialization
        )

    def make_features(self) -> np.ndarray:
        dangers = [self.in_danger(state) for state in self.states]
        dists = [Miner.dist_to_diamond(state) for state in self.states]
        pickup = [
            Miner.got_diamond(n_diamonds, last_n_diamonds, first)
            for n_diamonds, last_n_diamonds, first in zip(
                self.diamonds, self.last_diamonds, self.firsts
            )
        ]
        exits = [
            Miner.reached_exit(state, n_diamonds)
            for state, n_diamonds in zip(self.states, self.diamonds)
        ]

        features = np.array([pickup, exits, dangers, dists, self.diamonds], dtype=np.float32).T
        assert features.shape == (self.num, self.N_FEATURES)

        return features

    def in_danger(
        self, state: MinerState, return_time_to_die: bool = False, debug: bool = False
    ) -> Union[bool, Tuple[bool, int]]:
        self.danger_sim.set_state((state.serialization,))

        if debug:
            start_state = Miner.make_latent_state(
                self.danger_sim.get_info()[0], self.danger_sim.get_state()[0]
            )

        _, last_obs, _ = self.danger_sim.observe()

        self.danger_sim.act(np.array([Miner.ACTION_DICT["stay"]]))
        _, current_obs, first = self.danger_sim.observe()

        if debug and not first:
            after_state = Miner.make_latent_state(
                self.danger_sim.get_info()[0], self.danger_sim.get_state()[0]
            )
            assert (
                state.agent_pos[0] == after_state.agent_pos[0]
                and state.agent_pos[1] == after_state.agent_pos[1]
            ), f"Agent moved from {start_state.agent_pos} to {after_state.agent_pos} with exit at {start_state.exit_pos} on\n{start_state.grid}\nto\n{after_state.grid}"

        t = 1

        while np.all(current_obs["rgb"] != last_obs["rgb"]) and not first:
            last_obs = current_obs
            self.danger_sim.act(np.array([Miner.ACTION_DICT["stay"]]))
            _, current_obs, first = self.danger_sim.observe()
            t += 1
            if debug:
                print(t)
                print(
                    Miner.make_latent_state(
                        self.danger_sim.get_info()[0], self.danger_sim.get_state()[0]
                    ).grid
                )

        # first means that we died somehow, which means we're in danger
        # TODO(joschnei): This isn't quite true. There's a time horizon somewhere of 1000, and if
        # it is also here then we can get first by simply running out of time. I probably won't
        # fix this, as it only happens near the very end of the time horizon, and if you make it out
        # there the danger penalty probably isn't affecting much.
        if return_time_to_die:
            return first, t
        else:
            return first

    @staticmethod
    def dist_to_diamond(
        state: MinerState, return_pos: bool = False
    ) -> Union[int, Tuple[int, Tuple[int, int]]]:
        agent_x, agent_y = state.agent_pos
        min_dist = 35 * 2 + 1  # maximum possible L_1 distance on a 35x35 grid
        assert len(state.grid.shape) == 2
        for x in range(state.grid.shape[0]):
            for y in range(state.grid.shape[1]):
                if "diamond" in state.GRID_ITEM_NAMES[state.grid[x][y]]:
                    dist = np.abs(agent_x - x) + np.abs(agent_y - y)
                    if dist < min_dist:
                        min_dist = dist
                        pos_closest_diamond = (x, y)

        assert state.grid[pos_closest_diamond[0]][pos_closest_diamond[1]] in (2, 4)

        if return_pos:
            return min_dist, pos_closest_diamond
        else:
            return min_dist

    @staticmethod
    def diamonds_remaining(state: MinerState) -> int:
        return np.sum(["diamond" in state.GRID_ITEM_NAMES[item] for item in state.grid.flatten()])

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
