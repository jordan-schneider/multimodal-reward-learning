import logging
from typing import Any, Dict, Final, List, Tuple

import numpy as np
from mrl.envs.util import recover_grid
from procgen import ProcgenGym3Env


class Maze(ProcgenGym3Env):
    ACTION_DICT: Final = {
        "up": 5,
        "down": 3,
        "left": 1,
        "right": 7,
        "stay": 4,
    }

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
        super().__init__(
            num=num,
            env_name="maze",
            center_agent=center_agent,
            use_backgrounds=use_backgrounds,
            use_monochrome_assets=use_monochrome_assets,
            restrict_themes=restrict_themes,
            use_generated_assets=use_generated_assets,
            paint_vel_info=paint_vel_info,
            distribution_mode=distribution_mode,
            **kwargs,
        )

        self._reward_weights = reward_weights
        self.use_normalized_features = normalize_features

    def act(self, action: np.ndarray) -> None:
        super().act(action)

    def observe(self) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        _, observations, firsts = super().observe()
        self.features = self.make_features()
        rewards = self.features @ self._reward_weights

        return rewards, observations, firsts

    def make_features(self) -> np.ndarray:
        pass

    class MazeState:
        def __init__(
            self,
            grid_size: Tuple[int, int],
            grid: np.ndarray,
            agent_pos: Tuple[int, int],
        ) -> None:
            print(grid)
            self.grid = recover_grid(grid, grid_size)
            assert len(self.grid.shape) == 2
            self.agent_pos = tuple(agent_pos)

        def __eq__(self, other: Any) -> bool:
            if not isinstance(other, Maze.MazeState):
                return False
            if not np.array_equal(self.grid, other.grid):
                return False
            if not self.agent_pos == other.agent_pos:
                return False
            return True

    def make_latent_states(self) -> List[MazeState]:
        infos = self.get_info()
        return [self.make_latent_state(info) for info in infos]

    @staticmethod
    def make_latent_state(info: Dict[str, Any]) -> MazeState:
        return Maze.MazeState(info["grid_size"], info["grid"], info["agent_pos"])
