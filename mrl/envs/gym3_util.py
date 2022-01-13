from typing import Tuple

import numpy as np


def recover_grid(grid: np.ndarray, grid_size: Tuple[int, int]) -> np.ndarray:
    """Function to recover a 2d grid from the fixed length 1d grid vector passed by gym3."""
    assert len(grid.shape) == 1
    grid = grid[: grid_size[0] * grid_size[1]].reshape(grid_size)
    # Reshape does things according to (y, x), I wrote miner.py as x, y
    grid = grid.transpose()
    return grid
