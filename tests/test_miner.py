from typing import Final, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, lists
from mrl.envs.miner import Miner

UP: Final = 5
DOWN: Final = 3
LEFT: Final = 1
RIGHT: Final = 7
NONE: Final = 4
Action = Literal[1, 3, 5, 7]

DANGEROUS_OBJECTS: Final = (1, 2, 3, 4)
PATHABLE_OBJECTS: Final = (9, 100)
DIAMONDS = (1, 3)

N_FEATURES = 5


@settings(deadline=3000)
@given(seed=integers(0, 2 ** 31 - 1))
def test_first(seed: int) -> None:
    env = Miner(reward_weights=np.zeros(N_FEATURES), num=1, rand_seed=seed)
    _, _, first = env.observe()
    assert first

    start_state = env.make_latent_states()[0]
    path = find_dangerous_path_above(state=start_state)
    if path is not None:
        success = follow_path(env, path)
        if not success:
            pass

        go_down(env)

        env.act(np.array([NONE]))
        _, _, first = env.observe()
        assert (
            first
        ), f"Not first after death. start_state=\n{start_state.grid}, path={path}, current_state=\n{env.make_latent_states()[0].grid}"


@settings(deadline=3000)
@given(
    actions=lists(integers(min_value=0, max_value=15), min_size=1, max_size=10),
    seed=integers(0, 2 ** 31 - 1),
)
def test_grid_items(actions: List[int], seed: int) -> None:
    env = Miner(reward_weights=np.zeros(N_FEATURES), num=1, rand_seed=seed)
    for action in actions:
        env.act(np.array([action]))
        state = env.make_latent_states()[0]

        grid_shape = state.grid.shape
        assert grid_shape == (20, 20)

        grid_keys = state.GRID_ITEM_NAMES.keys()
        for row in state.grid:
            for item in row:
                assert item in grid_keys, f"Invalid item={item} in grid={state.grid}"


@settings(deadline=3000)
@given(
    actions=lists(integers(min_value=0, max_value=15), min_size=1, max_size=10),
    seed=integers(0, 2 ** 31 - 1),
)
def test_empty_increasing(actions: List[int], seed: int):
    env = Miner(reward_weights=np.zeros(N_FEATURES), num=1, rand_seed=seed)
    last_n_empty = -1
    for action in actions:
        env.act(np.array([action]))
        _, _, first = env.observe()

        if first:
            last_n_empty = -1

        state = env.make_latent_states()[0]

        n_empty = np.sum(state.grid == 100)

        assert n_empty >= last_n_empty
        last_n_empty = n_empty


@settings(deadline=3000)
@given(
    actions=lists(integers(min_value=0, max_value=15), min_size=1, max_size=10),
    seed=integers(0, 2 ** 31 - 1),
)
def test_diamonds_remaining_decreasing(actions: List[int], seed: int):
    env = Miner(reward_weights=np.zeros(N_FEATURES), num=1, rand_seed=seed)
    last_n_diamonds = 35 * 35 + 1  # number of cells + 1
    for action in actions:
        env.act(np.array([action]))
        _, _, first = env.observe()

        if first:
            last_n_diamonds = 35 * 35 + 1

        state = env.make_latent_states()[0]

        n_diamonds = env.diamonds_remaining(state)

        assert n_diamonds <= last_n_diamonds
        last_n_diamonds = n_diamonds


@settings(deadline=3000)
@given(
    seed=integers(0, 2 ** 31 - 1),
)
def test_start_diamonds_remaining(seed: int):
    frac = 12 / 400.0
    env = Miner(reward_weights=np.zeros(N_FEATURES), num=1, rand_seed=seed)
    state = env.make_latent_states()[0]
    diamonds_remaining = env.diamonds_remaining(state)
    assert diamonds_remaining == int(frac * state.grid.size)


@settings(deadline=3000)
@given(
    seed=integers(0, 2 ** 31 - 1),
)
def test_dist_to_diamond(seed: int):
    env = Miner(reward_weights=np.zeros(N_FEATURES), num=1, rand_seed=seed)
    start_state = env.make_latent_states()[0]
    diamonds_remaining = Miner.diamonds_remaining(start_state)
    starting_dist, pos = cast(
        Tuple[int, Tuple[int, int]],
        Miner.dist_to_diamond(start_state, diamonds_remaining, return_pos=True),
    )
    agent_x, agent_y = start_state.agent_pos

    # If the diamond is to the right of the agent and there isn't a boulder in the way, go right.
    if pos[0] > agent_x and start_state.grid[agent_x + 1][agent_y] not in DIAMONDS:
        action = np.array([RIGHT])
    elif pos[0] < agent_x and start_state.grid[agent_x - 1][agent_y] not in DIAMONDS:
        action = np.array([LEFT])
    elif pos[1] > agent_y and start_state.grid[agent_x][agent_y + 1] not in DIAMONDS:
        action = np.array([UP])
    elif pos[1] < agent_y and start_state.grid[agent_x][agent_y - 1] not in DIAMONDS:
        action = np.array([DOWN])
    else:
        return  # If there is a boulder in every direction closer to the diamond, then pass the test

    env.act(action)

    _, _, first = env.observe()

    end_state = env.make_latent_states()[0]
    diamonds_remaining = Miner.diamonds_remaining(end_state)
    ending_dist = cast(int, Miner.dist_to_diamond(end_state, diamonds_remaining))

    assert (
        first
        or np.all(end_state.agent_pos == pos)  # The agent died as a result of its move
        or starting_dist  # Or the agent got the diamond
        > ending_dist  # Or the agent moved closer to the diamond
    ), f"Distance increased after taking action={action}, from {start_state.agent_pos} to {end_state.agent_pos} for target {pos} on grid={start_state.grid} to grid={end_state.grid}"


@settings(deadline=3000)
@given(
    seed=integers(0, 2 ** 31 - 1),
)
def test_safe_at_start(seed):
    env = Miner(reward_weights=np.zeros(N_FEATURES), num=1, rand_seed=seed)
    start_state = env.make_latent_states()[0]
    agent_pos = start_state.agent_pos
    danger, t = cast(
        Tuple[bool, int],
        env.in_danger(start_state, return_time_to_die=True, debug=True),
    )
    assert (
        not danger
    ), f"Agent (pos={agent_pos}) should not be in danger at start of game, but is at t={t} for grid=\n{start_state.grid}"


def flood_search(
    state: Miner.State, goals: List[Tuple[int, int]]
) -> Optional[List[Action]]:
    """Perform a flood search for a path to one of the goal states.

    Not guaranteed to be the shortest path, or even a valid path, as falling objects can block the
    path.
    """
    last_action = np.empty(state.grid.shape, dtype=np.int8)
    last_action[state.agent_pos[0]][state.agent_pos[1]] = -1
    states = [state.agent_pos]
    visited_states = set()
    while len(states) > 0:
        x, y = states.pop()
        visited_states.add((x, y))

        if (x, y) in goals:
            # Recover path from sequence of last actions.
            actions = []
            while last_action[x][y] != -1:
                actions.append(last_action[x][y])
                if last_action[x][y] == UP:
                    y -= 1
                elif last_action[x][y] == DOWN:
                    y += 1
                elif last_action[x][y] == LEFT:
                    x += 1
                elif last_action[x][y] == RIGHT:
                    x -= 1
            actions.reverse()
            return actions

        # Check if you can move in each direction
        if (
            x > 0
            and (x - 1, y) not in visited_states
            and (x - 1, y) not in states
            and state.grid[x - 1][y] in PATHABLE_OBJECTS
        ):
            last_action[x - 1][y] = LEFT
            states.append((x - 1, y))
        if (
            x + 1 < state.grid.shape[0]
            and (x + 1, y) not in visited_states
            and (x + 1, y) not in states
            and state.grid[x + 1][y] in PATHABLE_OBJECTS
        ):
            last_action[x + 1][y] = RIGHT
            states.append((x + 1, y))
        if (
            y > 0
            and (x, y - 1) not in visited_states
            and (x, y - 1) not in states
            and state.grid[x][y - 1] in PATHABLE_OBJECTS
        ):
            last_action[x][y - 1] = DOWN
            states.append((x, y - 1))
        if (
            y + 1 < state.grid.shape[1]
            and (x, y + 1) not in visited_states
            and (x, y + 1) not in states
            and state.grid[x][y + 1] in PATHABLE_OBJECTS
        ):
            last_action[x][y + 1] = UP
            states.append((x, y + 1))
    return None


def find_dangerous_path_above(state: Miner.State) -> Optional[List[Action]]:
    agent_x, agent_y = state.agent_pos
    grid_height = state.grid.shape[1]

    # If there is an object directly above the agent and it can go down, move down.
    # If there's an object directly above and below, quit the heuristic.
    if (
        agent_y + 1 < grid_height
        and state.grid[agent_x][agent_y + 1] in DANGEROUS_OBJECTS
    ):
        if agent_y - 1 > 0 and state.grid[agent_x][agent_y - 1] in PATHABLE_OBJECTS:
            return []
        else:
            return None

    # If there's nothing directly above, search for things at least 2 above.
    for y in range(agent_y + 2, grid_height):
        if state.grid[agent_x][y] in DANGEROUS_OBJECTS:
            path = [UP] * (y - agent_y - 1)
            return cast(List[Action], path)

    # If there's nothing dangerous anywhere above, return None.
    return None


def find_dangerous_object_candidates(state: Miner.State) -> List[Tuple[int, int]]:
    """Find all dangerous objects with two empty spaces underneath"""
    danger_objs: List[Tuple[int, int]] = []
    for x, row in enumerate(state.grid):
        for y, item in enumerate(row):
            item_moves = item in DANGEROUS_OBJECTS
            dirt_below = y + 1 > 0 and state.grid[x][y - 1] == 9
            movable_two_below = y - 2 > 0 and state.grid[x][y - 2] in PATHABLE_OBJECTS
            if item_moves and dirt_below and movable_two_below:
                danger_objs.append((x, y))

    return danger_objs


def find_path_to_dangerous_state(state: Miner.State) -> Optional[List[Action]]:
    """Tries to find a path to a dangerous state, where an object is about to fall on you.

    The pathing does not model changes in the state from falling objects, and so sometimes the
    returned path will not be valid.
    """

    # First try a simple heuristic of looking for a heavy object above the agent.
    path = find_dangerous_path_above(state)
    if path is not None:
        return path

    # There's no heavy object above us, so we search properly.

    danger_objs = find_dangerous_object_candidates(state)

    # Can't path to something that doesn't exist
    if len(danger_objs) == 0:
        return None

    # Flood search to any space directly under a dangerous object
    return flood_search(state=state, goals=[(x, y - 1) for x, y in danger_objs])


@settings(deadline=3000)
@given(
    seed=integers(0, 2 ** 31 - 1),
)
def test_in_danger(seed):
    # We need to find a boulder or diamond with a dirt directly underneath it and a tile we can move
    # into two below it. Then we can find a path to underneath the object, move down, and be in
    # danger.

    env = Miner(reward_weights=np.zeros(N_FEATURES), num=1, rand_seed=seed)
    start_state = env.make_latent_states()[0]

    path = find_path_to_dangerous_state(start_state)

    if path is None:
        return

    success = follow_path(env, path)
    if not success:
        pass

    go_down(env)

    state = env.make_latent_states()[0]
    agent_x, agent_y = state.agent_pos
    danger, t = cast(Tuple[bool, int], env.in_danger(state, return_time_to_die=True))

    should_be_danger = state.grid[agent_x, agent_y + 1] in DANGEROUS_OBJECTS

    if should_be_danger:
        assert (
            danger
        ), f"No danger at t={t} for agent pos={state.agent_pos}, grid\n={state.grid}\n from start=\n{start_state.grid}"
    else:
        assert (
            not danger
        ), f"Bad path. There isn't danger but we think there is at t={t}, grid=\n{state.grid}\n from start=\n{start_state.grid} "


def go_down(env: Miner) -> None:
    old_agent_pos = env.make_latent_states()[0].agent_pos
    agent_pos = old_agent_pos

    # We're in a dangerous position, keep trying to go down.
    # Occassionally we can't do this the first timestep because something is in the process of
    # falling below us.
    while agent_pos[0] == old_agent_pos[0] and agent_pos[1] == old_agent_pos[1]:
        old_agent_pos = agent_pos
        env.act(np.array([DOWN]))
        agent_pos = env.make_latent_states()[0].agent_pos


def follow_path(env: Miner, path: Sequence[Action]) -> bool:
    old_agent_pos = env.make_latent_states()[0].agent_pos
    for action in path:
        env.act(np.array([action]))
        state = env.make_latent_states()[0]
        if (
            state.agent_pos[0] == old_agent_pos[0]
            and state.agent_pos[1] == old_agent_pos[1]
        ):
            # We tried to path through an immobile rock, I'm not going to deal with this case
            return False
    return True


@settings(deadline=3000)
@given(
    seed=integers(0, 2 ** 31 - 1),
    reward_weights=arrays(
        np.float64,
        (N_FEATURES,),
        elements=floats(
            min_value=1e10, max_value=1e10, allow_nan=False, allow_infinity=False
        ),
    ),
)
def test_reward(seed: int, reward_weights: np.ndarray) -> None:
    env = Miner(reward_weights=reward_weights, num=1, rand_seed=seed)
    features = env.make_features()

    rewards, _, _ = env.observe()
    assert np.array_equal(rewards, features @ reward_weights)


@settings(deadline=3000)
@given(
    actions=lists(integers(min_value=0, max_value=15), min_size=1, max_size=10),
    seed=integers(0, 2 ** 31 - 1),
)
def test_normalized_features(seed: int, actions: List[int]) -> None:
    env = Miner(
        reward_weights=np.zeros(5), num=1, rand_seed=seed, normalize_features=True
    )
    for action in actions:
        env.act(np.array([action]))
        features = env.make_features()
        assert np.all(features >= 0)
        assert np.all(features <= 1)
