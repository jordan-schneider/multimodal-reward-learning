import logging
from pathlib import Path
from typing import Final, List, Literal, Sequence, Tuple, Type, cast

import fire  # type: ignore
import numpy as np
import torch
import torch.nn as nn
from gym3.extract_dict_ob import ExtractDictObWrapper  # type: ignore
from mrl.envs import Maze, Miner
from mrl.envs.util import FEATURE_ENV_NAMES, make_env
from mrl.util import batch, find_best_gpu, get_policy, setup_logging
from tqdm import trange  # type: ignore

# TODO: Figure out where to put this
MINER_GRID_ITEMS_TO_ONE_HOT_POS: Final = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 9: 5, 100: 6}
miner_n_objects: Final = len(MINER_GRID_ITEMS_TO_ONE_HOT_POS) + 1


class SuccessorFeatureModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        obs_shape: Tuple[int, ...],
        n_actions: int = 0,
        activation: Type[nn.Module] = nn.ReLU,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.n_features = n_features
        self.device = device

        self.input_size = int(np.prod(obs_shape))

        self.layers = self.make_layers(
            in_size=self.input_size,
            out_size=self.n_features,
            n_heads=n_actions,
            aspect_ratio=2.0,
            activation=activation,
        )

    def make_layers(
        self,
        in_size: int,
        out_size: int,
        n_heads: int,
        aspect_ratio: float,
        activation: Type[nn.Module],
    ) -> nn.Sequential:
        assert in_size > out_size
        current_out_size = in_size

        if n_heads == 0:
            n_heads = 1

        layers: List[nn.Module] = []
        while (
            next_out_size := int(current_out_size // aspect_ratio)
        ) > out_size * n_heads:
            layers.append(
                nn.Linear(current_out_size, next_out_size, device=self.device)
            )
            layers.append(activation())
            current_out_size = next_out_size

        layers.append(
            nn.Linear(current_out_size, out_size * n_heads, device=self.device)
        )
        return nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        in_device = obs.device
        obs = batch(obs, 2).to(self.device)
        obs = obs.reshape(obs.shape[0], -1)
        x = obs

        assert x.shape[1:] == (self.input_size)

        x = cast(torch.Tensor, self.layers(x).to(device=in_device))
        assert x.shape[1:] == (self.n_features * self.n_actions,)
        if self.n_actions > 0:
            x = x.reshape(x.shape[0], self.n_actions, self.n_features)
        return x


def latent_state_to_torch(states: Sequence, device: torch.device):
    if isinstance(states[0], Miner.State):
        return miner_state_to_torch(states, device)
    elif isinstance(states[0], Maze.State):
        return maze_state_to_torch(states, device)


def miner_state_to_torch(
    states: Sequence[Miner.State], device: torch.device
) -> torch.Tensor:
    # TODO: This function isn't really specific to the miner env, we just need to tell the function
    # our state to one hot encoding. Refactor to make this one function.
    grid_size = states[0].grid.shape
    out = torch.zeros(
        (len(states), *grid_size, miner_n_objects), dtype=torch.bool, device=device
    )
    for i, state in enumerate(states):
        for (x, y), item in np.ndenumerate(state.grid):
            out[i, x, y, MINER_GRID_ITEMS_TO_ONE_HOT_POS[item]] = True
        agent_x, agent_y = state.agent_pos
        out[i, agent_x, agent_y, 0:-1] = False
        out[i, agent_x, agent_y, -1] = True
    return out


def maze_state_to_torch(
    states: Sequence[Maze.State], device: torch.device
) -> torch.Tensor:
    # TODO: Implement
    raise NotImplementedError()


def main(
    policy_path: Path,
    env_name: FEATURE_ENV_NAMES,
    n_envs: int,
    normalize_features: bool = True,
    timesteps: int = 1_000_000,
    batch_size: int = 256,
    discount_rate: float = 1.0,
    lr: float = 1.0,
    weight_decay: float = 0,
    inputs: Literal["state", "state-action"] = "state",
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
):
    setup_logging(level=verbosity)
    assert discount_rate <= 1.0 and discount_rate >= 0.0
    assert weight_decay >= 0

    # TODO: Modify Miner to provide rgb (for the policy), the full state, and the feature vector
    # TODO: Modify Miner to use the use_done_feature flag, and set the default to False.
    env = make_env(
        name=env_name,
        num=n_envs,
        reward=0,
        normalize_features=normalize_features,
        extract_rgb=False,
    )
    rgb_env = ExtractDictObWrapper(env, "rgb")
    assert isinstance(env, Miner) or isinstance(env, Maze)

    states = env.make_latent_states()
    obs_shape = states[0].grid.shape

    features = env.make_features()
    n_features = features.shape[1]

    device = find_best_gpu()
    policy = get_policy(path=policy_path, env=rgb_env, device=device)

    successor_model = SuccessorFeatureModel(
        n_features=4,
        obs_shape=obs_shape,
        n_actions=env.ac_space.size if inputs == "state-action" else 0,
    )

    optim = torch.optim.Adam(
        params=successor_model.parameters(), lr=lr, weight_decay=weight_decay
    )

    n_batches = timesteps // batch_size

    state_batch = torch.empty(
        (batch_size, n_envs, *obs_shape, miner_n_objects), device=device
    )
    feature_batch = torch.empty((batch_size, n_envs, n_features), device=device)
    action_batch = torch.empty((batch_size, n_envs, *env.ac_space.shape), device=device)

    for t in trange(n_batches):
        for i in range(batch_size):
            rewards, obs, firsts = rgb_env.observe()
            obs = torch.tensor(obs)
            firsts = torch.tensor(firsts)

            state_batch[i] = latent_state_to_torch(
                env.make_latent_states(), device=device
            )
            feature_batch[i] = torch.tensor(env.make_features(), device=device)

            actions, _, _ = policy.act(obs, firsts, policy.initial_state(n_envs))
            actions = cast(torch.Tensor, actions)
            action_batch[i] = actions

            rgb_env.act(actions.cpu().numpy())

        optim.zero_grad()
        pred_features = successor_model.forward(state_batch).gather(1, action_batch)
        # TODO: Figure out how to write the bellman update efficient without replay.
        with torch.no_grad():
            if inputs == "state":
                feature_target = (
                    features[1:] + discount_rate * pred_features[1:].detach()
                )
            elif inputs == "state-action":
                pass
        td_error = feature_target - pred_features


if __name__ == "__main__":
    fire.Fire(main)
