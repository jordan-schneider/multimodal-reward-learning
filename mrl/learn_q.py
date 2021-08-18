from copy import deepcopy
from pathlib import Path
from typing import Callable, cast

import fire  # type: ignore
import torch
from gym3 import Env  # type: ignore
from gym3 import ExtractDictObWrapper
from phasic_policy_gradient.ppg import PhasicValueModel
from phasic_policy_gradient.roller import Roller
from procgen import ProcgenGym3Env
from torch.utils.data import DataLoader

from mrl.offline_buffer import SarsDataset
from mrl.util import get_model_path


class QNetwork(torch.nn.Module):
    def __init__(self, policy: PhasicValueModel, n_actions: int, discount_rate: float):
        super().__init__()
        assert discount_rate >= 0.0 and discount_rate <= 1.0
        self.discount_rate = discount_rate

        self.enc = deepcopy(policy.get_encoder(policy.true_vf_key))

        # TODO: Try random initialization to see if we're getting negative transfer here.
        self.head = self.add_action_heads(
            n_actions=n_actions, value_head=policy.get_vhead(policy.true_vf_key)
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(obs))[action]

    @staticmethod
    def add_action_heads(n_actions: int, value_head: torch.nn.Linear) -> torch.nn.Linear:
        """Takes a state value head and copies it to n_action state-action value heads.

        Args:
            n_actions (int): Size of the action space
            value_head (torch.nn.Linear): An (-1, 1) shaped linear layer.

        Returns:
            torch.nn.Linear: An (-1, n_actions) shaped linear layer with copied weights.
        """
        assert value_head.out_features == 1

        out = type(value_head)(value_head.in_features, n_actions)
        out.weight.data[:] = value_head.weight

        return out


def rollout(env: Env, policy: PhasicValueModel, timesteps: int) -> dict:
    roller = Roller(venv=env, act_fn=policy.act, initial_state=policy.initial_state(env.num))
    return roller.multi_step(nstep=timesteps)


def train_q_with_v(
    buffer: SarsDataset,
    q: QNetwork,
    v: Callable[[torch.Tensor], torch.Tensor],
    optim: torch.optim.Optimizer,
    n_epochs: int,
    batch_size: int,
) -> QNetwork:
    for _ in range(n_epochs):
        for states, actions, rewards, next_states in DataLoader(buffer, batch_size=batch_size):
            optim.zero_grad()
            q_pred = q.forward(states, actions)
            q_targ = rewards + q.discount_rate * v(next_states)
            loss = cast(torch.Tensor, (q_pred - q_targ) ** 2)
            loss.backward()
            optim.step()

    return q


def learn_q(
    env: Env,
    policy: PhasicValueModel,
    q_network: QNetwork,
    optim: torch.optim.Optimizer,
    env_interactions: int = 1_000_000,
    training_epochs: int = 10,
    batch_size: int = 64,
) -> QNetwork:
    trajs = rollout(env, policy, env_interactions)
    print(trajs.keys())
    print(trajs["finalstate"])
    print(trajs["ob"][0, -1])
    # print(trajs)
    exit()
    buffer = SarsDataset.from_dict(rollout(env, policy, env_interactions))

    train_q_with_v(
        buffer=buffer,
        q=q_network,
        v=policy.value,
        optim=optim,
        n_epochs=training_epochs,
        batch_size=batch_size,
    )

    return q_network


def refine(datadir: Path, lr: float = 10e-3, discount_rate: float = 0.99) -> None:
    datadir = Path(datadir)
    env = ProcgenGym3Env(1, "miner")
    env = ExtractDictObWrapper(env, "rgb")

    model = torch.load(get_model_path(datadir)[0], map_location=torch.device("cuda:0"))
    q = QNetwork(model, n_actions=16, discount_rate=discount_rate)

    optim = torch.optim.Adam(q.parameters(), lr=lr)

    learn_q(
        env=env,
        policy=model,
        q_network=q,
        optim=optim,
        training_epochs=2,
        batch_size=2,
        env_interactions=2000,
    )


if __name__ == "__main__":
    fire.Fire(refine)
