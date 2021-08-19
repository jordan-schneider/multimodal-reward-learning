import pickle
from copy import deepcopy
from pathlib import Path
from typing import Callable, cast

import fire  # type: ignore
import numpy as np  # type: ignore
import torch
import tqdm  # type: ignore
from gym3 import Env  # type: ignore
from gym3 import ExtractDictObWrapper
from phasic_policy_gradient.ppg import PhasicValueModel
from procgen import ProcgenGym3Env
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mrl.offline_buffer import RLDataset, SarsDataset
from mrl.util import find_policy_path, procgen_rollout


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

        self.device = policy.device

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        assert obs.shape[1:] == (64, 64, 3)

        # ImpalaEncoder expects (batch, time, h, w, c)
        obs = obs.reshape((1, *obs.shape))

        z = self.enc.stateless_forward(obs)
        q_values = self.head(z)
        out = q_values[0].gather(1, action.view(-1, 1)).reshape(-1)
        return out

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
        out = out.to(value_head.weight.device)

        return out


def train_q_with_v(
    train_buffer: SarsDataset,
    val_buffer: RLDataset,
    q: QNetwork,
    v: Callable[[torch.Tensor], torch.Tensor],
    optim: torch.optim.Optimizer,
    n_epochs: int,
    batch_size: int,
    val_period: int,
    writer: SummaryWriter,
) -> QNetwork:
    val_counter = 0
    val_step = 0

    train_step = 0
    for epoch in range(n_epochs):
        with tqdm(DataLoader(train_buffer, batch_size=batch_size), unit="batch") as tepoch:
            for states, actions, rewards, next_states in tepoch:
                tepoch.set_description(f"Epoch {epoch}/{n_epochs}")

                n = len(states)

                optim.zero_grad()
                q_pred = q.forward(states, actions)
                q_targ = rewards + q.discount_rate * v(next_states).reshape(-1)
                assert q_pred.shape == (n,), f"q_pred={q_pred.shape} not expected ({n})"
                assert q_targ.shape == (n,), f"q_targ={q_targ.shape} not expected ({n})"
                loss = torch.sum((q_pred - q_targ) ** 2)

                writer.add_scalar("train/loss", loss, global_step=train_step)

                loss.backward()
                optim.step()

                train_step += 1
                val_counter += n

                if val_counter > val_period:
                    val_counter = 0
                    val_loss = eval_q_rmse(
                        q_fn=q.forward,
                        data=val_buffer,
                        discount_rate=q.discount_rate,
                        device=q.device,
                    )
                    writer.add_scalar("val/rmse", val_loss, global_step=val_step)
                    val_step += 1

    return q


def learn_q(
    env: Env,
    policy: PhasicValueModel,
    q_network: QNetwork,
    optim: torch.optim.Optimizer,
    writer: SummaryWriter,
    train_env_steps: int,
    val_env_steps: int,
    training_epochs: int,
    batch_size: int,
    val_period: int,
    outdir: Path,
) -> QNetwork:
    train_buffer = SarsDataset.from_rl_dataset(procgen_rollout(env, policy, train_env_steps))

    pickle.dump(train_buffer, open(outdir / "train_rollouts.pkl", "wb"))

    val_buffer = SarsDataset.from_rl_dataset(procgen_rollout(env, policy, val_env_steps))

    pickle.dump(val_buffer, open(outdir / "val_rollouts.pkl", "wb"))

    q_network = train_q_with_v(
        train_buffer=train_buffer,
        val_buffer=val_buffer,
        q=q_network,
        v=lambda x: policy.value(x.reshape(1, *x.shape))[0],  # V expects (batch, time, ...)
        optim=optim,
        n_epochs=training_epochs,
        batch_size=batch_size,
        writer=writer,
        val_period=val_period,
    )

    return q_network


def refine(
    datadir: Path,
    lr: float = 10e-3,
    discount_rate: float = 0.999,
    training_epochs: int = 10,
    batch_size: int = 64,
    train_env_steps: int = 1_000_000,
    val_env_steps: int = 100_000,
    val_period: int = 2000 * 10,
) -> None:
    datadir = Path(datadir)
    env = ProcgenGym3Env(1, "miner")
    env = ExtractDictObWrapper(env, "rgb")

    policy_path, policy_iter = find_policy_path(datadir / "models")
    model = torch.load(policy_path, map_location=torch.device("cuda:0"))
    q = QNetwork(model, n_actions=16, discount_rate=discount_rate)

    optim = torch.optim.Adam(q.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=datadir / "logs")

    q = learn_q(
        env=env,
        policy=model,
        q_network=q,
        optim=optim,
        training_epochs=training_epochs,
        batch_size=batch_size,
        train_env_steps=train_env_steps,
        val_env_steps=val_env_steps,
        val_period=val_period,
        writer=writer,
        outdir=datadir,
    )

    model_number = policy_iter // 100_000
    torch.save(q, datadir / f"q_model_{model_number}.jd")


def compute_returns(
    rewards: np.ndarray, discount_rate: float, use_conv: bool = False
) -> np.ndarray:
    assert discount_rate >= 0.0 and discount_rate <= 1.0

    # TODO(joschnei): Benchmark this
    if use_conv:
        discounts = np.array([discount_rate ** i for i in range(len(rewards))])
        # TODO(joschnei): There must be a better way to do a 1d vector convolution
        returns = np.array([rewards[i:] @ discounts[:-i] for i in range(len(rewards))])
    else:
        returns = np.empty(len(rewards))
        current_return = 0
        for i, reward in enumerate(reversed(rewards)):
            current_return = current_return * discount_rate + reward
            returns[-i] = current_return

    return returns


def eval_q_rmse(
    q_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data: RLDataset,
    discount_rate: float,
    device: torch.device,
) -> float:
    loss = 0.0
    for states, actions, rewards in data.trajs(include_incomplete=False):
        values = (
            q_fn(torch.tensor(states[:-1], device=device), torch.tensor(actions, device=device))
            .detach()
            .cpu()
        )
        returns = compute_returns(rewards.numpy(), discount_rate)[:-1]

        errors = values - returns
        loss += torch.sqrt(torch.mean(errors ** 2)).item()
    return loss


def eval(datadir: Path, discount_rate: float = 0.999, env_interactions: int = 1_000_000) -> None:
    datadir = Path(datadir)
    policy_path, iter = find_policy_path(datadir / "models")
    q_path = datadir / f"q_model_{iter}.jd"

    policy = cast(PhasicValueModel, torch.load(policy_path))
    q = cast(QNetwork, torch.load(q_path))

    env = ProcgenGym3Env(1, "miner")
    env = ExtractDictObWrapper(env, "rgb")
    print("Gathering environment interactions")
    data = procgen_rollout(env, policy, env_interactions)
    pickle.dump(data, open(datadir / "eval_rollouts.pkl", "wb"))

    print("Evaluating loss")
    loss = eval_q_rmse(q.forward, data, discount_rate, device=q.device)

    print(f"Loss={loss} over {env_interactions} env timesteps.")


if __name__ == "__main__":
    fire.Fire({"refine": refine, "eval": eval})
