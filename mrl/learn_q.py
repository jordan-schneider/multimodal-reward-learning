import logging
import pickle as pkl
from copy import deepcopy
from pathlib import Path
from typing import Callable, Literal, Optional, cast

import fire  # type: ignore
import numpy as np  # type: ignore
import torch
from gym3 import ExtractDictObWrapper  # type: ignore
from phasic_policy_gradient.ppg import PhasicValueModel
from procgen import ProcgenGym3Env
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange  # type: ignore

from mrl.offline_buffer import RLDataset
from mrl.online_batcher import BatchGenerator
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
        q_values = self.get_action_values(obs)
        out = q_values.gather(dim=1, index=action.view(-1, 1)).reshape(-1)
        return out

    def get_action_values(self, obs: torch.Tensor) -> torch.Tensor:
        assert obs.shape[1:] == (64, 64, 3)

        # ImpalaEncoder expects (batch, time, h, w, c)
        obs = obs.reshape((1, *obs.shape))

        z = self.enc.stateless_forward(obs)
        q_values = self.head(z)[0]
        return q_values

    def state_value(self, obs: torch.Tensor) -> torch.Tensor:
        q_values = self.get_action_values(obs)
        v, _ = q_values.max(dim=1)
        return v

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
    batch_gen: BatchGenerator,
    n_train_steps: int,
    batch_size: int,
    val_data: RLDataset,
    val_period: int,
    q: QNetwork,
    optim: torch.optim.Optimizer,
    writer: SummaryWriter,
) -> QNetwork:
    val_counter = 0
    val_step = 0
    train_step = 0
    n_batches = n_train_steps // batch_size
    for _ in trange(n_batches):
        states, actions, rewards, next_states = batch_gen.make_sars_batch(timesteps=batch_size)

        # n is not batch_size because batch_size actions generate batch_size - # dones - 1
        # usable transitions
        n = len(states)

        optim.zero_grad()
        q_pred = q.forward(states.to(device=q.device), actions.to(device=q.device)).cpu()

        v_next = q.state_value(next_states.to(q.device)).cpu()
        q_targ = rewards + q.discount_rate * v_next
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
                data=val_data,
                discount_rate=q.discount_rate,
                device=q.device,
            )
            writer.add_scalar("val/rmse", val_loss, global_step=val_step)
            val_step += 1

    return q


def learn_trunc_returns(
    horizon: int,
    batch_gen: BatchGenerator,
    n_train_steps: int,
    batch_size: int,
    val_data: RLDataset,
    val_period: int,
    q: QNetwork,
    optim: torch.optim.Optimizer,
    writer: SummaryWriter,
) -> QNetwork:
    val_counter = 0
    val_step = 0
    train_step = 0
    n_batches = n_train_steps // batch_size
    for _ in trange(n_batches):
        states, actions, partial_returns = batch_gen.make_trunc_return_batch(
            timesteps=batch_size, horizon=horizon, discount_rate=q.discount_rate
        )

        # n is not batch_size because batch_size actions generate batch_size - # dones - 1
        # usable transitions
        n = len(states)

        optim.zero_grad()
        q_pred = q.forward(states.to(device=q.device), actions.to(device=q.device)).cpu()
        assert q_pred.shape == (n,), f"q_pred={q_pred.shape} not expected ({n})"
        loss = torch.sum((q_pred - partial_returns) ** 2)

        writer.add_scalar("train/loss", loss, global_step=train_step)

        loss.backward()
        optim.step()

        train_step += 1
        val_counter += n

        if val_counter > val_period:
            val_counter = 0
            val_loss = eval_q_partial_rmse(
                q_fn=q.forward,
                data=val_data,
                k=horizon,
                discount_rate=q.discount_rate,
                device=q.device,
            )
            writer.add_scalar("val/rmse", val_loss, global_step=val_step)
            val_step += 1

    return q


def get_rollouts(
    env: ProcgenGym3Env,
    val_env_steps: int,
    policy: PhasicValueModel,
    datadir: Path,
    overwrite: bool,
) -> RLDataset:
    val_rollouts_path = datadir / "val_rollouts.pkl"

    val_data: Optional[RLDataset] = None

    if not overwrite and val_rollouts_path.exists():
        val_data = cast(RLDataset, pkl.load(val_rollouts_path.open("rb")))

        val_missing = val_env_steps - len(val_data.states)
    else:
        val_missing = val_env_steps

    if val_missing > 0:
        states, actions, rewards, firsts = procgen_rollout(env, policy, val_missing, tqdm=True)
        if val_data is not None:
            val_data.append_gym3(states, actions, rewards, firsts)
        else:
            val_data = RLDataset.from_gym3(states, actions, rewards, firsts)

        pkl.dump(val_data, open(datadir / "val_rollouts.pkl", "wb"))

    assert val_data is not None

    return val_data


def refine(
    indir: Path,
    outdir: Path,
    lr: float = 10e-3,
    discount_rate: float = 0.999,
    batch_size: int = 64,
    train_env_steps: int = 10_000_000,
    val_env_steps: int = 100_000,
    val_period: int = 2000 * 10,
    trunc_returns: bool = False,
    trunc_horizon: Optional[int] = None,
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    if trunc_returns:
        assert (
            trunc_horizon is not None
        ), f"Must specify a truncation horizon to use truncated returns."

    logging.basicConfig(level=verbosity)

    indir = Path(indir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    policy_path, policy_iter = find_policy_path(indir / "models")
    policy = torch.load(policy_path, map_location=torch.device("cuda:0"))

    model_outdir = outdir / "models"
    model_outdir.mkdir(parents=True, exist_ok=True)
    model_path = model_outdir / f"q_model_{policy_iter}.jd"

    if model_path.exists():
        logging.info(f"Loading Q model from {model_path}")
        q = cast(QNetwork, torch.load(model_path))
    else:
        q = QNetwork(policy, n_actions=16, discount_rate=discount_rate)

    env = ProcgenGym3Env(1, "miner")
    env = ExtractDictObWrapper(env, "rgb")

    val_data = get_rollouts(
        env=env, val_env_steps=val_env_steps, policy=policy, datadir=outdir, overwrite=overwrite
    )

    optim = torch.optim.Adam(q.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=outdir / "logs")

    if trunc_returns:
        q = learn_trunc_returns(
            horizon=trunc_horizon,
            batch_gen=BatchGenerator(env=env, policy=policy),
            n_train_steps=train_env_steps,
            batch_size=batch_size,
            val_data=val_data,
            val_period=val_period,
            q=q,
            optim=optim,
            writer=writer,
        )
    else:
        q = train_q_with_v(
            batch_gen=BatchGenerator(env=env, policy=policy),
            n_train_steps=train_env_steps,
            batch_size=batch_size,
            val_data=val_data,
            val_period=val_period,
            q=q,
            optim=optim,
            writer=writer,
        )

    torch.save(q, model_path)


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
        for i, reward in enumerate(reversed(rewards)):  # type: ignore
            current_return = current_return * discount_rate + reward
            returns[-i] = current_return

    return returns


@torch.no_grad()
def eval_q_rmse(
    q_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data: RLDataset,
    discount_rate: float,
    device: torch.device,
) -> float:
    loss = 0.0
    for states, actions, rewards in data.trajs(include_incomplete=False):
        values = q_fn(states[:-1].to(device=device), actions.to(device=device)).detach().cpu()
        returns = compute_returns(rewards.numpy(), discount_rate)[:-1]

        errors = values - returns
        loss += torch.sqrt(torch.mean(errors ** 2)).item()
    return loss


@torch.no_grad()
def eval_q_partial_rmse(
    q_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data: RLDataset,
    k: int,
    discount_rate: float,
    device: torch.device,
) -> float:
    states, actions, partial_returns = data.truncated_returns(
        horizon=k, discount_rate=discount_rate
    )

    loss = 0.0
    for state_batch, action_batch, return_batch in zip(
        np.array_split(states, len(states) // 100),
        np.array_split(actions, len(actions) // 100),
        np.array_split(partial_returns, len(partial_returns) // 100),
    ):
        values = q_fn(state_batch.to(device=device), action_batch.to(device=device)).detach().cpu()
        errors = values - return_batch
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
    logging.info("Gathering environment interactions")
    data = RLDataset.from_gym3(*procgen_rollout(env, policy, env_interactions, tqdm=True))
    pkl.dump(data, open(datadir / "eval_rollouts.pkl", "wb"))

    logging.info("Evaluating loss")
    loss = eval_q_rmse(q.forward, data, discount_rate, device=q.device)

    logging.info(f"Loss={loss} over {env_interactions} env timesteps.")


if __name__ == "__main__":
    fire.Fire({"refine": refine, "eval": eval})
