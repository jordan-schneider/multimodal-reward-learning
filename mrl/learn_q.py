import pickle as pkl
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional, Tuple, cast

import fire  # type: ignore
import numpy as np  # type: ignore
import torch
from gym3 import Env  # type: ignore
from gym3 import ExtractDictObWrapper
from phasic_policy_gradient.ppg import PhasicValueModel
from procgen import ProcgenGym3Env
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # type: ignore

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
    train_data: SarsDataset,
    val_data: RLDataset,
    q: QNetwork,
    optim: torch.optim.Optimizer,
    n_epochs: int,
    batch_size: int,
    val_period: int,
    writer: SummaryWriter,
    v: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> QNetwork:
    val_counter = 0
    val_step = 0
    train_step = 0
    for epoch in range(n_epochs):
        with tqdm(DataLoader(dataset=train_data, batch_size=batch_size), unit="batch") as tepoch:
            for states, actions, rewards, next_states in tepoch:
                tepoch.set_description(f"Epoch {epoch}/{n_epochs}")

                n = len(states)

                optim.zero_grad()
                q_pred = q.forward(states.to(device=q.device), actions.to(device=q.device)).cpu()

                if v is not None:
                    v_next = v(next_states.to(q.device)).reshape(-1)
                else:
                    q_values = q.get_action_values(next_states.to(q.device))
                    v_next, _ = q_values.max(dim=1)
                    v_next = v_next.cpu()

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


def get_rollouts(
    datadir: Path,
    train_env_steps: int,
    val_env_steps: int,
    policy: PhasicValueModel,
    overwrite: bool,
) -> Tuple[SarsDataset, RLDataset]:
    train_rollouts_path = datadir / "train_rollouts.pkl"
    val_rollouts_path = datadir / "val_rollouts.pkl"

    train_data: Optional[SarsDataset] = None
    val_data: Optional[RLDataset] = None

    if not overwrite and train_rollouts_path.exists() and val_rollouts_path.exists():
        train_data = cast(SarsDataset, pkl.load(train_rollouts_path.open("rb")))
        val_data = cast(RLDataset, pkl.load(val_rollouts_path.open("rb")))

        train_missing = train_env_steps - len(train_data.states)
        val_missing = val_env_steps - len(val_data.states)
    else:
        train_missing = train_env_steps
        val_missing = val_env_steps

    if train_missing > 0 or val_missing > 0:
        env = ProcgenGym3Env(1, "miner")
        env = ExtractDictObWrapper(env, "rgb")
        if train_missing > 0:
            print(train_missing)
            states, actions, rewards, firsts = procgen_rollout(env, policy, train_missing)
            if train_data is not None:
                train_data.append_gym3(states, actions, rewards, firsts)
            else:
                train_data = SarsDataset.from_gym3(states, actions, rewards, firsts)

            pkl.dump(train_data, open(datadir / "train_rollouts.pkl", "wb"))
        if val_missing > 0:
            print(val_missing)
            states, actions, rewards, firsts = procgen_rollout(env, policy, val_missing)
            if val_data is not None:
                val_data.append_gym3(states, actions, rewards, firsts)
            else:
                val_data = RLDataset.from_gym3(states, actions, rewards, firsts)

            pkl.dump(val_data, open(datadir / "val_rollouts.pkl", "wb"))

    assert train_data is not None
    assert val_data is not None

    return train_data, val_data


def refine(
    indir: Path,
    outdir: Path,
    lr: float = 10e-3,
    discount_rate: float = 0.999,
    training_epochs: int = 10,
    batch_size: int = 64,
    train_env_steps: int = 1_000_000,
    val_env_steps: int = 100_000,
    val_period: int = 2000 * 10,
    use_v_next_state: bool = False,
    overwrite: bool = False,
) -> None:
    indir = Path(indir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    policy_path, policy_iter = find_policy_path(indir / "models")
    policy = torch.load(policy_path, map_location=torch.device("cuda:0"))
    q = QNetwork(policy, n_actions=16, discount_rate=discount_rate)

    train_data, val_data = get_rollouts(outdir, train_env_steps, val_env_steps, policy, overwrite)

    optim = torch.optim.Adam(q.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=outdir / "logs")

    if use_v_next_state:
        # V expects (batch, time, ...)
        v: Optional[Callable] = lambda x: policy.value(x.reshape(1, *x.shape))[0].cpu()
    else:
        v = None

    q = train_q_with_v(
        train_data=train_data,
        val_data=val_data,
        q=q,
        v=v,
        optim=optim,
        n_epochs=training_epochs,
        batch_size=batch_size,
        writer=writer,
        val_period=val_period,
    )

    torch.save(q, outdir / f"models/q_model_{policy_iter}.jd")


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


def eval(datadir: Path, discount_rate: float = 0.999, env_interactions: int = 1_000_000) -> None:
    datadir = Path(datadir)
    policy_path, iter = find_policy_path(datadir / "models")
    q_path = datadir / f"q_model_{iter}.jd"

    policy = cast(PhasicValueModel, torch.load(policy_path))
    q = cast(QNetwork, torch.load(q_path))

    env = ProcgenGym3Env(1, "miner")
    env = ExtractDictObWrapper(env, "rgb")
    print("Gathering environment interactions")
    data = RLDataset.from_gym3(*procgen_rollout(env, policy, env_interactions))
    pkl.dump(data, open(datadir / "eval_rollouts.pkl", "wb"))

    print("Evaluating loss")
    loss = eval_q_rmse(q.forward, data, discount_rate, device=q.device)

    print(f"Loss={loss} over {env_interactions} env timesteps.")


if __name__ == "__main__":
    fire.Fire({"refine": refine, "eval": eval})
