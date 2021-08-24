from __future__ import annotations

from typing import Tuple

import torch
from gym3 import Env  # type: ignore
from phasic_policy_gradient.ppg import PhasicValueModel

from mrl.offline_buffer import RLDataset, SarsDataset
from mrl.util import procgen_rollout


class BatchGenerator:
    def __init__(self, env: Env, policy: PhasicValueModel) -> None:
        self.env = env
        self.policy = policy

    def make_sars_batch(
        self, timesteps: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = SarsDataset.from_gym3(*procgen_rollout(self.env, self.policy, timesteps))
        return data.make_sars()

    def make_trunc_return_batch(
        self, horizon: int, timesteps: int, discount_rate: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = RLDataset.from_gym3(*procgen_rollout(self.env, self.policy, timesteps))
        return data.truncated_returns(horizon=horizon, discount_rate=discount_rate)
