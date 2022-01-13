import numpy as np
import torch
from gym3.types import ValType  # type: ignore
from phasic_policy_gradient.ppg import PhasicValueModel
from torch.distributions import Categorical


class RandomPolicy(PhasicValueModel):
    def __init__(self, actype: ValType, num: int):
        self.actype = actype
        self.act_dist = Categorical(
            probs=torch.ones(actype.eltype.n) / np.prod(actype.eltype.n)
        )
        self.device = torch.device("cpu")
        self.num = num

    def act(self, ob, first: bool, state_in):
        return self.act_dist.sample((self.num,)), None, None

    def initial_state(self, batchsize: int):
        return None
