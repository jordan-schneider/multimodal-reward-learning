from typing import Tuple

import torch
from mrl.envs.util import make_env
from mrl.learn_values import QNetwork
from phasic_policy_gradient.train import make_model

MODEL_PATH = "tests/model.jd"
EPSILON = 1e-4


@torch.no_grad()
def model_weight_stats(model: torch.nn.Module) -> Tuple[float, float]:
    total = 0.0
    n_params = 0
    for param in model.parameters():
        total += torch.sum(param.data).item()
        n_params += param.numel()

    mean = total / n_params

    var = 0.0
    for param in model.parameters():
        var += torch.sum((param - mean) ** 2).item()

    return mean, var


def test_init():
    env = make_env(kind="miner", num=1, reward=0)
    model = make_model(env, arch="detach")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))

    random_init_q = QNetwork(model, n_actions=16, discount_rate=1.0)

    random_mean, random_var = model_weight_stats(random_init_q)

    assert abs(random_mean) < EPSILON

    value_init_q = QNetwork(model, n_actions=16, discount_rate=1.0, value_init=True)

    init_mean, init_var = model_weight_stats(value_init_q)

    assert abs(init_mean) > abs(random_mean)
    assert init_var > random_var


def test_forward():
    env = make_env(kind="miner", num=1, reward=0)
    model = make_model(env, arch="detach")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))

    obs = torch.ones((1, 64, 64, 3)).to(device=model.device)
    action = torch.ones((1, 1), dtype=torch.int64).to(device=model.device)

    random_init_q = QNetwork(model, n_actions=16, discount_rate=1.0)

    random_single_q = random_init_q(obs, action)
    random_qs = random_init_q.get_action_values(obs)

    assert random_single_q.shape == (1,)
    assert random_qs.shape == (1, 16)
    assert random_qs[:, 1] == random_single_q

    value_init_q = QNetwork(model, n_actions=16, discount_rate=1.0, value_init=True)

    value_single_q = value_init_q(obs, action)
    value_qs = value_init_q.get_action_values(obs)

    assert value_single_q.shape == (1,)
    assert value_qs.shape == (1, 16)
    assert value_qs[:, 1] == value_single_q
