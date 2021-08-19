from pathlib import Path
from typing import cast

import fire  # type: ignore
import torch
from phasic_policy_gradient.ppg import PhasicValueModel

from mrl.learn_q import QNetwork
from mrl.util import find_policy_path

# Manual patches to saved objects that are missing an attribute, or have an attribute
# that needs updating.

# STRONGLY prefer to regenrate the object if you can do so quickly. This is basically only
# for use with objects containing neural nets that are expensive to retrain.


def fix_policy_device(datadir: Path) -> None:
    """Adds a .device attribute to a PhasicValueModel"""
    datadir = Path(datadir)
    policy_path, _ = find_policy_path(datadir / "models")
    policy = cast(PhasicValueModel, torch.load(policy_path))
    policy.device = policy.pi_head.weight.device
    torch.save(policy, policy_path)


def fix_q_device(datadir: Path) -> None:
    datadir = Path(datadir)
    _, iter = find_policy_path(datadir / "models")
    q_path = datadir / f"models/q_model_{iter}.jd"
    q = cast(QNetwork, torch.load(q_path))
    q.device = q.head.weight.device
    torch.save(q, q_path)


if __name__ == "__main__":
    fire.Fire()
