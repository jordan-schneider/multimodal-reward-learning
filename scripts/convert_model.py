import os
from pathlib import Path

import fire  # type: ignore
import torch
from mrl.learn_values import QNetwork
from phasic_policy_gradient.ppg import PhasicValueModel


def main(rootdir: Path) -> None:
    for file in Path(rootdir).rglob("*.jd"):
        model = torch.load(file, map_location="cpu")
        if isinstance(model, torch.nn.Module):
            print(file)
            torch.save(model.state_dict(), file)
            os.rename(file, file.with_suffix(file.suffix + ".old"))


if __name__ == "__main__":
    fire.Fire(main)
