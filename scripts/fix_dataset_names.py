from pathlib import Path
from typing import Any, Dict

import fire  # type: ignore

from mrl.inference.results import Results
from mrl.run_human_experiment import format_dataset_name


def main(experiment_dir: Path, dry_run: bool = True):
    experiment_dir = Path(experiment_dir)
    results = Results(experiment_dir / "trials", load_contents=True)
    for experiment_name in results.experiments.keys():
        print(f"Processing {experiment_name}")
        results.start(experiment_name)
        for key, value in results.experiments[experiment_name].items():
            if isinstance(value, dict):
                new_dict: Dict[str, Any] = {}
                for dataset_name, dataset_value in value.items():
                    tokens = dataset_name.split("_")

                    i = 0
                    while i < len(tokens):
                        if tokens[i] in ("short", "long"):
                            tokens[i] = f"{tokens[i]}_traj"
                            del tokens[i + 1]
                        i += 1

                    if not dry_run:
                        new_dict[format_dataset_name(tokens)] = dataset_value
                        results.update(name=key, value=new_dict)
                    else:
                        print(f"{dataset_name} -> {format_dataset_name(tokens)}")


if __name__ == "__main__":
    fire.Fire(main)
