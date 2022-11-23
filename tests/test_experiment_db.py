import pickle as pkl
from pathlib import Path

from mrl.configs import SimulationExperimentConfig
from mrl.experiment_db.experiment import ExperimentDB


def test_insert():
    db = ExperimentDB(git_dir=Path("."))
    config = SimulationExperimentConfig()
    path = db.add(path=Path("/tmp/mrl-test"), config=config)

    metadata = db.get(path)
    assert metadata.config == config
    assert metadata.git_hash == open(".git/ORIG_HEAD", "r").readline().strip()
    db.remove(path)
