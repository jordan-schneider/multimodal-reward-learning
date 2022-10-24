import pickle as pkl
from dataclasses import dataclass
from pathlib import Path

import arrow
import redis  # type: ignore
from mrl.configs import Config


class ExperimentDB:
    def __init__(self, git_dir: Path) -> None:
        self.db = redis.Redis()
        self.git_dir = git_dir

    @dataclass
    class Metadata:
        config: Config
        git_hash: str

    def add(self, path: Path, config: Config) -> Path:
        now = arrow.utcnow()
        fullpath = path.absolute()
        fullpath /= now.format("YYYY-MM-DD:HH:mm:ss")
        fullpath.mkdir(parents=True)
        git_hash = open(self.git_dir / ".git/ORIG_HEAD", "r").readline().strip()

        value = self.Metadata(config=config, git_hash=git_hash)

        self.db.set(str(fullpath), pkl.dumps(value))
        return fullpath

    def get(self, path: Path) -> Metadata:
        return pkl.loads(self.db.get(str(path)))

    def remove(self, path: Path) -> None:
        self.db.delete(str(path))
