from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence


class HyperFolders:
    def __init__(self, rootdir: Path, schema: Sequence[str]) -> None:
        assert len(schema) > 0
        self.rootdir = rootdir
        self.schema = schema

        self.rootdir.mkdir(parents=True, exist_ok=True)

        self.check()

    def add_experiment(self, hyper_values: Dict[str, Any]) -> Path:
        exp_dir = self.rootdir
        for param in self.schema:
            exp_dir /= str(hyper_values[param])
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def update_schema(self, new_schema: Sequence[str], defaults: Sequence[Any]) -> None:
        assert len(new_schema) == len(defaults)

        out = Path("/tmp/new_folders")
        out.mkdir(parents=True, exist_ok=True)
        for child in out.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)

        for root, dirs, files in os.walk(self.rootdir):
            old_path = Path(root).relative_to(self.rootdir)
            if len(old_path.parts) == len(self.schema):
                new_path = out
                for hyper, default in zip(new_schema, defaults):
                    if hyper in self.schema:
                        new_path /= old_path.parts[self.schema.index(hyper)]
                    else:
                        new_path /= str(default)
                new_path.mkdir(parents=True, exist_ok=True)
                shutil.copytree(self.rootdir / old_path, new_path, dirs_exist_ok=True)
        shutil.rmtree(self.rootdir)
        out.rename(self.rootdir)
        self.schema = new_schema

    @staticmethod
    def __get_child_folders(path: Path) -> List[Path]:
        out = []
        for child in path.iterdir():
            if child.is_dir():
                out.append(child)
        return out

    def check(self) -> None:
        children = list(self.rootdir.iterdir())
        if len(children) == 0:
            return

        current = children[0]
        for i, _ in enumerate(self.schema[1:]):
            children = list(current.iterdir())
            if len(children) == 0:
                raise RuntimeError(
                    f"Schema longer than actual folder depth. No children at {current}."
                )

            # All children must be files
            for child in children:
                if not child.is_dir():
                    raise RuntimeError(
                        f"Rootdir {self.rootdir} exists but file {child} exists before end of schema"
                    )
            current = children[0]
        # At least one child must be a file
        children = list(current.iterdir())

        any_files = False
        for child in children:
            if child.is_file():
                any_files = True
                break
        if not any_files:
            raise RuntimeError(
                f"Schema exhausted at {current} but further folders exist"
            )
        return
