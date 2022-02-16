from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Sequence


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

    def check(self) -> None:
        children = list(self.rootdir.iterdir())
        if len(children) == 0:
            return

        schema_exhausted = False
        for root, dirs, files in os.walk(self.rootdir):
            path = Path(root)
            rel_path = path.relative_to(self.rootdir)
            if len(rel_path.parts) < len(self.schema) and len(files) > 0:
                logging.debug(f"{rel_path.parts=}, {self.schema=}")
                raise RuntimeError(
                    f"Found files {files} in non-leaf directory {rel_path}"
                )
            elif len(rel_path.parts) == len(self.schema):
                schema_exhausted = True
                if len(files) == 0:
                    raise RuntimeError(f"Found no files in leaf directory {root}")

        if not schema_exhausted:
            raise RuntimeError(f"Schema longer than actual depth")
