import logging
from pathlib import Path

import fire  # type: ignore
from mrl.folders import HyperFolders


def main(rootdir: Path, old_schema: str, new_schema: str, defaults: str) -> None:
    logging.basicConfig(level="DEBUG")
    folders = HyperFolders(Path(rootdir), schema=old_schema.split(","))
    folders.update_schema(new_schema.split(","), defaults.split(","))
    folders.check()


if __name__ == "__main__":
    fire.Fire(main)
