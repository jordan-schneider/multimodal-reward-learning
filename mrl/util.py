from pathlib import Path
from typing import Optional, Tuple


def get_model_path(modeldir: Path) -> Tuple[Optional[Path], int]:
    models = modeldir.glob("model[0-9][0-9][0-9].jd")
    if not models:
        return None, 0

    latest_model = sorted(models)[-1]
    start_time = int(str(latest_model)[-6:-3])
    return latest_model, start_time * 100_000
