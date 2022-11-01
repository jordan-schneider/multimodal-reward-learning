from __future__ import annotations

import pickle as pkl
import sqlite3
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Sequence, Set, Tuple, TypeVar

import arrow
import numpy as np
import yaml
from experiment_server.type import DataModality
from linear_procgen import make_env
from linear_procgen.util import get_root_env

from mrl.configs import FixedInference, HumanExperimentConfig
from mrl.dataset.human_responses import UserDataset
from mrl.experiment_db.experiment import ExperimentDB
from mrl.inference.posterior import cover_sphere, make_likelihoods
from mrl.inference.results import Results
from mrl.reward_model.boltzmann import boltzmann_likelihood
from mrl.reward_model.hinge import hinge_likelihood
from mrl.reward_model.likelihood import Likelihood
from mrl.util import normalize_diffs, setup_logging


def main() -> None:
    """Runs a reward inference experiment using human preferences."""

    config = HumanExperimentConfig()
    config.validate()

    experiment_db = ExperimentDB(git_dir=Path(config.git_dir))
    inference_outdir = experiment_db.add(Path(config.rootdir) / "inference", config)
    setup_logging(level=config.verbosity, outdir=inference_outdir, force=True)
    config.dump(inference_outdir)

    rng = np.random.default_rng(seed=config.seed)
    diffs_by_modality = get_diffs(
        response_dir=Path(config.rootdir) / "users",
        question_db_path=Path(config.question_db_path),
    )
    diffs: Dict[str, np.ndarray] = trim_diffs(diffs_by_modality)  # type: ignore
    diffs["joint"] = get_joint_diffs(diffs_by_modality)
    diffs = {k: v[rng.permutation(v.shape[0])] for k, v in diffs.items()}
    diffs = {
        k: normalize_diffs(diff, mode=config.norm_mode) for k, diff in diffs.items()
    }

    results = Results(inference_outdir / "trials")

    tmp_env = make_env(config.env.name, 1)
    env_features = tmp_env.get_info()["features"].shape[0]

    reward_samples = cover_sphere(
        n_samples=config.inference.reward_particles,
        ndims=env_features,
        rng=rng,
    )

    # TODO: Implement gamma temperature prior instead of using fixed for everything.
    assert isinstance(config.inference.noise, FixedInference)
    temps = {k: config.inference.noise.temp for k in diffs.keys()}

    results = make_likelihoods(
        reward_samples=reward_samples,
        diffs=diffs,
        reward_likelihood=get_likelihood_fn(config),
        use_shift=config.inference.use_shift,
        results=results,
        temps=temps,
        save_all=config.inference.save_all,
    )


def get_joint_diffs(diffs_by_modality: Dict[DataModality, np.ndarray]) -> np.ndarray:
    n_modalities = len(diffs_by_modality)
    n_diffs = len(diffs_by_modality.values().__iter__().__next__())
    return np.concatenate(
        [
            per_modality[: n_diffs // n_modalities]
            for per_modality in diffs_by_modality.values()
        ]
    )


def trim_diffs(diffs: Dict[DataModality, np.ndarray]) -> Dict[DataModality, np.ndarray]:
    """Returns an equal number of diffs for each modality, that number being the minimum across all modalities.

    Args:
        diffs (Dict[DataModality, np.ndarray]): (n_i, feature_dim) array of differences per data modality

    Returns:
        Dict[DataModality, np.ndarray]: (min_i(n_i), feature_dim) array of differences per data modality
    """
    min_len = min(arr.shape[0] for arr in diffs.values())
    return {modality: arr[:min_len] for modality, arr in diffs.items()}


def get_likelihood_fn(config: HumanExperimentConfig) -> Likelihood:
    if config.inference.likelihood_fn == "hinge":
        return hinge_likelihood
    elif config.inference.likelihood_fn == "boltzmann":
        return boltzmann_likelihood
    raise ValueError(f"Unknown likelihood: {config.inference.likelihood_fn}")


def get_diffs(
    response_dir: Path, question_db_path: Path
) -> Dict[DataModality, np.ndarray]:
    prefs = UserDataset(response_dir)
    question_ids = set(
        resp.question_id for user in prefs.users for resp in user.responses
    )
    db = sqlite3.connect(question_db_path)
    questions = get_questions_from_db(db, question_ids)

    features = get_features(
        questions,
        max_lens=[
            (resp.question_id, resp.steps)
            for user in prefs.users
            for resp in user.responses
        ],
    )
    diffs = [(left - right, modality) for _, left, right, modality in features]
    return sort_by_modality(diffs)


T = TypeVar("T")


def sort_by_modality(
    vals: Sequence[Tuple[np.ndarray, DataModality]]
) -> Dict[DataModality, np.ndarray]:
    by_modality: Dict[DataModality, List[np.ndarray]] = {}
    for val, modality in vals:
        current_modality = by_modality.get(modality, [])
        current_modality.append(val)
        by_modality[modality] = current_modality

    out = {modality: np.stack(arrs, axis=0) for modality, arrs in by_modality.items()}

    return out


def get_features(
    questions: Dict[
        int,
        Tuple[
            Tuple[bytes, Sequence[int], DataModality],
            Tuple[bytes, Sequence[int], DataModality],
        ],
    ],
    max_lens: List[Tuple[int, Tuple[int, int]]],
) -> List[Tuple[int, np.ndarray, np.ndarray, DataModality]]:
    # TODO: We start assuming here that the data modalities are the same
    out = []
    # TODO: This is inefficient, we should really only roll out each question once, but it makes the logic simpler and
    # probably isn't the bottleneck.
    for question_id, (left_max, right_max) in max_lens:
        (left_cstates, left_actions, left_modality), (
            right_cstates,
            right_actions,
            right_modality,
        ) = questions[question_id]
        out.append(
            (
                question_id,
                total_feature_rollout(left_cstates, left_actions, max_len=left_max),
                total_feature_rollout(right_cstates, right_actions, max_len=right_max),
                left_modality,
            )
        )
    return out


def get_questions_from_db(
    db: sqlite3.Connection, question_ids: Set[int]
) -> Dict[
    int,
    Tuple[
        Tuple[bytes, Sequence[int], DataModality],
        Tuple[bytes, Sequence[int], DataModality],
    ],
]:
    question_id_list = ", ".join(f":question_id_{i}" for i in range(len(question_ids)))
    question_id_values = {
        f"question_id_{i}": question_id for i, question_id in enumerate(question_ids)
    }
    cursor = db.execute(
        f"""
SELECT
    questions.id,
    left_traj.cstates as left_cstates,
    left_traj.actions as left_actions,
    left_traj.modality as left_modality,
    right_traj.cstates as right_cstates,
    right_traj.actions as right_actions,
    right_traj.modality as right_modality
FROM 
    (SELECT * FROM questions WHERE id IN ({question_id_list})) as questions
    INNER JOIN trajectories AS left_traj
        ON questions.first_id = left_traj.id
        INNER JOIN trajectories AS right_traj
            ON questions.second_id = right_traj.id;""",
        question_id_values,
    )

    out = {
        question_id: (
            (left_cstate, pkl.loads(left_actions), left_modality),
            (right_cstate, pkl.loads(right_actions), right_modality),
        )
        for question_id, left_cstate, left_actions, left_modality, right_cstate, right_actions, right_modality in cursor
    }
    import pdb

    pdb.set_trace()
    return out


def total_feature_rollout(
    start_cstate: bytes, actions: Sequence[int], max_len: int
) -> np.ndarray:
    env = make_env("miner", 1, 0)
    root_env = get_root_env(env)
    root_env.set_state([start_cstate])

    total_features = env.get_info()["features"]

    for action in actions[:max_len]:
        env.step(action)
        total_features += env.get_info()["features"]

    return total_features


if __name__ == "__main__":
    main()
