from __future__ import annotations

import logging
import pickle as pkl
import sqlite3
from pathlib import Path
from typing import Dict, Sequence, TypeVar, Union, cast

import numpy as np
from experiment_server.type import DataModality, State
from linear_procgen import Miner, make_env
from linear_procgen.util import get_root_env

from mrl.configs import FixedInference, HumanExperimentConfig
from mrl.dataset.human_responses import UserDataset
from mrl.experiment_db.experiment import ExperimentDB
from mrl.inference.analysis import analysis
from mrl.inference.plots import plot_comparison
from mrl.inference.posterior import cover_sphere, make_likelihoods
from mrl.inference.results import Results
from mrl.reward_model.boltzmann import boltzmann_likelihood
from mrl.reward_model.hinge import hinge_likelihood
from mrl.reward_model.likelihood import Likelihood
from mrl.util import get_normalized_diff, setup_logging


def main() -> None:
    """Runs a reward inference experiment using human preferences."""

    config = HumanExperimentConfig()
    config.validate()

    experiment_db = ExperimentDB(git_dir=Path(config.git_dir))
    inference_outdir = experiment_db.add(Path(config.rootdir) / "inference", config)
    setup_logging(level=config.verbosity, outdir=inference_outdir, force=True)
    config.dump(inference_outdir)

    rng = np.random.default_rng(seed=config.seed)
    pairs_by_modality = cast(
        Dict[str, np.ndarray],
        get_feature_pairs(
            response_dir=Path(config.rootdir) / "users",
            question_db_path=Path(config.question_db_path),
        ),
    )
    pairs_by_modality = trim_by_modality(pairs_by_modality)
    pairs_by_modality["joint"] = get_joint(pairs_by_modality)
    pairs_by_modality = {
        k: v[rng.permutation(v.shape[0])] for k, v in pairs_by_modality.items()
    }
    diffs_by_modality = {
        k: get_normalized_diff(pair, mode=config.norm_mode)
        for k, pair in pairs_by_modality.items()
    }

    results = Results(inference_outdir / "trials")
    results.start("")

    tmp_env = make_env(name=config.env.name, num=1, reward=1)
    root_env = get_root_env(tmp_env)
    env_features = root_env.get_features()[0].shape[0]

    reward_samples = cover_sphere(
        n_samples=config.inference.reward_particles,
        ndims=env_features,
        rng=rng,
    )
    results.update("reward_sample", reward_samples)

    # TODO: Implement gamma temperature prior instead of using fixed for everything.
    assert isinstance(config.inference.noise, FixedInference)
    temps = {k: config.inference.noise.temp for k in diffs_by_modality.keys()}
    results.update("temp", temps)

    logging.info("Starting inference on human prefs")
    results.start("human")
    results = make_likelihoods(
        reward_samples=reward_samples,
        diffs=diffs_by_modality,
        reward_likelihood=get_likelihood_fn(config),
        use_shift=config.inference.use_shift,
        results=results,
        temps=temps,
        save_all=config.inference.save_all,
    )
    results.close()

    logging.info("Computing post-inference metrics")
    analysis(results, compute_centroids=True, compute_mean_dispersions=True)

    logging.info("Plotting metrics")
    results.start("human")
    plotdir = inference_outdir / "trials" / "human" / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)
    plot_comparison(results, plotdir)


def get_joint(diffs_by_modality: dict[str, np.ndarray]) -> np.ndarray:
    n_modalities = len(diffs_by_modality)
    n_diffs = len(diffs_by_modality.values().__iter__().__next__())
    return np.concatenate(
        [
            per_modality[: n_diffs // n_modalities]
            for per_modality in diffs_by_modality.values()
        ]
    )


def trim_by_modality(diffs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Returns an equal number of diffs for each modality, that number being the minimum across all modalities.

    Args:
        diffs (dict[str, np.ndarray]): (n_i, feature_dim) array of differences per data modality

    Returns:
        dict[str, np.ndarray]: (min_i(n_i), feature_dim) array of differences per data modality
    """
    min_len = min(arr.shape[0] for arr in diffs.values())
    return {modality: arr[:min_len] for modality, arr in diffs.items()}


def get_likelihood_fn(config: HumanExperimentConfig) -> Likelihood:
    if config.inference.likelihood_fn == "hinge":
        return hinge_likelihood
    elif config.inference.likelihood_fn == "boltzmann":
        return boltzmann_likelihood
    raise ValueError(f"Unknown likelihood: {config.inference.likelihood_fn}")


def get_feature_pairs(
    response_dir: Path, question_db_path: Path
) -> dict[DataModality, np.ndarray]:
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
    diffs = [
        (np.stack((left, right)), modality) for _, left, right, modality in features
    ]
    return sort_by_modality(diffs)


T = TypeVar("T")


def sort_by_modality(
    vals: Sequence[tuple[np.ndarray, DataModality]]
) -> dict[DataModality, np.ndarray]:
    by_modality: dict[DataModality, list[np.ndarray]] = {}
    for val, modality in vals:
        current_modality = by_modality.get(modality, [])
        current_modality.append(val)
        by_modality[modality] = current_modality

    out = {modality: np.stack(arrs, axis=0) for modality, arrs in by_modality.items()}

    return out


def get_features(
    questions: dict[
        int,
        tuple[
            tuple[State, Sequence[int], DataModality],
            tuple[State, Sequence[int], DataModality],
        ],
    ],
    max_lens: list[tuple[int, tuple[int, int]]],
) -> list[tuple[int, np.ndarray, np.ndarray, DataModality]]:
    # TODO: We start assuming here that the data modalities are the same
    out = []
    # TODO: This is inefficient, we should really only roll out each question once, but it makes the logic simpler and
    # probably isn't the bottleneck.
    for question_id, (left_max, right_max) in max_lens:
        (left_state, left_actions, left_modality), (
            right_state,
            right_actions,
            right_modality,
        ) = questions[question_id]
        out.append(
            (
                question_id,
                total_feature_rollout(left_state, left_actions, max_len=left_max),
                total_feature_rollout(right_state, right_actions, max_len=right_max),
                left_modality,
            )
        )
    return out


def get_questions_from_db(
    db: sqlite3.Connection, question_ids: set[int]
) -> dict[
    int,
    tuple[
        tuple[State, Sequence[int], DataModality],
        tuple[State, Sequence[int], DataModality],
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
    left_traj.start_state as left_state,
    left_traj.actions as left_actions,
    left_traj.modality as left_modality,
    right_traj.start_state as right_state,
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
            (pkl.loads(left_state), pkl.loads(left_actions), left_modality),
            (pkl.loads(right_state), pkl.loads(right_actions), right_modality),
        )
        for question_id, left_state, left_actions, left_modality, right_state, right_actions, right_modality in cursor
    }
    return out


def total_feature_rollout(
    start_state: State, actions: Union[Sequence[int], np.ndarray], max_len: int
) -> np.ndarray:
    env = make_env("miner", 1, 0)
    root_env = get_root_env(env)

    if isinstance(start_state.grid_shape, np.ndarray):
        start_state.grid_shape = start_state.grid_shape.astype(dtype=int)
    if isinstance(start_state.agent_pos, np.ndarray):
        start_state.agent_pos = start_state.agent_pos.astype(dtype=int)
    if isinstance(start_state.exit_pos, np.ndarray):
        start_state.exit_pos = start_state.exit_pos.astype(dtype=int)

    root_env.set_miner_state(
        [
            Miner.State(
                grid_size=(
                    int(start_state.grid_shape[0]),
                    int(start_state.grid_shape[1]),
                ),
                grid=start_state.grid,
                agent_pos=(
                    int(start_state.agent_pos[0]),
                    int(start_state.agent_pos[1]),
                ),
                exit_pos=(int(start_state.exit_pos[0]), int(start_state.exit_pos[1])),
            )
        ]
    )

    total_features = root_env.get_features()[0]

    for action in actions[:max_len]:
        if not isinstance(action, np.ndarray):
            action = np.array([action])
        env.act(action)
        total_features += root_env.get_features()[0]

    return total_features


if __name__ == "__main__":
    main()
