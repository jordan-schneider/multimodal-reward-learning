import numpy as np
from hypothesis import given
from hypothesis.strategies import integers
from question_gen.biyik import infogain, successive_elimination
from question_gen.util import remove_duplicates

from .strategies import halfplanes_strategy, rewards_strategy


@given(rewards=rewards_strategy(), halfplanes=halfplanes_strategy())
def test_infogain_even(rewards: np.ndarray, halfplanes: np.ndarray):
    forward_infogain = infogain(halfplanes, rewards)
    backward_infogain = infogain(-halfplanes, rewards)
    assert np.allclose(forward_infogain, backward_infogain)

    backward_infogain = infogain(halfplanes, -rewards)
    assert np.allclose(forward_infogain, backward_infogain)


@given(rewards=rewards_strategy(), halfplanes=halfplanes_strategy())
def test_infogain_nonnegative(rewards: np.ndarray, halfplanes: np.ndarray):
    infogains = infogain(halfplanes, rewards)
    assert np.all(infogains >= 0)


@given(rewards=rewards_strategy(), questions=halfplanes_strategy())
def test_successive_elimination_noop(rewards: np.ndarray, questions: np.ndarray):
    n_questions = questions.shape[0]
    indices = successive_elimination(
        reward_samples=rewards,
        question_samples=questions,
        n_out_questions=n_questions,
        initial_questions=n_questions,
    )
    final_questions = questions[np.sort(indices)]
    assert np.array_equal(questions, final_questions)


@given(
    rewards=rewards_strategy(),
    questions=halfplanes_strategy(n_halfplanes=integers(min_value=3, max_value=10)),
    n_out_questions=integers(min_value=1, max_value=10),
)
def test_succesive_elimination_output_size(
    rewards: np.ndarray, questions: np.ndarray, n_out_questions: int
):
    questions, _ = remove_duplicates(questions)
    if questions.shape[0] < 3:
        return
    n_out_questions = min(n_out_questions, questions.shape[0] - 2)
    indices = successive_elimination(
        reward_samples=rewards,
        question_samples=questions,
        n_out_questions=n_out_questions,
        initial_questions=n_out_questions + 1,
    )
    assert n_out_questions == indices.shape[0]
