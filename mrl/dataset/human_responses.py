from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arrow


@dataclass
class User:
    responses: list[Response]
    interact_times: tuple[arrow.Arrow, arrow.Arrow]

    @staticmethod
    def from_json(path: Path) -> User:
        data = json.load(path.open("r"))
        interaction_times = User._validate_interaction_times(data["interaction_times"])
        return User(
            responses=[Response.from_json(d) for d in data["responses"]],
            interact_times=(
                arrow.get(interaction_times[0]),
                arrow.get(interaction_times[1]),
            ),
        )

    @staticmethod
    def _validate_interaction_times(
        times: tuple[arrow.Arrow, arrow.Arrow]
    ) -> tuple[arrow.Arrow, arrow.Arrow]:
        if len(times) != 2:
            raise ValueError("interaction_times must be a tuple of length 2")
        if times[0] > times[1]:
            logging.warning("interaction_times must be in ascending order, flipping")
            times = (times[1], times[0])
        return times

    def response_durations(self) -> list[float]:
        return [r.duration() for r in self.responses]

    def interaction_duration(self) -> float:
        # See https://github.com/python/mypy/issues/11613 for explanation of typing issue
        return (self.interact_times[1] - self.interact_times[0]).total_seconds()  # type: ignore


@dataclass
class Response:
    question_id: int
    answer: bool
    times: tuple[arrow.Arrow, arrow.Arrow]
    steps: tuple[int, int]

    @staticmethod
    def from_json(data: dict[str, Any]) -> Response:
        return Response(
            question_id=data["question_id"],
            answer=data["answer"],
            times=(arrow.get(data["start_time"]), arrow.get(data["end_time"])),
            steps=data["max_steps"],
        )

    @staticmethod
    def _validate_times(
        times: tuple[arrow.Arrow, arrow.Arrow]
    ) -> tuple[arrow.Arrow, arrow.Arrow]:
        if len(times) != 2:
            raise ValueError("response times must be a tuple of length 2")
        if times[0] > times[1]:
            logging.warning("response times must be in ascending order, flipping")
            times = (times[1], times[0])
        return times

    def duration(self) -> float:
        # See https://github.com/python/mypy/issues/11613 for explanation of typing issue
        return (self.times[1] - self.times[0]).total_seconds()  # type: ignore

    def min_steps(self) -> int:
        return min(self.steps)

    def max_steps(self) -> int:
        return max(self.steps)


class UserDataset:
    def __init__(self, path: Path):
        self.path = path
        self.users = [User.from_json(p) for p in path.glob("user_*.json")]
