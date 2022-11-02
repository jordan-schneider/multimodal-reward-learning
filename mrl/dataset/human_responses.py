from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arrow


@dataclass
class User:
    user_id: str
    payment_code: str
    responses: list[Response]
    interact_times: tuple[arrow.Arrow, arrow.Arrow]

    def get_response_durations(self) -> list[datetime.timedelta]:
        return [r.duration() for r in self.responses]

    def get_interact_duration(self) -> datetime.timedelta:
        duration = self.interact_times[1] - self.interact_times[0]
        return duration  # type: ignore

    def get_total_duration(self) -> datetime.timedelta:
        return self.get_interact_duration() + sum(
            self.get_response_durations(), datetime.timedelta(seconds=0)
        )

    def get_min_response_time(self) -> datetime.timedelta:
        return (
            min(self.get_response_durations())
            if len(self.responses) > 0
            else datetime.timedelta(seconds=0)
        )

    def get_max_response_time(self) -> datetime.timedelta:
        return (
            max(self.get_response_durations())
            if len(self.responses) > 0
            else datetime.timedelta(seconds=0)
        )

    def get_fraction_both_started(self) -> float:
        return sum(
            [
                (r.steps[0] > 0 and r.steps[1] > 0)
                or (
                    r.steps[0] == 0
                    and r.steps[1] == 0
                    and r.duration() > datetime.timedelta(seconds=1)
                )
                for r in self.responses
            ]
        ) / len(self.responses)

    @staticmethod
    def from_json(path: Path) -> User:
        data: dict[str, Any] = json.load(path.open("r"))
        if "interact_times" not in data.keys() or data["interact_times"] is None:
            raise ValueError(f"Missing interact_times in {path}")
        interaction_times = User._validate_interaction_times(data["interact_times"])
        return User(
            user_id=data["user_id"],
            payment_code=data["payment_code"],
            responses=[Response.from_dict(d) for d in data["responses"]],
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

    def response_durations(self) -> list[datetime.timedelta]:
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
    def from_dict(data: dict[str, Any]) -> Response:
        return Response(
            question_id=data["question_id"],
            answer=data["answer"],
            times=Response._validate_times(
                (arrow.get(data["start_time"]), arrow.get(data["end_time"]))
            ),
            steps=(int(data["max_steps"][0]), int(data["max_steps"][1])),
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

    def duration(self) -> datetime.timedelta:
        return self.times[1] - self.times[0]  # type: ignore

    def min_steps(self) -> int:
        return min(self.steps)

    def max_steps(self) -> int:
        return max(self.steps)


class UserDataset:
    def __init__(self, path: Path):
        self.path = path
        self.users = []
        for p in path.glob("*.json"):
            try:
                user = User.from_json(p)
                self.users.append(user)
            except ValueError as e:
                logging.warning(f"Skipping {p.name} due to {e}")
