import enum
from typing import Any

import pydantic


class Game(pydantic.BaseModel):
    id: str
    description: str | None = None
    answers: list[str]
    franchise: str
    submitted_by: str | None = None
    release_year: str
    metacritic_score: str
    genre: str
    console_platform: str
    developer: str


class Verdict(enum.StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    SAME_FRANCHISE = "same_franchise"


class Guess(pydantic.BaseModel):
    answer: str
    verdict: Verdict


class EvaluationState(pydantic.BaseModel):
    game: Game
    guesses: list[Guess] = []
    messages: list = []

    @property
    def done(self) -> bool:
        return (
            len(self.guesses) >= 6
            or len(self.guesses) > 0
            and self.guesses[-1].verdict == Verdict.CORRECT
        )

    @property
    def solved(self) -> bool:
        return len(self.guesses) > 0 and self.guesses[-1].verdict == Verdict.CORRECT

    @property
    def attempts(self) -> int:
        return len(self.guesses)

    @property
    def same_franchise_at(self) -> int | None:
        return next(
            (
                i + 1
                for i, guess in enumerate(self.guesses)
                if guess.verdict in (Verdict.SAME_FRANCHISE, Verdict.CORRECT)
            ),
            None,
        )


class PromptTemplate(pydantic.BaseModel):
    initial: str
    incorrect: str
    same_franchise: str
    after_1: str
    after_2: str
    after_3: str
    after_4: str
    after_5: str


class Trace(pydantic.BaseModel):
    game_id: str
    final_state: EvaluationState

    total_time: float
    step_times: list[float]
    average_step_time: float

    # Token usage information
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    solved: bool
    attempts: int
    same_franchise_at: int | None = None


class Metrics(pydantic.BaseModel):
    accuracy_at_1: float
    accuracy_at_2: float
    accuracy_at_3: float
    accuracy_at_4: float
    accuracy_at_5: float
    accuracy_at_6: float
    franchise_accuracy_at_1: float
    franchise_accuracy_at_2: float
    franchise_accuracy_at_3: float
    franchise_accuracy_at_4: float
    franchise_accuracy_at_5: float
    franchise_accuracy_at_6: float
    first_correct_round: float
    solved_rate: float


class EvalMetadata(pydantic.BaseModel):
    model: str
    model_params: dict[str, Any]
    prompt: PromptTemplate
    dataset: str
    timestamp: float
    games_count: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Output(pydantic.BaseModel):
    metadata: EvalMetadata
    metrics: Metrics
    traces: dict[str, Trace]
