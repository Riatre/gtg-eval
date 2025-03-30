import pydantic
import enum
import litellm.types.llms.openai as litellm_openai


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
    messages: list[litellm_openai.AllMessageValues] = []

    @property
    def done(self) -> bool:
        return (
            len(self.guesses) >= 6
            or self.guesses
            and self.guesses[-1].verdict == Verdict.CORRECT
        )

    @property
    def solved(self) -> bool:
        return self.guesses and self.guesses[-1].verdict == Verdict.CORRECT

    @property
    def attempts(self) -> int:
        return len(self.guesses)

    @property
    def same_franchise_at(self) -> int | None:
        return next(
            (
                i
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
