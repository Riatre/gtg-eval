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


class PromptTemplate(pydantic.BaseModel):
    initial: str
    incorrect: str
    same_franchise: str
    after_1: str
    after_2: str
    after_3: str
    after_4: str
    after_5: str
