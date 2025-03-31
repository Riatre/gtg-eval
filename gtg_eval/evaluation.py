import base64
import collections
import pathlib
import re
import sqlite3
import unicodedata
from typing import Callable

import httpx
import tenacity
from loguru import logger

from gtg_eval import dataset, schema

NormalizedAnswer = collections.namedtuple("NormalizedAnswer", ["name", "franchise"])


def parse_answer(completion: str):
    answer = re.search(r"<name>(.+?)</name>", completion)
    if answer is None:
        return None
    return answer.group(1)


def normalize_title(title: str) -> str:
    """Normalize a game title string.

    This function:
    - Converts title to lowercase
    - Removes special characters like '!', '?', etc.
    - Normalizes quotes (both single and double)
    - Converts accented characters to their ASCII equivalents
    - Normalizes whitespace

    Args:
        title: The game title to normalize

    Returns:
        A normalized version of the title
    """
    if not title:
        return ""

    # Convert to lowercase
    result = str(title).lower()

    # Replace full-width quotes with standard quotes
    result = result.replace("’", "'").replace('"', "'")

    # Normalize unicode characters (accented characters to ASCII)
    result = (
        unicodedata.normalize("NFKD", result).encode("ASCII", "ignore").decode("ASCII")
    )

    # Remove special characters like !, ?, etc.
    result = re.sub(r"[!?:;&@#$%^*()+=\[\]{}<>|/\\~`-]", " ", result)
    result = re.sub(r"\.", "", result)

    # Normalize whitespace (replace multiple spaces with single space)
    result = re.sub(r"\s+", " ", result).strip()

    return result


class _TitleCache:
    """A cache for game titles and franchises using SQLite."""

    def __init__(self, cache_dir: pathlib.Path | None = None):
        """Initialize the title cache with a SQLite database."""
        # Create cache directory if it doesn't exist
        if cache_dir is None:
            cache_dir = pathlib.Path.home() / ".cache" / "gtg-eval"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Connect to SQLite database
        db_path = cache_dir / "titles.db"
        # Use WAL mode for better concurrency
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")

        # Create table if it doesn't exist
        self._create_tables()

    def _create_tables(self):
        """Create the necessary tables if they don't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS game_titles (
            query_normalized TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            franchise TEXT
        )
        """)
        self.conn.commit()

    def get(self, query: str) -> NormalizedAnswer | None:
        """Get a game title from the cache.

        Args:
            query: The query string to look up

        Returns:
            A NormalizedAnswer tuple if found, None otherwise
        """
        if not isinstance(query, str):
            return None

        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT name, franchise FROM game_titles WHERE query_normalized = ?",
            (normalize_title(query),),
        )
        result = cursor.fetchone()

        if result:
            name, franchise = result
            return NormalizedAnswer(name=name, franchise=franchise)
        return None

    def put(self, query: str, name: str, franchise: str | None):
        """Store a game title in the cache.

        Args:
            query: The query string to use as key
            name: The game title name
            franchise: The franchise name or None
        """
        if not isinstance(query, str):
            return
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO game_titles (query_normalized, name, franchise) VALUES (?, ?, ?)",
            (normalize_title(query), name, franchise),
        )
        self.conn.commit()


# Initialize title cache
_title_cache = _TitleCache()


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(),
    retry=tenacity.retry_if_exception_type(httpx.RequestError),
)
async def normalize_answer(answer: str) -> NormalizedAnswer:
    query = answer.strip()

    if not query:
        return NormalizedAnswer(name=query, franchise=None)

    # Check cache first (using normalized query)
    cached_result = _title_cache.get(query)
    if cached_result:
        logger.debug(
            "Using cached franchise",
            query=query,
            result=cached_result,
        )
        return cached_result

    # Normalize the query
    normalized_query = normalize_title(query)

    # Prepare API request - use original query for API
    base_url = "https://api.guessthe.game/api/autocomplete/"
    params = {"q": query, "item_type": "game", "puzzle_type": "gtg", "pnum": 0}

    try:
        logger.debug("Querying franchise API", query=query, normalized=normalized_query)
        async with httpx.AsyncClient(headers={"User-Agent": "GTGEval/1.0"}) as session:
            response = await session.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

        matched: NormalizedAnswer | None = None

        if data.get("status") == "ok" and data.get("results"):
            results = data["results"]
            logger.debug("Received results", result_len=len(results), query=query)

            # Iterate through all results to find a match and cache all franchises
            for result in results:
                title = result.get("title")
                franchise = result.get("franchise")

                if not title:
                    continue

                # Cache this result using normalized title
                _title_cache.put(title, title, franchise)
                logger.debug("Added to cache", franchise=franchise, title=title)

                # Check for exact match with the normalized query
                normalized_title = normalize_title(title)
                if normalized_title == normalized_query:
                    matched = NormalizedAnswer(name=title, franchise=franchise)
                    logger.info(
                        "Found exact match",
                        query=query,
                        normalized=normalized_query,
                        title=title,
                        normalized_title=normalized_title,
                        franchise=franchise,
                    )
                    # Continue caching other results but keep this franchise

            if not matched:
                # No exact match found among the results
                logger.warning(
                    "No exact title match found in API results",
                    query=query,
                    normalized=normalized_query,
                    results=results,
                    results_count=len(results),
                )
                matched = NormalizedAnswer(name=query, franchise=None)
            # Ensure the original query key points to the matched franchise
            _title_cache.put(query, matched.name, matched.franchise)
            return matched
        else:
            # API returned ok status but no results, or status was not ok
            logger.warning(
                "No results found or API status not ok",
                query=query,
                normalized=normalized_query,
                status=data.get("status"),
                results_count=len(data.get("results", [])),
            )
            # Cache None for this query
            _title_cache.put(query, query, None)
            return NormalizedAnswer(name=query, franchise=None)
    except httpx.RequestError:
        logger.exception("Franchise API request failed", query=query)
        raise
    except (ValueError, KeyError, TypeError):
        # JSON parsing error or unexpected response format
        logger.exception("Franchise API response parsing error", query=query)
        raise


def judge(game: schema.Game, normalized: NormalizedAnswer) -> schema.Verdict:
    # Normalize all game answers
    normalized_game_answers = [normalize_title(answer) for answer in game.answers]
    # Normalize the submitted answer
    normalized_answer_name = normalize_title(normalized.name)

    if normalized_answer_name in normalized_game_answers:
        return schema.Verdict.CORRECT
    if (
        game.franchise
        and normalized.franchise is not None
        and game.franchise == normalized.franchise
    ):
        return schema.Verdict.SAME_FRANCHISE
    return schema.Verdict.INCORRECT


async def build_next_prompt(
    dataset: dataset.Dataset,
    state: schema.EvaluationState,
    template: schema.PromptTemplate,
    *,
    allow_video: bool = False,
):
    nth_guess = len(state.guesses)
    assert 0 <= nth_guess < 6, f"Invalid state: {nth_guess=}"
    game = state.game
    next_screenshot = await dataset.screenshot_of(
        game, nth_guess + 1, allow_video=allow_video
    )
    next_screenshot_b64 = base64.b64encode(next_screenshot.read()).decode()
    if nth_guess == 0:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": template.initial},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/webp;base64,{next_screenshot_b64}",
                    },
                },
            ],
        }
    prompt: str = getattr(template, f"after_{nth_guess}")
    last_verdict = state.guesses[-1].verdict
    assert last_verdict != schema.Verdict.CORRECT, "Last verdict cannot be correct"
    last_verdict_str = (
        template.incorrect
        if last_verdict == schema.Verdict.INCORRECT
        else template.same_franchise
    )
    hint = [
        game.metacritic_score,
        game.console_platform,
        game.genre,
        game.release_year,
        game.developer,
    ][nth_guess - 1]
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt.format(verdict=last_verdict_str, hint=hint),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/webp;base64,{next_screenshot_b64}"},
            },
        ],
    }


async def progress(
    dataset: dataset.Dataset,
    state: schema.EvaluationState,
    template: schema.PromptTemplate,
    completion: Callable,
) -> schema.EvaluationState:
    """Run one step in the evaluation process.

    This function takes the current evaluation state, builds the next prompt,
    sends it to the VLM using the provided completion function, processes the
    response, judges the answer, and returns the updated state.

    Args:
        dataset: The dataset containing game images
        state: The current evaluation state
        template: The prompt template to use
        completion: A callable that takes a 'messages' argument and returns a completion
                   (other arguments like model should be partial bound before passing)

    Returns:
        Updated evaluation state with the next prompt, model response, and judgment

    Raises:
        AssertionError: If the state is invalid (e.g., too many guesses or last verdict was correct)
    """
    assert not state.done, "Evaluation already complete"

    # Build the next prompt
    next_prompt = await build_next_prompt(dataset, state, template, allow_video=False)

    # Create messages list for the model
    messages = state.messages + [next_prompt]

    # Call the LLM using the provided completion function
    logger.info("Calling LLM for evaluation step", step=len(state.guesses) + 1)
    response = await completion(messages=messages)
    logger.debug("Model response", response=response)

    if isinstance(response, dict):
        model_message = response["choices"][0]["message"]
        assert model_message["role"] == "assistant"
        content = model_message["content"]
    else:
        model_message = response.choices[0].message
        assert model_message.role == "assistant"
        content = model_message.content
    if not content:
        content = ""

    # Extract the game name from the response
    logger.info("Model response message", content=content)
    answer_text = parse_answer(content)
    if answer_text is None:
        logger.warning("No answer found in model response", content=content)
        # Default to empty string if no answer found; effectively a skip
        answer_text = ""

    # Normalize the answer to get the franchise information
    normalized_answer = await normalize_answer(answer_text)
    logger.debug("Normalized answer", answer=normalized_answer)

    # Judge the answer
    verdict = judge(state.game, normalized_answer)
    logger.debug("Judged", verdict=verdict)

    # Create a new guess
    new_guess = schema.Guess(answer=answer_text, verdict=verdict)

    # Create a new state with the updated messages and guesses
    new_state = schema.EvaluationState(
        game=state.game,
        guesses=state.guesses.copy() + [new_guess],
        messages=messages
        + [
            model_message.model_dump()
            if not isinstance(model_message, dict)
            else model_message
        ],
    )

    logger.info(
        "Evaluation step completed",
        step=len(state.guesses) + 1,
        answer=answer_text,
        verdict=verdict,
    )

    return new_state
