import base64
import collections
import pathlib
import re
import sqlite3

import litellm.types.llms.openai as litellm_openai
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from loguru import logger
from gtg_eval import dataset, schema

NormalizedAnswer = collections.namedtuple("NormalizedAnswer", ["name", "franchise"])


def parse_answer(completion: str):
    answer = re.search(r"<name>(.+?)</name>", completion)
    if answer is None:
        return None
    return answer.group(1)


def _setup_api_session():
    """Set up an HTTP session with retry capabilities."""
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,  # Maximum number of retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
        backoff_factor=1,  # Backoff factor for exponential backoff
        respect_retry_after_header=True,  # Honor Retry-After header
        raise_on_status=True,  # Raise exception on status
    )

    # Create adapter with retry strategy
    adapter = HTTPAdapter(max_retries=retry_strategy)

    # Create session and mount adapter
    _api_session = requests.Session()
    _api_session.mount("https://", adapter)
    _api_session.mount("http://", adapter)

    # Set user agent
    _api_session.headers.update({"User-Agent": "GTGEval/1.0"})
    return _api_session


_api_session = _setup_api_session()


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
            query_lower TEXT PRIMARY KEY,
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
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT name, franchise FROM game_titles WHERE query_lower = ?",
            (query.lower(),),
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
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO game_titles (query_lower, name, franchise) VALUES (?, ?, ?)",
            (query.lower(), name, franchise),
        )
        self.conn.commit()


# Initialize title cache
_title_cache = _TitleCache()


def normalize_answer(answer: str) -> NormalizedAnswer:
    query = answer.strip()

    if not query:
        return NormalizedAnswer(name=query, franchise=None)

    # Check cache first (using lower case key)
    cached_result = _title_cache.get(query)
    if cached_result:
        logger.debug(
            "Using cached franchise",
            query=query,
            result=cached_result,
        )
        return cached_result

    # Prepare API request
    base_url = "https://api.guessthe.game/api/autocomplete/"
    params = {"q": query, "item_type": "game", "puzzle_type": "gtg", "pnum": 0}

    try:
        logger.debug("Querying franchise API", query=query)
        response = _api_session.get(base_url, params=params, timeout=10)
        # raise_for_status is handled by the retry mechanism

        data = response.json()

        matched: NormalizedAnswer | None = None

        if data.get("status") == "ok" and data.get("results"):
            results = data["results"]
            logger.debug("Received results", result_len=len(results), query=query)

            # Iterate through all results to find a match and cache all franchises
            for result in results:
                title = result.get("title")
                franchise = result.get("franchise")

                if not title or not franchise:
                    continue

                # Cache this result regardless of match
                _title_cache.put(title, title, franchise)
                logger.debug("Added to cache", franchise=franchise, title=title)

                # Check for exact case-insensitive match with the original query
                if title.lower() == query.lower():
                    matched = NormalizedAnswer(name=title, franchise=franchise)
                    logger.info(
                        "Found exact match",
                        query=query,
                        title=title,
                        franchise=franchise,
                    )
                    # Continue caching other results but keep this franchise

            if not matched:
                # No exact match found among the results
                logger.warning(
                    "No exact title match found in API results",
                    query=query,
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
                status=data.get("status"),
                results_count=len(data.get("results", [])),
            )
            # Cache None for this query
            _title_cache.put(query, query, None)
            return NormalizedAnswer(name=query, franchise=None)
    except requests.RequestException as e:
        logger.error(
            "API request failed after retries",
            query=query,
            error=str(e),
            exc_info=True,
        )
        raise
    except (ValueError, KeyError, TypeError) as e:
        # JSON parsing error or unexpected response format
        logger.error(
            "API response parsing error", query=query, error=str(e), exc_info=True
        )
        raise


def judge(game: schema.Game, normalized: NormalizedAnswer) -> schema.Verdict:
    if normalized.name in game.answers:
        return schema.Verdict.CORRECT
    if (
        game.franchise
        and normalized.franchise is not None
        and game.franchise == normalized.franchise
    ):
        return schema.Verdict.SAME_FRANCHISE
    return schema.Verdict.INCORRECT


def build_next_prompt(
    dataset: dataset.Dataset,
    state: schema.EvaluationState,
    template: schema.PromptTemplate,
    *,
    allow_video: bool = False,
) -> litellm_openai.AllMessageValues:
    nth_guess = len(state.guesses)
    assert 0 <= nth_guess < 6, f"Invalid state: {nth_guess=}"
    game = state.game
    next_screenshot = dataset.screenshot_of(
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
