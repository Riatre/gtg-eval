import asyncio
import functools
import inspect
import re
from typing import Any, Callable

import typer

from gtg_eval import schema


def parse_id_range(id_range: str) -> list[str]:
    """Parse a comma-separated ID range string into a list of IDs.

    Supports dash-separated ranges, e.g., '1,3,55-77' becomes ['1', '3', '55', '56', ..., '77']

    Args:
        id_range: Comma-separated ID range string

    Returns:
        List of expanded IDs
    """
    ids = []
    for part in id_range.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            try:
                # Expand the range if both parts are numeric
                start_num, end_num = int(start), int(end)
                ids.extend(str(i) for i in range(start_num, end_num + 1))
            except ValueError:
                # If not numeric, just add the original part
                ids.append(part)
        else:
            ids.append(part)
    return ids


def sanitize_messages_for_trace(messages: list[dict], game_id: str) -> list[dict]:
    """Sanitize messages for trace output by replacing base64 image data with URLs.

    Args:
        messages: List of messages
        game_id: Game ID

    Returns:
        Sanitized messages
    """
    sanitized = []
    image_pattern = re.compile(r'data:image/[^;]+;base64,[^"]+"')

    image_num = 1
    for msg in messages:
        sanitized_msg = msg.copy()

        # Handle content field which could be a string or a list of content blocks
        if isinstance(msg.get("content"), list):
            sanitized_content = []
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    # Replace base64 data with URL
                    sanitized_block = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"https://guessthe.game/games/{game_id}/{image_num}.webp"
                        },
                    }
                    sanitized_content.append(sanitized_block)
                    image_num += 1
                else:
                    sanitized_content.append(block)
            sanitized_msg["content"] = sanitized_content
        elif isinstance(msg.get("content"), str):
            # Replace any base64 image data in string content
            sanitized_msg["content"] = image_pattern.sub(
                f'"https://guessthe.game/games/{game_id}/X.webp"', msg["content"]
            )

        sanitized.append(sanitized_msg)

    return sanitized


def sanitize_state_for_trace(state: schema.EvaluationState) -> schema.EvaluationState:
    return state.model_copy(
        update={"messages": sanitize_messages_for_trace(state.messages, state.game.id)}
    )


def calculate_metrics(traces: dict[str, schema.Trace]) -> schema.Metrics:
    """Calculate evaluation metrics from traces.

    Args:
        traces: Dictionary of game traces

    Returns:
        Evaluation metrics
    """
    # Initialize counters
    total_games = len(traces)
    correct_at_k = [0] * 6  # Accuracy@k (k=1..6)
    franchise_at_k = [0] * 6  # Franchise Accuracy@k (k=1..6)
    first_correct_rounds = []  # First Correct Round

    for _, trace in traces.items():
        state = trace.final_state

        if state.solved:
            first_correct_rounds.append(state.attempts)

            for k in range(state.attempts, 7):
                correct_at_k[k - 1] += 1

        if state.same_franchise_at:
            for k in range(state.same_franchise_at, 7):
                franchise_at_k[k - 1] += 1

    # Calculate final metrics
    metrics = {}
    for k in range(1, 7):
        metrics[f"accuracy_at_{k}"] = (
            correct_at_k[k - 1] / total_games if total_games > 0 else 0
        )
        metrics[f"franchise_accuracy_at_{k}"] = (
            franchise_at_k[k - 1] / total_games if total_games > 0 else 0
        )

    metrics["first_correct_round"] = (
        sum(first_correct_rounds) / len(first_correct_rounds)
        if first_correct_rounds
        else float("nan")
    )
    metrics["solved_rate"] = (
        len(first_correct_rounds) / total_games if total_games > 0 else 0
    )

    return schema.Metrics(**metrics)


async def limited(limiter: asyncio.Semaphore, func, *args, **kwargs):
    async with limiter:
        return await func(*args, **kwargs)


class AsyncTyper(typer.Typer):
    @staticmethod
    def maybe_run_async(decorator: Callable, func: Callable) -> Any:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            def runner(*args: Any, **kwargs: Any) -> Any:
                return asyncio.run(func(*args, **kwargs))

            decorator(runner)
        else:
            decorator(func)
        return func

    def callback(self, *args: Any, **kwargs: Any) -> Any:
        decorator = super().callback(*args, **kwargs)
        return functools.partial(self.maybe_run_async, decorator)

    def command(self, *args: Any, **kwargs: Any) -> Any:
        decorator = super().command(*args, **kwargs)
        return functools.partial(self.maybe_run_async, decorator)
