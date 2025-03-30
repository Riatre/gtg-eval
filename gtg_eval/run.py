#!/usr/bin/env python3

import argparse
import dataclasses
import json
import os
import re
import sqlite3
import time

import litellm
import tqdm
from loguru import logger

from gtg_eval import dataset, evaluation, logging, schema


@dataclasses.dataclass
class _Checkpoint:
    state: schema.EvaluationState
    step_times: list[float] = dataclasses.field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


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


def setup_checkpoint_db(db_path: str) -> sqlite3.Connection:
    """Set up the checkpoint database for storing evaluation states.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        SQLite connection object
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table for storing evaluation states if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluation_states (
        game_id TEXT PRIMARY KEY,
        state_json TEXT NOT NULL,
        step_times TEXT,  -- JSON array of step times
        prompt_tokens INTEGER DEFAULT 0,
        completion_tokens INTEGER DEFAULT 0,
        total_tokens INTEGER DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    return conn


def load_checkpoint(conn: sqlite3.Connection, game_id: str) -> _Checkpoint | None:
    """Get the evaluation state, timing, and token usage information for a game from the checkpoint database.

    Args:
        conn: SQLite connection
        game_id: Game ID

    Returns:
        _Checkpoint object with evaluation state and metrics
        If no state is found, state field will be None
    """
    cursor = conn.cursor()
    cursor.execute(
        """SELECT state_json, step_times, prompt_tokens, completion_tokens, total_tokens 
           FROM evaluation_states WHERE game_id = ?""",
        (game_id,),
    )
    result = cursor.fetchone()

    if result:
        state = schema.EvaluationState.model_validate_json(result[0])
        step_times = json.loads(result[1]) if result[1] else []
        return _Checkpoint(
            state=state,
            step_times=step_times,
            prompt_tokens=result[2],
            completion_tokens=result[3],
            total_tokens=result[4],
        )


def save_checkpoint(conn: sqlite3.Connection, game_id: str, ckpt: _Checkpoint):
    """Save the game state to the checkpoint database.

    Args:
        conn: SQLite connection
        game_id: Game ID
        ckpt: _Checkpoint object containing evaluation state and metrics
    """
    conn.cursor().execute(
        """
        INSERT OR REPLACE INTO evaluation_states 
        (game_id, state_json, step_times, prompt_tokens, completion_tokens, total_tokens, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (
            game_id,
            ckpt.state.model_dump_json(),
            json.dumps(ckpt.step_times),
            ckpt.prompt_tokens,
            ckpt.completion_tokens,
            ckpt.total_tokens,
        ),
    )
    conn.commit()


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

    for msg in messages:
        sanitized_msg = msg.copy()

        # Handle content field which could be a string or a list of content blocks
        if isinstance(msg.get("content"), list):
            sanitized_content = []
            image_num = 1
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


# TODO(riatre): This is correct but claude made it very verbose. Cleanup later.
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
        guesses = trace.final_state.guesses

        # Skip if no guesses
        if not guesses:
            continue

        # Find first correct guess
        first_correct = None
        for i, guess in enumerate(guesses):
            if guess.verdict == schema.Verdict.CORRECT:
                first_correct = i + 1  # 1-indexed
                break

        if first_correct is not None:
            first_correct_rounds.append(first_correct)

        # Calculate Accuracy@k
        for k in range(1, 7):
            if k <= len(guesses) and any(
                g.verdict == schema.Verdict.CORRECT for g in guesses[:k]
            ):
                correct_at_k[k - 1] += 1

        # Calculate Franchise Accuracy@k
        for k in range(1, 7):
            if k <= len(guesses) and any(
                g.verdict in [schema.Verdict.CORRECT, schema.Verdict.SAME_FRANCHISE]
                for g in guesses[:k]
            ):
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


class LLM:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0

    def completion(self, **kwargs):
        """Wrapper for litellm.completion that tracks token usage."""
        start_time = time.time()
        response = litellm.completion(**self._kwargs, **kwargs)
        elapsed = time.time() - start_time

        # Extract token usage
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage"):
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
            completion_tokens = getattr(response.usage, "completion_tokens", 0)
        elif "usage" in response:
            prompt_tokens = response["usage"].get("prompt_tokens", 0)
            completion_tokens = response["usage"].get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        # Update token tracker
        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = completion_tokens
        self.last_total_tokens = total_tokens

        # Log token usage
        logger.info(
            "Completion finished",
            elapsed_seconds=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        return response


def _run_game(
    llm: LLM,
    ds: dataset.Dataset,
    conn: sqlite3.Connection,
    template: schema.PromptTemplate,
    game_id: str,
) -> schema.Trace:
    game = ds[game_id]
    logger.info("Processing game", game_id=game_id)

    # Check if we have a checkpoint for this game
    ckpt = load_checkpoint(conn, game_id)

    if ckpt.state is None:
        # Initialize new state
        ckpt.state = schema.EvaluationState(game=game)
        logger.info("Starting new evaluation", game_id=game_id)
    else:
        logger.info(
            "Resuming evaluation from checkpoint",
            game_id=game_id,
            guesses=len(ckpt.state.guesses),
            previous_steps=len(ckpt.step_times),
            prompt_tokens=ckpt.prompt_tokens,
            completion_tokens=ckpt.completion_tokens,
            total_tokens=ckpt.total_tokens,
        )

    # Run evaluation steps until done
    while not ckpt.state.done:
        step_start_time = time.time()
        # Run the evaluation step
        ckpt.state = evaluation.progress(ds, ckpt.state, template, llm.completion)
        step_time = time.time() - step_start_time
        ckpt.step_times.append(step_time)

        # Update token usage from the token tracker
        ckpt.prompt_tokens += llm.last_prompt_tokens
        ckpt.completion_tokens += llm.last_completion_tokens
        ckpt.total_tokens += llm.last_total_tokens

        # Save checkpoint after each step
        save_checkpoint(conn, game_id, ckpt)

        logger.info(
            "Evaluation step completed",
            game_id=game_id,
            step=len(ckpt.state.guesses),
            step_time=step_time,
            verdict=ckpt.state.guesses[-1].verdict if ckpt.state.guesses else None,
            step_tokens=llm.last_total_tokens,
        )

    # Remove the raw messages from the state and replace with sanitized messages
    sanitized_state = ckpt.state.model_copy(
        update={"messages": sanitize_messages_for_trace(ckpt.state.messages, game_id)}
    )

    avg_step_time = (
        sum(ckpt.step_times) / len(ckpt.step_times) if ckpt.step_times else 0
    )

    return schema.Trace(
        game_id=game_id,
        final_state=sanitized_state,
        total_time=sum(ckpt.step_times),
        step_times=ckpt.step_times,
        average_step_time=avg_step_time,
        prompt_tokens=ckpt.prompt_tokens,
        completion_tokens=ckpt.completion_tokens,
        total_tokens=ckpt.total_tokens,
        solved=ckpt.state.solved,
        attempts=ckpt.state.attempts,
        same_franchise_at=ckpt.state.same_franchise_at,
    )


def _build_argparse():
    parser = argparse.ArgumentParser(description="Run GTG evaluation")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--game-ids",
        type=str,
        required=True,
        help="Comma-separated game IDs or ranges (e.g., '1,3,55-77')",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name for litellm (e.g., 'openai/gpt-4-vision-preview')",
    )
    parser.add_argument(
        "--checkpoint-db",
        type=str,
        required=True,
        help="Path to checkpoint SQLite database",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file for traces",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="prompt_template/english.json",
        help="Path to prompt template JSON file",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for model sampling (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens in the response (default: 4096)",
    )
    return parser


def main():
    args = _build_argparse().parse_args()
    logging.setup_logging(level=args.log_level)

    if not litellm.supports_vision(model=args.model):
        logger.error("Model does not support vision", model=args.model)
        return 1

    game_ids = parse_id_range(args.game_ids)
    logger.info("games to evaluate", count=len(game_ids))

    try:
        ds = dataset.Dataset(args.dataset)
        logger.info("Loaded dataset", path=args.dataset, game_count=len(ds))
    except Exception as e:
        logger.error("Failed to load dataset", error=str(e), exc_info=True)
        return 1

    # Filter game IDs to those that exist in the dataset
    valid_game_ids = [gid for gid in game_ids if gid in ds.games]
    if len(valid_game_ids) != len(game_ids):
        missing = set(game_ids) - set(valid_game_ids)
        logger.error("Some game IDs not found in dataset", missing=list(missing))
        return 1

    if not valid_game_ids:
        logger.error("No valid game IDs to evaluate")
        return 1

    try:
        with open(args.prompt_template, "r") as f:
            template = schema.PromptTemplate.model_validate_json(f.read())
        logger.info("Loaded prompt template", path=args.prompt_template)
    except Exception as e:
        logger.error("Failed to load prompt template", error=str(e), exc_info=True)
        return 1

    try:
        conn = setup_checkpoint_db(args.checkpoint_db)
        logger.info("Connected to checkpoint database", path=args.checkpoint_db)
    except Exception as e:
        logger.error("Failed to setup checkpoint database", error=str(e), exc_info=True)
        return 1

    # Prepare completion function with model-specific settings
    completion_kwargs = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    # Add Gemini-specific safety settings if needed
    if "gemini" in args.model.lower():
        completion_kwargs["safety_settings"] = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    llm = LLM(**completion_kwargs)
    traces = {}

    # Process each game
    try:
        for i, game_id in enumerate(tqdm.tqdm(valid_game_ids)):
            traces[game_id] = trace = _run_game(llm, ds, conn, template, game_id)

            logger.info(
                "Game evaluation completed",
                game_id=game_id,
                solved=trace.solved,
                attempts=trace.attempts,
                same_franchise_at=trace.same_franchise_at,
                total_time=sum(trace.step_times),
            )

            if i > 0 and (i + 1) % 5 == 0:
                metrics = calculate_metrics(traces)
                logger.info("Metrics till now", metrics=metrics, games_completed=i + 1)
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        return 1
    metrics = calculate_metrics(traces)
    logger.info("Final metrics", metrics=metrics)

    # Write traces to output file
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens_used = 0
    for trace_obj in traces.values():
        total_prompt_tokens += trace_obj.prompt_tokens
        total_completion_tokens += trace_obj.completion_tokens
        total_tokens_used += trace_obj.total_tokens

    output = schema.Output(
        metadata=schema.EvalMetadata(
            model=args.model,
            model_params=completion_kwargs,
            dataset=args.dataset,
            timestamp=time.time(),
            games_count=len(traces),
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_tokens_used,
        ),
        metrics=metrics,
        traces=traces,
    )

    with open(args.output, "w") as f:
        f.write(output.model_dump_json(indent=2))

    logger.info("Wrote traces to output file", path=args.output)
    return 0


if __name__ == "__main__":
    exit(main())
