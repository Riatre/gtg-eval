#!/usr/bin/env python3

import argparse
import functools
import json
import os
import re
import sqlite3
import time

import litellm
import tqdm
from loguru import logger

from gtg_eval import dataset, evaluation, logging, schema


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
        total_time REAL DEFAULT 0,
        step_times TEXT,  -- JSON array of step times
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Create table for storing evaluation metadata
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluation_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP,
        total_tokens INTEGER DEFAULT 0,
        prompt_tokens INTEGER DEFAULT 0,
        completion_tokens INTEGER DEFAULT 0,
        total_games INTEGER DEFAULT 0,
        completed_games INTEGER DEFAULT 0
    )
    """)

    conn.commit()
    return conn


def get_evaluation_state(
    conn: sqlite3.Connection, game_id: str
) -> tuple[schema.EvaluationState | None, float, list[float]]:
    """Get the evaluation state and timing information for a game from the checkpoint database.

    Args:
        conn: SQLite connection
        game_id: Game ID

    Returns:
        Tuple of (evaluation_state, total_time, step_times)
        evaluation_state will be None if not found
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT state_json, total_time, step_times FROM evaluation_states WHERE game_id = ?",
        (game_id,),
    )
    result = cursor.fetchone()

    if result:
        state_json, total_time, step_times_json = result
        state = schema.EvaluationState.model_validate_json(state_json)
        step_times = json.loads(step_times_json) if step_times_json else []
        return state, total_time, step_times
    return None, 0.0, []


def save_evaluation_state(
    conn: sqlite3.Connection,
    game_id: str,
    state: schema.EvaluationState,
    total_time: float,
    step_times: list[float],
):
    """Save the evaluation state and timing for a game to the checkpoint database.

    Args:
        conn: SQLite connection
        game_id: Game ID
        state: Evaluation state
        total_time: Total time spent on the game so far
        step_times: List of times for each step
    """
    cursor = conn.cursor()
    state_json = state.model_dump_json()
    step_times_json = json.dumps(step_times)

    cursor.execute(
        """
        INSERT OR REPLACE INTO evaluation_states (game_id, state_json, total_time, step_times, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (game_id, state_json, total_time, step_times_json),
    )
    conn.commit()


def update_metadata(conn: sqlite3.Connection, metadata_id: int, **kwargs):
    """Update evaluation metadata in the checkpoint database.

    Args:
        conn: SQLite connection
        metadata_id: Metadata ID
        **kwargs: Fields to update
    """
    if not kwargs:
        return

    cursor = conn.cursor()
    # lol claude while this is fine here you are going to trigger a lot of
    # "OMG potential SQLi" red herrings
    set_clause = ", ".join(f"{key} = ?" for key in kwargs.keys())
    values = list(kwargs.values()) + [metadata_id]

    cursor.execute(f"UPDATE evaluation_metadata SET {set_clause} WHERE id = ?", values)
    conn.commit()


def create_metadata_entry(
    conn: sqlite3.Connection, model_name: str, total_games: int
) -> int:
    """Create a new metadata entry in the checkpoint database.

    Args:
        conn: SQLite connection
        model_name: Model name
        total_games: Total number of games to evaluate

    Returns:
        Metadata ID
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO evaluation_metadata 
        (model_name, start_time, total_games)
        VALUES (?, CURRENT_TIMESTAMP, ?)
        """,
        (model_name, total_games),
    )
    conn.commit()
    return cursor.lastrowid


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

    # Setup logging
    logging.setup_logging(level=args.log_level)

    # Check if model supports vision
    if not litellm.supports_vision(model=args.model):
        logger.error("Model does not support vision", model=args.model)
        return 1

    # Parse game IDs
    game_ids = parse_id_range(args.game_ids)
    logger.info("games to evaluate", count=len(game_ids))

    # Load dataset
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

    # Load prompt template
    try:
        with open(args.prompt_template, "r") as f:
            template = schema.PromptTemplate.model_validate_json(f.read())
        logger.info("Loaded prompt template", path=args.prompt_template)
    except Exception as e:
        logger.error("Failed to load prompt template", error=str(e), exc_info=True)
        return 1

    # Setup checkpoint database
    try:
        conn = setup_checkpoint_db(args.checkpoint_db)
        logger.info("Connected to checkpoint database", path=args.checkpoint_db)
    except Exception as e:
        logger.error("Failed to setup checkpoint database", error=str(e), exc_info=True)
        return 1

    # Create metadata entry
    metadata_id = create_metadata_entry(conn, args.model, len(valid_game_ids))

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

    def completion_with_tracking(messages, **kwargs):
        """Wrapper for litellm.completion that tracks token usage."""
        start_time = time.time()
        try:
            response = litellm.completion(messages=messages, **kwargs)
            elapsed = time.time() - start_time

            # Extract token usage
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            total_tokens = prompt_tokens + completion_tokens

            # Log token usage
            logger.info(
                "Completion finished",
                elapsed_seconds=elapsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

            # Update metadata
            update_metadata(
                conn,
                metadata_id,
                prompt_tokens=f"prompt_tokens + {prompt_tokens}",
                completion_tokens=f"completion_tokens + {completion_tokens}",
                total_tokens=f"total_tokens + {total_tokens}",
            )

            return response
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "Completion failed",
                elapsed_seconds=elapsed,
                error=str(e),
                exc_info=True,
            )
            raise

    # Create partial function with model settings
    completion_func = functools.partial(completion_with_tracking, **completion_kwargs)

    # Initialize traces dictionary
    traces = {}

    # Process each game
    try:
        for i, game_id in enumerate(tqdm.tqdm(valid_game_ids)):
            game = ds[game_id]
            logger.info(
                "Processing game",
                game_id=game_id,
                progress=f"{i+1}/{len(valid_game_ids)}",
            )

            # Check if we have a checkpoint for this game
            state, total_time, step_times = get_evaluation_state(conn, game_id)

            if state is None:
                # Initialize new state
                state = schema.EvaluationState(game=game)
                logger.info("Starting new evaluation", game_id=game_id)
            else:
                logger.info(
                    "Resuming evaluation from checkpoint",
                    game_id=game_id,
                    guesses=len(state.guesses),
                    previous_time=total_time,
                    previous_steps=len(step_times),
                )

            # Track timing for the game
            game_start_time = time.time()

            # Run evaluation steps until done
            while not state.done:
                step_start_time = time.time()
                try:
                    state = evaluation.progress(ds, state, template, completion_func)
                    step_time = time.time() - step_start_time
                    step_times.append(step_time)

                    # Update timing information
                    total_time += step_time

                    # Save checkpoint and timing after each step
                    save_evaluation_state(conn, game_id, state, total_time, step_times)

                    logger.info(
                        "Evaluation step completed",
                        game_id=game_id,
                        step=len(state.guesses),
                        step_time=step_time,
                        verdict=state.guesses[-1].verdict if state.guesses else None,
                    )
                except Exception as e:
                    logger.error(
                        "Error in evaluation step",
                        game_id=game_id,
                        error=str(e),
                        exc_info=True,
                    )
                    break

            # Record final game timing
            current_session_time = time.time() - game_start_time
            total_time += current_session_time
            save_evaluation_state(conn, game_id, state, total_time, step_times)

            # Remove the raw messages from the state and replace with sanitized messages
            sanitized_state = state.model_copy(
                update={
                    "messages": sanitize_messages_for_trace(state.messages, game_id)
                }
            )

            # Create trace entry using typed models
            trace = schema.Trace(
                game_id=game_id,
                final_state=sanitized_state,
                total_time=total_time,
                step_times=step_times,
                average_step_time=sum(step_times) / len(step_times)
                if step_times
                else 0,
                solved=state.solved,
                attempts=state.attempts,
                same_franchise_at=state.same_franchise_at,
            )

            traces[game_id] = trace

            # Update metadata
            update_metadata(conn, metadata_id, completed_games="completed_games + 1")

            # Log game result
            logger.info(
                "Game evaluation completed",
                game_id=game_id,
                solved=state.solved,
                attempts=state.attempts,
                same_franchise_at=state.same_franchise_at,
                total_time=total_time,
            )

            # Calculate and log partial metrics
            if i > 0 and (i + 1) % 5 == 0:
                metrics = calculate_metrics(traces)
                logger.info("Metrics till now", metrics=metrics, games_completed=i + 1)

    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
    except Exception as e:
        logger.error("Unexpected error during evaluation", error=str(e), exc_info=True)
    finally:
        # Calculate final metrics
        metrics = calculate_metrics(traces)
        logger.info("Final metrics", metrics=metrics)

        # Update metadata with end time
        update_metadata(conn, metadata_id, end_time="CURRENT_TIMESTAMP")

        # Write traces to output file
        try:
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Create evaluation output object
            output = schema.Output(
                metadata=schema.EvalMetadata(
                    model=args.model,
                    model_config=completion_kwargs,
                    dataset=args.dataset,
                    timestamp=time.time(),
                    games_count=len(traces),
                ),
                metrics=metrics,
                traces=traces,
            )

            with open(args.output, "w") as f:
                f.write(output.model_dump_json(indent=2))

            logger.info("Wrote traces to output file", path=args.output)
        except Exception as e:
            logger.error("Failed to write traces", error=str(e), exc_info=True)

    return 0


if __name__ == "__main__":
    exit(main())
