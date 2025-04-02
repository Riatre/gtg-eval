#!/usr/bin/env python3

import asyncio
import dataclasses
import json
import os
import sqlite3
import time
from typing import Annotated

import typer
from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from gtg_eval import dataset, evaluation, lm_adapter, logging, schema, utils


@dataclasses.dataclass
class _Checkpoint:
    state: schema.EvaluationState
    step_times: list[float] = dataclasses.field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def setup_checkpoint_db(db_path: str) -> sqlite3.Connection:
    """Set up the checkpoint database for storing evaluation states.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        SQLite connection object
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    conn.execute("PRAGMA journal_mode=WAL")

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
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
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


async def _run_game(
    ds: dataset.Dataset,
    db_path: str,
    template: schema.PromptTemplate,
    game_id: str,
    image_before_text: bool,
    adapter_type: lm_adapter.AdapterType,
    lm_kwargs: dict,
) -> schema.Trace:
    game = ds[game_id]
    logger.info("Processing game", game_id=game_id)

    # Each thread needs its own DB connection and LLM instance
    conn = setup_checkpoint_db(db_path)
    llm = lm_adapter.make_lm_adapter(adapter_type, **lm_kwargs)

    try:
        # Check if we have a checkpoint for this game
        ckpt = load_checkpoint(conn, game_id)

        if ckpt is None:
            # Initialize new state
            ckpt = _Checkpoint(state=schema.EvaluationState(game=game))
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
            ckpt.state = await evaluation.progress(
                ds,
                ckpt.state,
                template,
                llm.completion,
                image_before_text=image_before_text,
            )
            step_time = time.time() - step_start_time
            ckpt.step_times.append(step_time)

            # Update token usage from the token tracker
            ckpt.prompt_tokens += llm.last_prompt_tokens
            ckpt.completion_tokens += llm.last_completion_tokens
            ckpt.total_tokens += llm.last_total_tokens

            save_checkpoint(conn, game_id, ckpt)

            logger.trace(
                "Evaluation step completed",
                game_id=game_id,
                step=len(ckpt.state.guesses),
                step_time=step_time,
                verdict=ckpt.state.guesses[-1].verdict if ckpt.state.guesses else None,
                step_tokens=llm.last_total_tokens,
            )

        avg_step_time = (
            sum(ckpt.step_times) / len(ckpt.step_times) if ckpt.step_times else 0
        )

        return schema.Trace(
            game_id=game_id,
            final_state=utils.sanitize_state_for_trace(ckpt.state),
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
    finally:
        conn.close()  # Ensure connection is closed in this thread


async def main(
    dataset_path: Annotated[str, typer.Option("--dataset", help="Path to dataset dir")],
    game_ids: Annotated[
        str, typer.Option(help="Comma-separated game IDs or ranges (e.g., '1,3,55-77')")
    ],
    model: Annotated[str, typer.Option(help="Model name (e.g., 'gpt-4o')")],
    checkpoint_db: Annotated[str, typer.Option(help="Path to checkpoint")],
    output: Annotated[str, typer.Option(help="Path to output traces JSON file")],
    prompt_template: Annotated[
        str, typer.Option(help="Path to prompt template JSON file")
    ] = "prompt_template/english-v3.json",
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.7,
    max_tokens: Annotated[int, typer.Option(help="Max number of output tokens")] = 4096,
    concurrency: Annotated[int, typer.Option(help="Number of concurrent workers")] = 1,
    adapter_type: Annotated[
        lm_adapter.AdapterType, typer.Option()
    ] = lm_adapter.AdapterType.LITELLM,
    api_base: Annotated[
        str | None, typer.Option(help="OpenAI-compatible API base")
    ] = None,
    api_key_env: Annotated[
        str | None,
        typer.Option(help="Environment variable name of API key for --api-base"),
    ] = None,
    image_before_text: Annotated[
        bool, typer.Option(help="Whether to put the image before the text")
    ] = False,
) -> int:
    logging.setup_logging(
        term=lambda msg: tqdm_asyncio.write(msg, end=""), colorize=True
    )

    api_key = None
    if api_base is not None:
        assert adapter_type == lm_adapter.AdapterType.LITELLM, (
            "Cannot use --api-base if adapter is not LITELLM"
        )
        assert api_key_env is not None, "Must set --api-key-env if --api-base is set"
        api_key = os.environ[api_key_env]

    game_ids_list = utils.parse_id_range(game_ids)
    logger.info("games to evaluate", count=len(game_ids_list))

    try:
        ds = dataset.Dataset(dataset_path)
        logger.info("Loaded dataset", path=dataset_path, game_count=len(ds))
    except Exception:
        logger.exception("Failed to load dataset")
        return 1

    assert ds.validate()

    # Filter game IDs to those that exist in the dataset
    valid_game_ids = [gid for gid in game_ids_list if gid in ds.games]
    if len(valid_game_ids) != len(game_ids_list):
        missing = set(game_ids_list) - set(valid_game_ids)
        logger.error("Some game IDs not found in dataset", missing=list(missing))
        return 1

    if not valid_game_ids:
        logger.error("No valid game IDs to evaluate")
        return 1

    try:
        with open(prompt_template, "r") as f:
            template = schema.PromptTemplate.model_validate_json(f.read())
        logger.info("Loaded prompt template", path=prompt_template)
    except Exception:
        logger.exception("Failed to load prompt template")
        return 1

    os.makedirs(os.path.dirname(checkpoint_db), exist_ok=True)

    # Prepare completion function with model-specific settings
    completion_kwargs = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": 1337,
    }
    if api_base is not None:
        completion_kwargs |= {
            "api_base": api_base,
            "api_key": api_key,
        }

    traces = {}

    # Process games concurrently
    limiter = asyncio.Semaphore(concurrency)
    async with asyncio.TaskGroup() as tg:
        tasks = []
        for game_id in valid_game_ids:
            tasks.append(
                tg.create_task(
                    utils.limited(
                        limiter,
                        _run_game,
                        ds,
                        checkpoint_db,
                        template,
                        game_id,
                        image_before_text,
                        adapter_type,
                        completion_kwargs,
                    )
                )
            )

        logger.info("Launched tasks", total_tasks=len(valid_game_ids))
        # Process completed futures
        i = 0
        async for task in tqdm_asyncio(
            asyncio.as_completed(tasks),
            total=len(valid_game_ids),
            desc="Evaluating Games",
        ):
            trace: schema.Trace = await task  # type: ignore
            traces[trace.game_id] = trace
            logger.info(
                "Game evaluation completed",
                game_id=trace.game_id,
                solved=trace.solved,
                attempts=trace.attempts,
                same_franchise_at=trace.same_franchise_at,
                total_time=sum(trace.step_times),
                total_tokens=trace.total_tokens,
            )
            # Remove the periodic metric calculation, calculate once after all done.
            if (i + 1) % 5 == 0:
                metrics = utils.calculate_metrics(traces)
                logger.info("Metrics till now", metrics=metrics, games_completed=i + 1)
            i += 1

    if not traces:
        logger.warning("No traces were generated. Cannot calculate metrics.")
        return 1

    metrics = utils.calculate_metrics(traces)
    logger.info("Final metrics", metrics=metrics)

    # Write traces to output file
    output_dir = os.path.dirname(output)
    os.makedirs(output_dir, exist_ok=True)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens_used = 0
    for trace_obj in traces.values():
        total_prompt_tokens += trace_obj.prompt_tokens
        total_completion_tokens += trace_obj.completion_tokens
        total_tokens_used += trace_obj.total_tokens

    out = schema.Output(
        metadata=schema.EvalMetadata(
            model=model,
            model_params=completion_kwargs,
            prompt=template,
            dataset=dataset_path,
            timestamp=time.time(),
            games_count=len(traces),
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_tokens_used,
        ),
        metrics=metrics,
        traces=traces,
    )

    with open(output, "w") as f:
        f.write(out.model_dump_json(indent=2))

    logger.info("Wrote traces to output file", path=output)
    return 0


if __name__ == "__main__":
    app = utils.AsyncTyper(add_completion=False)
    app.command()(main)
    app()
