import io
import pathlib
import asyncio
from typing import BinaryIO, Iterator

from loguru import logger

from gtg_eval import schema


class Dataset:
    def __init__(self, path: str | pathlib.Path):
        """Initialize the dataset from the given path.

        Args:
            path: Path to the dataset directory containing gtg_puzzles.jsonl and game folders
        """
        self.path = pathlib.Path(path)
        self.games: dict[str, schema.Game] = {}
        # Cache for extracted video frames
        self._frame_cache = {}

        # Load the JSONL file
        jsonl_path = self.path / "gtg_puzzles.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found at {jsonl_path}")

        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    game = schema.Game.model_validate_json(line)
                    self.games[game.id] = game

        logger.info("Dataset loaded", path=str(jsonl_path), game_count=len(self.games))

    async def screenshot_of(
        self, game: schema.Game | str, nth: int, *, allow_video: bool = False
    ) -> BinaryIO:
        """Get the nth screenshot of the given game.

        Args:
            game: Game object or game ID
            nth: Screenshot number (1-6)
            allow_video: If True, return video file if no image is available

        Returns:
            File-like object containing the image data
        """
        if isinstance(game, str):
            game_id = game
        else:
            game_id = game.id

        if nth < 1 or nth > 6:
            raise ValueError(f"Screenshot number must be between 1 and 6, got {nth}")

        # Check for image file
        img_path = self.path / game_id / f"{nth}.webp"
        if img_path.exists():
            logger.debug(
                "Found image", game_id=game_id, screenshot=nth, path=str(img_path)
            )
            return open(img_path, "rb")

        # Check for video file
        video_extensions = [".mp4", ".webm"]
        video_path = None
        for ext in video_extensions:
            path = self.path / game_id / f"{nth}{ext}"
            if path.exists():
                video_path = path
                logger.debug(
                    "Found video", game_id=game_id, screenshot=nth, path=str(path)
                )
                break

        if video_path is None:
            raise FileNotFoundError(
                f"No image or video found for game {game_id}, screenshot {nth}"
            )

        if allow_video:
            return open(video_path, "rb")

        # Check if frame is in cache
        cache_key = f"{game_id}_{nth}"
        if cache_key in self._frame_cache:
            logger.debug("Using cached frame", game_id=game_id, screenshot=nth)
            # Create a new BytesIO with the cached data to avoid position issues
            return io.BytesIO(self._frame_cache[cache_key])

        # Extract first frame from video and convert to webp
        logger.debug(
            "Extracting frame from video",
            game_id=game_id,
            screenshot=nth,
            video_path=str(video_path),
        )
        # Use ffmpeg to extract the first frame
        result = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-f",
            "image2pipe",
            "-c:v",
            "webp",
            "-lossless",
            "0",
            "-compression_level",
            "6",
            "-quality",
            "90",
            "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await result.communicate()
        if result.returncode != 0:
            raise RuntimeError(f"Failed to extract frame from video: {stderr}")

        # Store the frame data in cache
        self._frame_cache[cache_key] = result.stdout

        # Return a new BytesIO with the frame data
        return io.BytesIO(stdout)

    def __getitem__(self, idx: str) -> schema.Game:
        """Get a game by ID.

        Args:
            idx: Game ID (str)

        Returns:
            Game object
        """
        if idx not in self.games:
            raise KeyError(f"Game with ID {idx} not found in dataset")
        return self.games[idx]

    def __len__(self) -> int:
        """Get the number of games in the dataset.

        Returns:
            Number of games
        """
        return len(self.games)

    def __iter__(self) -> Iterator[schema.Game]:
        """Iterate over all games in the dataset.

        Returns:
            Iterator over Game objects
        """
        return iter(self.games.values())

    def validate(self) -> bool:
        """Validate that all games have the required files.

        Returns:
            True if all games have the required files, False otherwise
        """
        valid = True
        logger.info("Validating dataset", game_count=len(self.games))

        missing_dirs = []
        missing_files = []

        for game_id in self.games:
            game_dir = self.path / game_id

            if not game_dir.exists() or not game_dir.is_dir():
                logger.warning("Game directory not found", game_id=game_id)
                missing_dirs.append(game_id)
                valid = False
                continue

            for i in range(1, 7):
                img_path = game_dir / f"{i}.webp"
                if img_path.exists():
                    continue

                video_found = False
                # Check for video files
                for ext in [".mp4", ".webm"]:
                    video_path = game_dir / f"{i}{ext}"
                    if video_path.exists():
                        video_found = True
                        break

                if not video_found:
                    logger.warning("Missing screenshot", game_id=game_id, screenshot=i)
                    missing_files.append((game_id, i))
                    valid = False

        if valid:
            logger.success("Dataset validation successful", game_count=len(self.games))
        else:
            logger.error(
                "Dataset validation failed",
                missing_directories=len(missing_dirs),
                missing_files=len(missing_files),
            )

        return valid
