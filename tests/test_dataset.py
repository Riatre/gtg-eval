import os
import pathlib
import pytest
from unittest import mock

from gtg_eval.dataset import Dataset
from gtg_eval import schema


@pytest.fixture
def real_dataset():
    base_path = pathlib.Path(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return Dataset(base_path / "data" / "20250329")


class TestDatasetInitialization:
    def test_init_with_real_data(self, real_dataset):
        assert isinstance(real_dataset, Dataset)
        assert len(real_dataset.games) > 0
        assert all(
            isinstance(game, schema.Game) for game in real_dataset.games.values()
        )
        assert real_dataset.games["1"].answers == ["Mass Effect 3", "Mass Effect III"]

    def test_init_with_nonexistent_path(self):
        with pytest.raises(FileNotFoundError):
            Dataset("/nonexistent/path")


class TestDatasetAccess:
    def test_getitem(self, real_dataset):
        game_id = next(iter(real_dataset.games.keys()))
        game = real_dataset[game_id]
        assert isinstance(game, schema.Game)
        assert game.id == game_id

    def test_getitem_nonexistent(self, real_dataset):
        with pytest.raises(KeyError):
            real_dataset["nonexistent_id"]

    def test_len(self, real_dataset):
        assert len(real_dataset) == len(real_dataset.games)

    def test_iter(self, real_dataset):
        games = list(real_dataset)
        assert len(games) == len(real_dataset.games)
        assert all(isinstance(game, schema.Game) for game in games)


class TestScreenshotRetrieval:
    @pytest.mark.asyncio
    async def test_screenshot_of_with_image(self, real_dataset):
        screenshot = await real_dataset.screenshot_of("33", 1)
        assert screenshot.read() is not None
        screenshot = await real_dataset.screenshot_of("33", 6)
        assert screenshot.read() is not None
        video = await real_dataset.screenshot_of("888", 6, allow_video=True)
        assert video.read() is not None

    @pytest.mark.asyncio
    async def test_extract_video_frame(self, real_dataset):
        video_frame = await real_dataset.screenshot_of("888", 6, allow_video=False)
        assert video_frame.read() is not None

    @pytest.mark.asyncio
    async def test_screenshot_of_nonexistent(self, real_dataset):
        with pytest.raises(FileNotFoundError):
            await real_dataset.screenshot_of("bruh", 4)

    @pytest.mark.asyncio
    async def test_screenshot_of_invalid_number(self, real_dataset):
        with pytest.raises(ValueError):
            await real_dataset.screenshot_of("1", 0)  # Too low

        with pytest.raises(ValueError):
            await real_dataset.screenshot_of("1", 7)  # Too high


class TestDatasetValidation:
    def test_validate_all_valid(self, real_dataset):
        assert real_dataset.validate() is True

    def test_validate_missing_directory(self, real_dataset):
        def mock_exists(path):
            # Make one game directory not exist
            if str(path).endswith("/1"):
                return False
            return True

        with mock.patch("pathlib.Path.exists", mock_exists):
            with mock.patch("pathlib.Path.is_dir", return_value=True):
                assert real_dataset.validate() is False

    def test_validate_missing_files(self, real_dataset):
        def mock_exists(path):
            # Make directories exist but files not exist
            if ".webp" in str(path) or ".mp4" in str(path) or ".webm" in str(path):
                return False
            return True

        with mock.patch("pathlib.Path.exists", mock_exists):
            with mock.patch("pathlib.Path.is_dir", return_value=True):
                assert real_dataset.validate() is False
