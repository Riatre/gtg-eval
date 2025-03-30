import json
import os
import pathlib
import pytest
from unittest import mock

from gtg_eval.evaluation import (
    parse_answer,
    _TitleCache,
    normalize_answer,
    NormalizedAnswer,
    judge,
    build_next_prompt,
)
from gtg_eval import schema, dataset


@pytest.mark.parametrize(
    "completion,expected",
    [
        ("<name>The Legend of Zelda</name>", "The Legend of Zelda"),
        ("I think it's <name>Halo 3</name>", "Halo 3"),
        (
            "Not sure, but my guess is <name>Final Fantasy VII</name>.",
            "Final Fantasy VII",
        ),
        ("There is no answer here", None),
        ("Incomplete <name> tag", None),
        ("", None),
        (
            "我的猜测是 <name>Ghost of Tsushima</name>。\n理由：第二张图片展示了樱花树、飘落的花瓣和雪山背景，这些元素与《对马岛之魂》（Ghost of Tsushima）中标志性的樱花场景高度吻合。游戏以日本战国时期的对马岛为背景，樱花作为核心视觉符号贯穿整个游戏，营造出独特的东方美学氛围。此外，Metacritic评分为89%也符合该游戏的实际评分（原版为83%，但可能指重制版或特定版本）。虽然第一次猜测错误，但结合樱花主题和高评分，此推测更为合理。",
            "Ghost of Tsushima",
        ),
    ],
)
def test_parse_answer(completion, expected):
    result = parse_answer(completion)
    assert result == expected


class TestTitleCache:
    def test_put_and_get(self, tmp_path: pathlib.Path):
        cache = _TitleCache(cache_dir=tmp_path)

        cache.put("test game", "Test Game", "Test Franchise")
        result = cache.get("test game")

        assert result is not None
        assert result.name == "Test Game"
        assert result.franchise == "Test Franchise"

        # Test case insensitivity
        result = cache.get("TEST GAME")
        assert result is not None
        assert result.name == "Test Game"
        assert result.franchise == "Test Franchise"

    def test_get_nonexistent(self, tmp_path: pathlib.Path):
        cache = _TitleCache(cache_dir=tmp_path)
        result = cache.get("nonexistent game")
        assert result is None

    def test_put_update(self, tmp_path: pathlib.Path):
        cache = _TitleCache(cache_dir=tmp_path)
        cache.put("test game", "Test Game", "Test Franchise")
        cache.put("test game", "Test Game Updated", "Test Franchise Updated")
        result = cache.get("test game")
        assert result is not None
        assert result.name == "Test Game Updated"
        assert result.franchise == "Test Franchise Updated"


@pytest.mark.parametrize(
    "answer,expected",
    [
        (
            "Resident Evil 7: Biohazard",
            NormalizedAnswer(
                name="Resident Evil 7: Biohazard", franchise="Resident Evil"
            ),
        ),
        (
            "Xenoblade Chronicles 2",
            NormalizedAnswer(name="Xenoblade Chronicles 2", franchise="Xenoblade"),
        ),
    ],
)
def test_normalize_answer(answer, expected):
    assert normalize_answer(answer) == expected


@pytest.fixture
def test_game():
    return schema.Game(
        id="1",
        title="Test Game",
        answers=["Test Game", "Test Game 1"],
        franchise="Test Franchise",
        metacritic_score="85",
        console_platform="PC",
        genre="Action",
        release_year="2020",
        developer="Test Developer",
    )


class TestJudge:
    def test_judge_correct(self, test_game):
        # Test correct answer
        normalized = NormalizedAnswer(name="Test Game", franchise="Test Franchise")
        verdict = judge(test_game, normalized)
        assert verdict == schema.Verdict.CORRECT

        # Test alternative correct answer
        normalized = NormalizedAnswer(name="Test Game 1", franchise="Test Franchise")
        verdict = judge(test_game, normalized)
        assert verdict == schema.Verdict.CORRECT

    def test_judge_same_franchise(self, test_game):
        # Test game from the same franchise
        normalized = NormalizedAnswer(name="Other Game", franchise="Test Franchise")
        verdict = judge(test_game, normalized)
        assert verdict == schema.Verdict.SAME_FRANCHISE

    def test_judge_incorrect(self, test_game):
        # Test game from a different franchise
        normalized = NormalizedAnswer(name="Other Game", franchise="Other Franchise")
        verdict = judge(test_game, normalized)
        assert verdict == schema.Verdict.INCORRECT

        # Test game with no franchise
        normalized = NormalizedAnswer(name="Other Game", franchise=None)
        verdict = judge(test_game, normalized)
        assert verdict == schema.Verdict.INCORRECT


@pytest.fixture
def test_dataset():
    mock_ds = mock.MagicMock(spec=dataset.Dataset)

    # Create a MagicMock for screenshot_of method
    mock_ds.screenshot_of = mock.MagicMock()

    # Configure the mock to return a file-like object
    def side_effect(game, nth_guess, allow_video=False):
        mock_file = mock.MagicMock()
        mock_file.read.return_value = (
            f"mock_image_data ({game.id}, {nth_guess}, {allow_video})"
        ).encode("utf-8")
        return mock_file

    mock_ds.screenshot_of.side_effect = side_effect
    return mock_ds


@pytest.fixture
def test_template():
    return schema.PromptTemplate(
        initial="This is the initial prompt.",
        after_1="This is after 1. {verdict} Hint: {hint}",
        after_2="This is after 2. {verdict} Hint: {hint}",
        after_3="This is after 3. {verdict} Hint: {hint}",
        after_4="This is after 4. {verdict} Hint: {hint}",
        after_5="This is after 5. {verdict} Hint: {hint}",
        incorrect="That's incorrect.",
        same_franchise="That's from the same franchise.",
    )


class TestBuildNextPrompt:
    def _compare_with_golden_file(self, result, golden_file_name):
        """Compare the result with a golden file."""
        golden_file_path = os.path.join(
            os.path.dirname(__file__), "golden", golden_file_name
        )

        with open(golden_file_path, "r") as f:
            expected = json.load(f)

        # Compare the entire dictionary
        assert result == expected

    def test_build_initial_prompt(self, test_dataset, test_template, test_game):
        state = schema.EvaluationState(game=test_game)

        result = build_next_prompt(test_dataset, state, test_template)

        # Compare with golden file
        self._compare_with_golden_file(result, "build_next_prompt_initial.json")

        # Verify the screenshot was requested
        test_dataset.screenshot_of.assert_called_once_with(
            test_game, 1, allow_video=False
        )

    def test_build_subsequent_prompt_incorrect(
        self, test_dataset, test_template, test_game
    ):
        # Test with one incorrect guess
        guess = schema.Guess(
            answer="Wrong Game",
            verdict=schema.Verdict.INCORRECT,
        )
        state = schema.EvaluationState(game=test_game, guesses=[guess])

        result = build_next_prompt(test_dataset, state, test_template)

        # Compare with golden file
        self._compare_with_golden_file(result, "build_next_prompt_incorrect.json")

        # Verify the screenshot was requested for the next guess
        test_dataset.screenshot_of.assert_called_once_with(
            test_game, 2, allow_video=False
        )

    def test_build_subsequent_prompt_same_franchise(
        self, test_dataset, test_template, test_game
    ):
        # Test with one same-franchise guess
        guess = schema.Guess(
            answer="Same Franchise Game",
            verdict=schema.Verdict.SAME_FRANCHISE,
        )
        state = schema.EvaluationState(game=test_game, guesses=[guess])

        result = build_next_prompt(test_dataset, state, test_template)

        # Compare with golden file
        self._compare_with_golden_file(result, "build_next_prompt_same_franchise.json")

        # Verify the screenshot was requested for the next guess
        test_dataset.screenshot_of.assert_called_once_with(
            test_game, 2, allow_video=False
        )

    def test_build_prompt_with_multiple_guesses(
        self, test_dataset, test_template, test_game
    ):
        # Test with multiple guesses
        guesses = [
            schema.Guess(answer="Wrong Game 1", verdict=schema.Verdict.INCORRECT),
            schema.Guess(answer="Wrong Game 2", verdict=schema.Verdict.INCORRECT),
            schema.Guess(
                answer="Same Franchise Game", verdict=schema.Verdict.SAME_FRANCHISE
            ),
        ]
        state = schema.EvaluationState(game=test_game, guesses=guesses)

        result = build_next_prompt(test_dataset, state, test_template)

        # Compare with golden file
        self._compare_with_golden_file(
            result, "build_next_prompt_multiple_guesses.json"
        )

        # Verify the screenshot was requested for the next guess
        test_dataset.screenshot_of.assert_called_once_with(
            test_game, 4, allow_video=False
        )

    def test_build_prompt_with_allow_video(
        self, test_dataset, test_template, test_game
    ):
        state = schema.EvaluationState(game=test_game, guesses=[])

        result = build_next_prompt(test_dataset, state, test_template, allow_video=True)

        # Compare with golden file
        self._compare_with_golden_file(result, "build_next_prompt_allow_video.json")

        # Verify the screenshot was requested with allow_video=True
        test_dataset.screenshot_of.assert_called_once_with(
            test_game, 1, allow_video=True
        )

    def test_build_prompt_with_invalid_state(
        self, test_dataset, test_template, test_game
    ):
        # Test with too many guesses
        guesses = [
            schema.Guess(answer="Wrong Game 1", verdict=schema.Verdict.INCORRECT),
            schema.Guess(answer="Wrong Game 2", verdict=schema.Verdict.INCORRECT),
            schema.Guess(answer="Wrong Game 3", verdict=schema.Verdict.INCORRECT),
            schema.Guess(answer="Wrong Game 4", verdict=schema.Verdict.INCORRECT),
            schema.Guess(answer="Wrong Game 5", verdict=schema.Verdict.INCORRECT),
            schema.Guess(answer="Wrong Game 6", verdict=schema.Verdict.INCORRECT),
        ]
        state = schema.EvaluationState(game=test_game, guesses=guesses)

        with pytest.raises(AssertionError):
            build_next_prompt(test_dataset, state, test_template)

    def test_build_prompt_with_correct_verdict(
        self, test_dataset, test_template, test_game
    ):
        # Test with a correct guess, which should not be allowed
        guesses = [
            schema.Guess(answer="Test Game", verdict=schema.Verdict.CORRECT),
        ]
        state = schema.EvaluationState(game=test_game, guesses=guesses)

        with pytest.raises(AssertionError):
            build_next_prompt(test_dataset, state, test_template)
