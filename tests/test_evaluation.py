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
    progress,
    normalize_title,
)
from gtg_eval import schema, dataset


@pytest.mark.parametrize(
    "completion,expected",
    [
        ("<answer>The Legend of Zelda</answer>", "The Legend of Zelda"),
        ("I think it's <answer>Halo 3</answer>", "Halo 3"),
        (
            "Not sure, but my guess is <answer>Final Fantasy VII</answer>.",
            "Final Fantasy VII",
        ),
        ("There is no answer here", None),
        ("Incomplete <answer> tag", None),
        ("", None),
        (
            "我的猜测是 <answer>Ghost of Tsushima</answer>。\n理由：第二张图片展示了樱花树、飘落的花瓣和雪山背景，这些元素与《对马岛之魂》（Ghost of Tsushima）中标志性的樱花场景高度吻合。游戏以日本战国时期的对马岛为背景，樱花作为核心视觉符号贯穿整个游戏，营造出独特的东方美学氛围。此外，Metacritic评分为89%也符合该游戏的实际评分（原版为83%，但可能指重制版或特定版本）。虽然第一次猜测错误，但结合樱花主题和高评分，此推测更为合理。",
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
@pytest.mark.asyncio
async def test_normalize_answer(answer, expected):
    assert await normalize_answer(answer) == expected


@pytest.fixture
def test_game():
    return schema.Game(
        id="1",
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

    # Create a AsyncMock for screenshot_of method
    mock_ds.screenshot_of = mock.AsyncMock()

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

    @pytest.mark.asyncio
    async def test_build_initial_prompt(self, test_dataset, test_template, test_game):
        state = schema.EvaluationState(game=test_game)

        result = await build_next_prompt(test_dataset, state, test_template)

        # Compare with golden file
        self._compare_with_golden_file(result, "build_next_prompt_initial.json")

        # Verify the screenshot was requested
        test_dataset.screenshot_of.assert_called_once_with(
            test_game, 1, allow_video=False
        )

    @pytest.mark.asyncio
    async def test_build_subsequent_prompt_incorrect(
        self, test_dataset, test_template, test_game
    ):
        # Test with one incorrect guess
        guess = schema.Guess(
            answer="Wrong Game",
            verdict=schema.Verdict.INCORRECT,
        )
        state = schema.EvaluationState(game=test_game, guesses=[guess])

        result = await build_next_prompt(test_dataset, state, test_template)

        # Compare with golden file
        self._compare_with_golden_file(result, "build_next_prompt_incorrect.json")

        # Verify the screenshot was requested for the next guess
        test_dataset.screenshot_of.assert_called_once_with(
            test_game, 2, allow_video=False
        )

    @pytest.mark.asyncio
    async def test_build_subsequent_prompt_same_franchise(
        self, test_dataset, test_template, test_game
    ):
        # Test with one same-franchise guess
        guess = schema.Guess(
            answer="Same Franchise Game",
            verdict=schema.Verdict.SAME_FRANCHISE,
        )
        state = schema.EvaluationState(game=test_game, guesses=[guess])

        result = await build_next_prompt(test_dataset, state, test_template)

        # Compare with golden file
        self._compare_with_golden_file(result, "build_next_prompt_same_franchise.json")

        # Verify the screenshot was requested for the next guess
        test_dataset.screenshot_of.assert_called_once_with(
            test_game, 2, allow_video=False
        )

    @pytest.mark.asyncio
    async def test_build_prompt_with_multiple_guesses(
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

        result = await build_next_prompt(test_dataset, state, test_template)

        # Compare with golden file
        self._compare_with_golden_file(
            result, "build_next_prompt_multiple_guesses.json"
        )

        # Verify the screenshot was requested for the next guess
        test_dataset.screenshot_of.assert_called_once_with(
            test_game, 4, allow_video=False
        )

    @pytest.mark.asyncio
    async def test_build_prompt_with_allow_video(
        self, test_dataset, test_template, test_game
    ):
        state = schema.EvaluationState(game=test_game, guesses=[])

        result = await build_next_prompt(
            test_dataset, state, test_template, allow_video=True
        )

        # Compare with golden file
        self._compare_with_golden_file(result, "build_next_prompt_allow_video.json")

        # Verify the screenshot was requested with allow_video=True
        test_dataset.screenshot_of.assert_called_once_with(
            test_game, 1, allow_video=True
        )

    @pytest.mark.asyncio
    async def test_build_prompt_with_invalid_state(
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
            await build_next_prompt(test_dataset, state, test_template)

    @pytest.mark.asyncio
    async def test_build_prompt_with_correct_verdict(
        self, test_dataset, test_template, test_game
    ):
        # Test with a correct guess, which should not be allowed
        guesses = [
            schema.Guess(answer="Test Game", verdict=schema.Verdict.CORRECT),
        ]
        state = schema.EvaluationState(game=test_game, guesses=guesses)

        with pytest.raises(AssertionError):
            await build_next_prompt(test_dataset, state, test_template)


@pytest.mark.asyncio
async def test_progress_initial_incorrect(test_dataset, test_template, test_game):
    # Create a mock completion function that returns an incorrect answer
    async def mock_completion(messages):
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I think this is <answer>Wrong Game</answer>. The image shows...",
                    }
                }
            ]
        }

    # Initial state with no guesses
    state = schema.EvaluationState(game=test_game)

    # Run progress with mock completion
    result = await progress(test_dataset, state, test_template, mock_completion)

    # Verify the result
    assert len(result.guesses) == 1
    assert result.guesses[0].answer == "Wrong Game"
    assert result.guesses[0].verdict == schema.Verdict.INCORRECT
    assert len(result.messages) == 2  # Initial prompt + model response
    assert not result.done
    assert not result.solved
    assert result.attempts == 1
    assert result.same_franchise_at is None


@pytest.mark.asyncio
async def test_progress_initial_correct(test_dataset, test_template, test_game):
    # Create a mock completion function that returns a correct answer
    async def mock_completion(messages):
        assert len(messages) == 1
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I think this is <answer>Test Game</answer>. The image shows...",
                    }
                }
            ]
        }

    # Initial state with no guesses
    state = schema.EvaluationState(game=test_game)

    # Run progress with mock completion
    result = await progress(test_dataset, state, test_template, mock_completion)

    # Verify the result
    assert len(result.guesses) == 1
    assert result.guesses[0].answer == "Test Game"
    assert result.guesses[0].verdict == schema.Verdict.CORRECT
    assert len(result.messages) == 2  # Initial prompt + model response
    assert result.done  # Should be done because the answer is correct
    assert result.solved  # Should be solved because the answer is correct
    assert result.attempts == 1
    assert result.same_franchise_at == 1


@pytest.mark.asyncio
async def test_progress_same_franchise(test_dataset, test_template, test_game):
    # Create a mock completion function that returns a same franchise answer
    async def mock_completion(messages):
        assert len(messages) == 1
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I think this is <answer>Same Franchise Game</answer>. The image shows...",
                    }
                }
            ]
        }

    # Mock the normalize_answer function to return a same franchise result
    with mock.patch(
        "gtg_eval.evaluation.normalize_answer",
        return_value=NormalizedAnswer(
            name="Same Franchise Game", franchise="Test Franchise"
        ),
    ):
        # Initial state with no guesses
        state = schema.EvaluationState(game=test_game)

        # Run progress with mock completion
        result = await progress(test_dataset, state, test_template, mock_completion)

        # Verify the result
        assert len(result.guesses) == 1
        assert result.guesses[0].answer == "Same Franchise Game"
        assert result.guesses[0].verdict == schema.Verdict.SAME_FRANCHISE
        assert len(result.messages) == 2  # Initial prompt + model response
        assert not result.done  # Should not be done because the answer is not correct
        # Should not be solved because the answer is not correct
        assert not result.solved
        assert result.same_franchise_at == 1  # First guess was same franchise


@pytest.mark.asyncio
async def test_progress_subsequent_step(test_dataset, test_template, test_game):
    # Create a mock completion function for the second step
    async def mock_completion(messages):
        # Should have initial prompt + first response + second prompt
        assert len(messages) == 3
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Now I think it's <answer>Test Game</answer>. The metacritic score helps...",
                    }
                }
            ]
        }

    # Initial state with one incorrect guess
    initial_state = schema.EvaluationState(
        game=test_game,
        guesses=[schema.Guess(answer="Wrong Game", verdict=schema.Verdict.INCORRECT)],
        messages=[
            {"role": "user", "content": ["Initial prompt"]},
            {
                "role": "assistant",
                "content": "I think this is <answer>Wrong Game</answer>.",
            },
        ],
    )

    # Run progress with mock completion
    result = await progress(
        test_dataset,
        state=initial_state,
        template=test_template,
        completion=mock_completion,
    )

    # Verify the result
    assert len(result.guesses) == 2
    assert result.guesses[0].answer == "Wrong Game"
    assert result.guesses[0].verdict == schema.Verdict.INCORRECT
    assert result.guesses[1].answer == "Test Game"
    assert result.guesses[1].verdict == schema.Verdict.CORRECT
    assert len(result.messages) == 4  # 2 initial + 2 new
    assert result.done  # Should be done because the second answer is correct
    assert result.solved  # Should be solved because the second answer is correct
    assert result.attempts == 2  # Two attempts were made


@pytest.mark.asyncio
async def test_progress_no_answer_tag(test_dataset, test_template, test_game):
    # Create a mock completion function that returns a response without <n> tags
    async def mock_completion(messages):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I'm not sure what game this is. The image is unclear.",
                    }
                }
            ]
        }

    # Initial state with no guesses
    state = schema.EvaluationState(game=test_game)

    # Run progress with mock completion
    result = await progress(test_dataset, state, test_template, mock_completion)

    # Verify the result
    assert len(result.guesses) == 1
    assert result.guesses[0].answer == ""  # Empty answer when no tags found
    assert result.guesses[0].verdict == schema.Verdict.INCORRECT
    assert len(result.messages) == 2  # Initial prompt + model response


@pytest.mark.asyncio
async def test_progress_with_done_state(test_dataset, test_template, test_game):
    # Create a mock completion function (should not be called)
    async def mock_completion(messages):
        pytest.fail("Completion function should not be called when state is done")

    # Create a state with a correct guess (done state)
    state = schema.EvaluationState(
        game=test_game,
        guesses=[schema.Guess(answer="Test Game", verdict=schema.Verdict.CORRECT)],
    )
    assert state.done

    # Run progress with mock completion should raise AssertionError
    with pytest.raises(AssertionError):
        await progress(test_dataset, state, test_template, mock_completion)

    # Create a state with 6 guesses (done state)
    state = schema.EvaluationState(
        game=test_game,
        guesses=[schema.Guess(answer="Wrong Game", verdict=schema.Verdict.INCORRECT)]
        * 6,
    )
    assert state.done

    # Run progress with mock completion should raise AssertionError
    with pytest.raises(AssertionError):
        await progress(test_dataset, state, test_template, mock_completion)


@pytest.mark.parametrize(
    "input_title,expected_output",
    [
        ("Overcooked! 2", "overcooked 2"),
        ("Pokémon Scarlet", "pokemon scarlet"),
        ("God of War", "god of war"),
        ("Assassin's Creed", "assassin's creed"),
        ("SUPER MARIO BROS.", "super mario bros"),
        ("Half-Life 2", "half life 2"),
        (
            "The Legend of Zelda: Breath of the Wild",
            "the legend of zelda breath of the wild",
        ),
        ("Tony Hawk's Pro Skater", "tony hawk's pro skater"),
        ("F.E.A.R.", "fear"),
        ("Metal Gear Solid 3: Snake Eater", "metal gear solid 3 snake eater"),
        ("Sid Meier's Civilization VI", "sid meier's civilization vi"),
        ("Elden Ring (Extended Edition)", "elden ring extended edition"),
        ("Resident Evil 7: Biohazard", "resident evil 7 biohazard"),
        ("Nier:Automata", "nier automata"),
        ("Nier: Automata", "nier automata"),
        ("Pokémon GO", "pokemon go"),
        ("Ōkami", "okami"),
        ("Viva Piñata", "viva pinata"),
        ("Viva Piñata", "viva pinata"),
        ("Brütal Legend", "brutal legend"),
        ("Star Wars: Battlefront (2004)", "star wars battlefront 2004"),
        ("Star Wars Battlefront (2004)", "star wars battlefront 2004"),
        ("God of War: Ragnarök", "god of war ragnarok"),
        ("God of War Ragnarök", "god of war ragnarok"),
        ("Dante’s Inferno", "dante's inferno"),
        ("Super Mario Bros.", "super mario bros"),
        ("Need for Speed: Unbound", "need for speed unbound"),
        ("Need For Speed Unbound", "need for speed unbound"),
        ("Sam & Max Hit the Road", "sam max hit the road"),
        ("Sam & Max: Hit the Road", "sam max hit the road"),
        ("Monster Hunter: Stories", "monster hunter stories"),
        (
            "Star Wars: Knights of the Old Republic II - The Sith Lords",
            "star wars knights of the old republic ii the sith lords",
        ),
        (
            "Star Wars: Knights of the Old Republic II: The Sith Lords",
            "star wars knights of the old republic ii the sith lords",
        ),
        ("South Park: Snow Day!", "south park snow day"),
        ("Mario + Rabbids: Kingdom Battle", "mario rabbids kingdom battle"),
        ("Gauntlet: Legends", "gauntlet legends"),
        ("Astro Bot: Rescue Mission", "astro bot rescue mission"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_title(input_title, expected_output):
    """Test that normalize_title properly cleans game titles."""
    result = normalize_title(input_title)
    assert result == expected_output
