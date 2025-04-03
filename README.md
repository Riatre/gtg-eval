## What / Why

I was curious about how well VLMs can play [Guess The Game](https://guessthe.game) <sub>and it's too easy to vibe coding things like this so why not.</sub>

This repository contains code for evaluating models on all of the games available on the website.

**Note that since the game data is publicly available online, it likely appears in the training data of many models. Do not take the results too seriously.** Filter out rounds dated before the knowledge cutoff of a model may help, but there's probably a lot more work to do besides this if you want to make any sense out of the results.

The last clue of each round may be a video, in this case we extract the first frame as a screenshot.

`prompt_template/english-v3-nohint.json` implements a harder variant (that is, no gradual revealing of metacritic score, genre, platform, year, developer etc) of GTG as I realized that sometimes it's enough to figure out the game by Metacritics score + developer without reading screenshots at all.

## Results

There are a few traces in the `traces` directory. I lost interest in burning $ and running against other proprietary models after seeing that Gemini 2.0 Flash is already too good on this.

Percentage solved after x screenshots (without hint):

|               Model               |   1  |   2  |   3  |   4  |   5  |   6      |
|-----------------------------------|------|------|------|------|------|----------|
| gemini-2.5-pro-exp-03-25          | 18.3 | 41.0 | 62.1 | 76.2 | 87.4 | **93.9** |
| gemini-2.0-flash                  | 10.3 | 24.7 | 40.3 | 56.2 | 72.4 | **80.2** |
| gemini-2.0-flash-lite             |  9.9 | 22.4 | 36.6 | 49.1 | 64.3 | **72.8** |
| doubao-1.5-vision-pro-32k-250115  |  3.7 |  9.5 | 21.4 | 32.0 | 49.4 | **63.9** |
| gemini-1.5-flash                  |  5.7 | 12.7 | 22.0 | 31.1 | 43.4 | **53.9** |
| gemini-1.5-flash-8b               |  3.7 |  7.9 | 11.6 | 16.0 | 20.9 | **25.1** |

Percentage solved afetr x screenshots (with hint):


|               Model               |   1  |   2  |   3  |   4  |   5  |   6      |
|-----------------------------------|------|------|------|------|------|----------|
| gemini-2.5-pro-exp-03-25          | 18.1 | 53.1 | 80.4 | 91.0 | 97.7 | **99.2** |
| gemini-2.0-flash                  |  9.4 | 24.5 | 47.5 | 68.5 | 88.0 | **96.1** |
| gemini-2.0-flash-lite             |  9.5 | 23.0 | 43.1 | 63.1 | 83.3 | **92.6** |
| gemini-1.5-flash                  |  6.4 | 14.0 | 28.3 | 45.0 | 64.1 | **78.0** |
| gemini-1.5-flash-8b               |  3.0 |  7.1 | 13.6 | 20.1 | 30.8 | **43.5** |

## How

In case someone wants to run this:

```bash
pushd data
  python download_puzzles.py
  deno puzzle_answers_to_jsonl.ts
  python download_raw_screenshots.py -d 20250329
popd
python -m gtg_eval.run \
    --dataset data/20250329 \
    --model 'gemini/gemini-2.5-pro-exp-03-25' \
    --checkpoint-db /tmp/blah.db \
    --output traces/gemini-2.5-pro-exp-03-25-20250329-prompt-v3-nohint.json \
    --prompt-template prompt_template/english-v3-nohint.json \
    --max-tokens 8192 \
    --game-ids 1-1070
```
