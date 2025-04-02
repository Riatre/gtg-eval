## What / Why

I was curious about how well VLMs can play [Guess The Game](https://guessthe.game) <sub>and it's too easy to vibe coding things like this so why not.</sub>

This repository contains code for evaluating models on all of the games available on the website.

**Note that since the game data is publicly available online, it likely appears in the training data of many models. Do not take the results too seriously.** Filter out rounds dated before the knowledge cutoff of a model may help, but there's probably a lot more work to do besides this if you want to make any sense out of the results.

The last clue of each round may be a video, in this case we extract the first frame as a screenshot.

`prompt_template/english-v3-nohint.json` implements a harder variant (that is, no gradual revealing of metacritic score, genre, platform, year, developer etc) of GTG as I realized that sometimes it's enough to figure out the game by Metacritics score + developer without reading screenshots at all.

## Results

There are a few traces in the `traces` directory. I lost interest in burning $ and running against other proprietary models after seeing that Gemini 2.0 Flash is already too good on this.

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
