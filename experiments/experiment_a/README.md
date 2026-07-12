# Experiment A — GT-grounded typed affordance evaluation

Each VLM sees the **full scene** plus a **ground-truth object crop** and predicts a 7-way typed
affordance label (0–6) with a one-sentence explanation and consequence for every exception. Predictions
are scored against [ADE-Affordance](https://github.com/EmoFuncs/ADE-Affordance) ground truth. This is
the *label-grounded* counterpart to Experiment B (which is GT-free — see
[`../experiment_b/HOW_TO_RUN_EXP_B.md`](../experiment_b/HOW_TO_RUN_EXP_B.md)).

**Status: done.** 200 ADE20K images / 13,512 (instance, action) pairs / 579 GT exceptions. Released raw
predictions + scorer live in [`results/`](results/) — every metric is reproducible with **no cache or
API call** (see [`results/README.md`](results/README.md)).

## Taxonomy (canonical codes)
`0` Positive · `1` Firmly Negative · `2` Object Non-functional · `3` Physical Obstacle ·
`4` Socially Awkward · `5` Socially Forbidden · `6` Dangerous. Codes 2–6 are exceptions and require a
grounded explanation + consequence. (ADE-Affordance's on-disk codes are rotated; the parser converts via
`(file+1)%7`.)

## Just want the numbers? (no GPU, no keys)
```bash
cd results
python3 score_from_raw.py                  # mAcc-7 / mAcc-3 per model, from raw_<model>.jsonl
python3 score_from_raw.py --exceptions_only # over GT exception instances only (the reasoning subset)
```

Headline (full run, **mAcc-7 / mAcc-3**):

| Model | mAcc-7 | mAcc-3 |
|---|---|---|
| Claude Sonnet 5 | 0.289 | 0.504 |
| Gemini 3.5 Flash | 0.276 | 0.531 |
| GPT-5.5 | 0.251 | 0.480 |
| Llama 4 Maverick | 0.240 | 0.471 |

On the **exception subset** (579 pairs), the standard Claude Sonnet 5 (mAcc-3 **0.763**) matches or
exceeds the dedicated reasoning model o4-mini (**0.701**) — chain-of-thought helps but is not necessary.

## Re-run from scratch (needs API keys)

### 1. Setup
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"   # METEOR needs WordNet
export OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...  GEMINI_API_KEY=...  OPENROUTER_API_KEY=...
```
Models (`configs/llms.json`): `gpt_5_5`, `claude_sonnet_5`, `gemini_3_5_flash` (free tier),
`llama_4_maverick` (OpenRouter), `o4_mini` (reasoning, exception subset only). No GPU required — the
scene/crop images come from the `experiment_a_bundle/` (data + cache are git-ignored, archived separately).

### 2. One-time data prep (builds instance masks + full images)
```bash
python3 build_instance_masks.py            # fetches ADE20K (HF: 1aurent/ADE20K) -> experiment_a_bundle/
```

### 3. Run the evaluation
```bash
# full run (4 standard VLMs, all 200 images) -> writes cache_a_vision/ + metrics JSON
python3 eval_experiment_a_vision.py --actions sit,run,grasp \
  --models gpt_5_5,claude_sonnet_5,gemini_3_5_flash,llama_4_maverick \
  --out results/results_a_200.json --workers 4

# reasoning subset (adds o4-mini, GT exceptions only)
python3 eval_experiment_a_vision.py --gt_exceptions_only \
  --models gpt_5_5,claude_sonnet_5,gemini_3_5_flash,llama_4_maverick,o4_mini \
  --out results/results_a_reasoning.json
```
Interruptible: re-run the same command to resume (cached calls are free). Add `--dry_run` for a
no-keys/no-network smoke test.

### 4. Snapshot raw predictions (so results are reproducible without the cache)
```bash
python3 export_raw_results.py              # cache_a_vision/ -> results/raw_<model>.jsonl
```

## File map
| File | Role |
|---|---|
| `eval_experiment_a_vision.py` | **runner** — full image + GT crop → 7-way typed label (uses `../experiment_b/vision_llm_clients.py`) |
| `build_instance_masks.py` | one-time data prep — ADE20K instance masks + full images |
| `export_raw_results.py` | cache → compact `results/raw_<model>.jsonl` |
| `ade_parsing.py` | ADE20K annotation + `*_relationship.txt` / `*_exco.json` parsing |
| `metrics_relationship.py` | mAcc, 7→3 taxonomy mapping |
| `metrics_caption.py` | BLEU / METEOR / ROUGE / CIDEr for explanation & consequence |
| `results/score_from_raw.py` | released scorer — recompute mAcc from raw jsonl, zero deps |
| `configs/llms.json` | model lineup (API keys resolved from env at runtime) |
