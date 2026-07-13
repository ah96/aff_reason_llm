# Experiment A — raw results

Self-contained record of every VLM prediction, so all metrics can be recomputed and every
model's answer/explanation can be extracted, without the (54k-file) cache or any API call.

**These `raw_*.jsonl` are committed to the repo.** The paper's mAcc-7 / mAcc-3 (Tables 1 and 4)
reproduce exactly from them via `score_from_raw.py` — no cache, dataset download, or API key needed
(each row carries both `gt` and `pred`). The heavy inputs (`experiment_a_bundle/`) stay git-ignored
and are regenerable with `build_instance_masks.py`; you only need them to *re-run* the VLM queries.

## Files
- `raw_<model>.jsonl` — one line per prediction, per model. Fields:
  | field | meaning |
  |---|---|
  | `model` | which VLM produced it (`gpt_5_5`, `claude_sonnet_5`, `gemini_3_5_flash`, `llama_4_maverick`, `o4_mini`) |
  | `image` | ADE20K image id (`ADE_train_XXXXXXXX`) |
  | `action` | `sit` / `run` / `grasp` (ADE-Affordance actions) |
  | `region_id` | ADE-Affordance instance id (blue channel of the object seg) |
  | `gt` | ground-truth code, **canonical taxonomy** (see below) |
  | `pred` | the model's predicted code, same taxonomy |
  | `explanation` | the model's one-sentence explanation (exception cases) |
  | `consequence` | the model's one-sentence predicted consequence |
- `all_predictions.jsonl` — all models concatenated (git-ignored: redundant 13 MB; regenerate with
  `cat raw_*.jsonl`. `score_from_raw.py` reads the per-model `raw_*.jsonl` directly, not this).
- `results_a_200.json` — headline run metrics (mAcc-7/mAcc-3 + BLEU/METEOR/ROUGE/CIDEr on explanation & consequence).
- `results_a_reasoning.json` — the exception-subset run (all 5 models, incl. o4-mini).
- `results_a_50.json`, `clean_results.json` — earlier/aggregate metric snapshots.
- `score_from_raw.py` — recompute mAcc from the raw jsonl (no deps).

## Taxonomy codes (canonical)
`0` Positive · `1` Firmly Negative · `2` Object Non-functional · `3` Physical Obstacle ·
`4` Socially Awkward · `5` Socially Forbidden · `6` Dangerous.
(ADE-Affordance's on-disk codes are rotated; `gt` here is already converted via `(file+1)%7`.)

## Recompute metrics
```bash
python3 score_from_raw.py                 # mAcc-3 / mAcc-7 per model
python3 score_from_raw.py --exceptions_only   # over GT exception instances only (the reasoning subset)
```

## Extract what a model said
```bash
# every exception explanation Claude wrote
python3 -c "import json;[print(r['image'],r['action'],'->',r['explanation']) for r in map(json.loads,open('raw_claude_sonnet_5.jsonl')) if 2<=r['pred']<=6]"
# or with jq:
jq -r 'select(.pred>=2) | [.image,.action,.explanation] | @tsv' raw_claude_sonnet_5.jsonl
```

## Coverage (2026-07-13, complete)
All four standard models at **13,512 / 13,512 (100%)**: gpt_5_5, claude_sonnet_5, gemini_3_5_flash,
llama_4_maverick · o4_mini 579 (exception subset). Gemini's earlier free-tier gaps were filled after
enabling billing, so no model is now sub-100%.
