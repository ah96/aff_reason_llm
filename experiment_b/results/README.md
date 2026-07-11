# Experiment B — raw results

Self-contained record of every VLM prediction over the SAM pipeline, so all agreement statistics can
be recomputed and every model's answer/region can be extracted — without the cache or any API call.
Same idea as `experiment_a/results/`.

Populate this folder after the runs with: `cd experiment_b && python3 snapshot_results.py`.

## Files
- `<model>_<mode>_K<K>.jsonl` — one line per prediction, per model. `mode` ∈ {`sam2_area`,
  `sam3_concept`}, `K` = region budget. Fields:
  | field | meaning |
  |---|---|
  | `image` | ADE20K val image id (in `../experiment_b_bundle/images/`, committed) |
  | `region_id` | SAM-proposed region index (area-rank for sam2_area; per-action for sam3_concept) |
  | `bbox` | `[x1,y1,x2,y2]` of the SAM region — **SAM's output**, enough to redraw it on the image |
  | `action` | one of sit_on / hold / carry / cut / throw / ride |
  | `relationship_id` | the model's typed label (canonical 0..6, see below) — **the VLM's output** |
  | `explanation` / `consequence` | the model's one sentence each (exception cases) |
  The filename tells you *which model / mode / K*; the row tells you *which image / region / action*.
- `agree_<mode>_K<K>.json` — the agreement summary (4-way / pairwise agreement, per-model
  consensus accuracy, exception rate, `n_majority`) that fills `tab:main` / `tab:selection`.

Everything here is **committed & pushable** — the raw `*.jsonl` are small (~few MB), so they flow
lab PC → git → laptop for analysis and paper writing. (Unlike Exp A's larger raw, which was archived
separately.) Only the runner's working `../experiment_b_bundle/out/` and `cache_b/` stay git-ignored.

## Taxonomy codes
`0` Positive · `1` Firmly Negative · `2` Object Non-functional · `3` Physical Obstacle ·
`4` Socially Awkward · `5` Socially Forbidden · `6` Dangerous. Codes 2–6 are exceptions.

## Recompute the statistics
```bash
cd experiment_b
STD=gpt_5_5,claude_sonnet_5,gemini_3_5_flash,llama_4_maverick
python3 experiment_b_agreement.py --outdir results --mode sam2_area    --K 3 --models $STD
python3 experiment_b_agreement.py --outdir results --mode sam3_concept --K 3 --models $STD
```

## Build a qualitative example (image + SAM region + VLM outputs) for the paper
```bash
cd experiment_b
python3 make_example.py --mode sam2_area --K 3 --list          # find interesting (disagreement/exception) cases
python3 make_example.py --mode sam2_area --K 3 \
    --image ADE_val_00000013.jpg --region 0 --action sit_on    # writes example.png (bbox drawn) + prints all 4 models
```

## Extract what a model said
```bash
# every exception Claude wrote in the area-mode run:
jq -r 'select(.relationship_id>=2) | [.image,.action,.explanation] | @tsv' results/claude_sonnet_5_sam2_area_K3.jsonl
```
