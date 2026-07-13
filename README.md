# Typed Affordance Reasoning with Frontier VLMs

**ECCV 2026 Workshop X-Reason** — *Visual Perception and Reasoning in the Interactable World*

This project evaluates whether frontier Vision-Language Models (VLMs) can reason about **affordances**
— not just *whether* an action is possible, but the **type** of constraint (physical, functional,
social, or safety) that makes it appropriate or forbidden. Two experiments:

- **Experiment A** (done) — GT-grounded: each VLM sees the full scene + a ground-truth object crop
  and predicts a 7-way typed label, scored against [ADE-Affordance](https://github.com/EmoFuncs/ADE-Affordance).
- **Experiment B** (done) — GT-free: a SAM pipeline proposes regions and the VLMs' **inter-model
  agreement** is the reliability signal, comparing SAM 2 area-ranked vs SAM 3 concept-targeted selection.

We evaluate four standard frontier VLMs — **GPT-5.5, Claude Sonnet 5, Gemini 3.5 Flash,
Llama 4 Maverick** — plus the reasoning model **o4-mini** on the hardest typed judgments.

**Headline finding:** models largely agree on *whether* an action is possible but diverge sharply on
*why* it is not — typed attribution, not the binary judgment, is where today's VLMs disagree, and
explicit chain-of-thought helps but is not necessary.

---

## Affordance Taxonomy

| Code | Category | Description |
|------|----------|-------------|
| 0 | **Positive** | Action is appropriate and feasible |
| 1 | **Firmly Negative** | Clearly inappropriate, no specific reason |
| 2 | **Object Non-functional** | Object condition (broken, depleted) prevents it |
| 3 | **Physical Obstacle** | A scene constraint blocks it |
| 4 | **Socially Awkward** | Possible but contextually inappropriate |
| 5 | **Socially Forbidden** | Violates a strong social/legal norm |
| 6 | **Dangerous** | Poses a physical safety risk |

Exception categories (2–6) require a one-sentence explanation and consequence grounded in the image.

---

## Repository Structure

```
aff_reason_llm/
├── experiments/
│   ├── experiment_a/                # Exp A — GT-grounded typed eval (DONE)
│   │   ├── README.md                   # ← run guide + headline numbers
│   │   ├── eval_experiment_a_vision.py # runner (full image + GT crop → 7-way)
│   │   ├── build_instance_masks.py     # one-time ADE20K data prep
│   │   ├── export_raw_results.py       # cache → compact per-model JSONL
│   │   ├── ade_parsing.py, metrics_relationship.py, metrics_caption.py
│   │   ├── configs/llms.json
│   │   └── results/                    # committed raw predictions (raw_*.jsonl) + scorer + summaries
│   │
│   ├── experiment_b/                # Exp B — GT-free agreement pipeline (DONE)
│   │   ├── HOW_TO_RUN_EXP_B.md          # ← full run guide (setup, smoke tests, cost)
│   │   ├── experiment_b_run_v2.py       # runner (sam2_area / sam3_concept / mock)
│   │   ├── experiment_b_agreement.py    # N-way / pairwise agreement + consensus scorer
│   │   ├── make_example.py              # qualitative figure builder
│   │   ├── snapshot_results.py          # runner output → tracked results/
│   │   ├── download_sam.py              # self-downloads SAM 2 + SAM 3 weights
│   │   ├── vision_llm_clients.py        # hardened REST clients (retry, Flex tier, thinking-off)
│   │   ├── configs/llms.json, action_concepts.json
│   │   └── results/                     # released raw predictions + agreement summaries
│   │
│   └── experiment_b_bundle/images/  # 200 ADE20K val scenes for Exp B (committed)
│
├── README.md
├── requirements.txt
└── LICENSE
```

The **raw per-model predictions are committed** — `experiments/experiment_a/results/raw_*.jsonl` and
`experiments/experiment_b/results/*.jsonl` — so every reported metric reproduces from the repo with no
dataset download or API call (each row carries `gt` + `pred`). Large or regenerable artifacts are
**git-ignored**: `overleaf/` (the paper lives on Overleaf), `datasets/`, `experiments/experiment_a_bundle/`
(images/seg/GT — regenerable via `build_instance_masks.py`), the `cache_a_vision/` / `cache_b/` caches,
`experiments/experiment_b/checkpoints/` (SAM weights), and `experiments/experiment_b/legacy/`.

---

## Experiment A (done)

200 ADE20K images / 13,512 (instance, action) pairs, scored vs ADE-Affordance. Full run guide and the
reproduce-from-raw commands are in **[`experiments/experiment_a/README.md`](experiments/experiment_a/README.md)**.
The **committed** raw predictions (one row per model per instance, with `gt`, `pred`, and each model's
explanation) live in [`experiments/experiment_a/results/`](experiments/experiment_a/results/). All four
standard models cover all 13,512 pairs (100%); recompute the paper's numbers with no download or API key:

```bash
cd experiments/experiment_a/results && python3 score_from_raw.py   # mAcc-7 / mAcc-3 per model
```

Headline (mAcc-7 / mAcc-3): Claude Sonnet 5 0.289 / 0.504 · Gemini 3.5 Flash 0.277 / 0.531 ·
GPT-5.5 0.251 / 0.480 · Llama 4 Maverick 0.240 / 0.471. On the exception subset, standard Claude
(mAcc-3 0.763) matches/exceeds the reasoning model o4-mini (0.701).

## Experiment B (done, GPU)

GT-free inter-model agreement over a SAM segment-then-query pipeline. Full run guide (setup, smoke
tests, cost, resume) in **[`experiments/experiment_b/HOW_TO_RUN_EXP_B.md`](experiments/experiment_b/HOW_TO_RUN_EXP_B.md)**.
Released predictions + agreement summaries are in
[`experiments/experiment_b/results/`](experiments/experiment_b/results/); recompute with:

```bash
cd experiments/experiment_b
STD=gpt_5_5,claude_sonnet_5,gemini_3_5_flash,llama_4_maverick
python3 experiment_b_agreement.py --outdir results --mode sam2_area    --K 3 --models $STD
python3 experiment_b_agreement.py --outdir results --mode sam3_concept --K 3 --models $STD
```

4-way agreement (K=3): **SAM 2 area 7.2%** (pairwise 36.0%) vs **SAM 3 concept 23.6%** (pairwise 46.2%).
Concept-targeting raises agreement but lowers the exception rate — the models agree on *whether*, not
*why*. Re-running the pipeline needs **both** a GPU (SAM) and the 4 API keys (the VLMs do the labelling).

---

## Citation

```bibtex
@inproceedings{affbench2026,
  title     = {Can Frontier Vision-Language Models Reason About the Interactable World?
               A Typed Affordance Evaluation},
  author    = {TBD},
  booktitle = {ECCV 2026 Workshop on Visual Perception and Reasoning in the
               Interactable World (X-Reason)},
  year      = {2026}
}
```
