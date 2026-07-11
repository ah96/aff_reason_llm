# AffBench: Structured Affordance Reasoning with Frontier VLMs

**ECCV 2026 Workshop X-Reason** — *Visual Perception and Reasoning in the Interactable World*

AffBench evaluates whether frontier Vision-Language Models (VLMs) can reason about **affordances**
— not just *whether* an action is possible, but the **type** of constraint (physical, functional,
social, or safety) that makes it appropriate or forbidden. Two experiments:

- **Experiment A** (done) — GT-grounded: each VLM sees the full scene + a ground-truth object crop
  and predicts a 7-way typed label, scored against [ADE-Affordance](https://github.com/EmoFuncs/ADE-Affordance).
- **Experiment B** (ready to run on GPU) — GT-free: a SAM pipeline proposes regions and the VLMs'
  **inter-model agreement** is the reliability signal. Compares SAM 2 area-ranked vs SAM 3
  concept-targeted selection.

We evaluate four standard frontier VLMs — **GPT-5.5, Claude Sonnet 5, Gemini 3.5 Flash,
Llama 4 Maverick** — plus the reasoning model **o4-mini** on the hardest typed judgments.

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
├── experiment_a/                   # Exp A — GT-grounded typed eval (DONE)
│   ├── eval_experiment_a_vision.py     # runner (full image + GT crop → 7-way)
│   ├── score_from_cache.py             # recompute metrics from the cache
│   ├── export_raw_results.py           # cache → compact per-model JSONL
│   ├── metrics_caption.py, ade_parsing.py, build_instance_masks.py
│   ├── configs/llms.json
│   └── results/                        # released raw predictions + scorer (see results/README.md)
│
├── experiment_b/                   # Exp B — GT-free agreement pipeline
│   ├── experiment_b_run_v2.py          # runner (sam2_area / sam3_concept / mock)
│   ├── experiment_b_agreement.py       # 4-way / pairwise agreement + consensus scorer
│   ├── download_sam.py                 # self-downloads SAM 2 + SAM 3 weights
│   ├── vision_llm_clients.py           # hardened REST clients (retry, Flex tier, thinking-off)
│   ├── configs/llms.json, action_concepts.json
│   ├── requirements.txt
│   ├── HOW_TO_RUN_EXP_B.md             # ← full run guide (setup, smoke tests, cost)
│   └── legacy/                         # archived OOAL/saliency design (git-ignored)
│
├── experiment_b_bundle/images/     # 200 ADE20K val scenes for Exp B (committed)
└── README.md
```

Large / regenerated artifacts are **git-ignored** and archived separately: `overleaf/` (the paper
lives on Overleaf), `datasets/`, `experiment_a_bundle/`, `experiment_a/cache_a_vision/`,
`experiment_a/results/*.jsonl`, `experiment_b/checkpoints/` (SAM weights), `cache_b/`, and
`experiment_b_bundle/out/`.

---

## Experiment A (done)

200 ADE20K images / 13,512 (instance, action) pairs, scored vs ADE-Affordance. Released raw
predictions (one row per model per instance, with each model's explanation) live in
[`experiment_a/results/`](experiment_a/results/) — recompute any metric with:

```bash
cd experiment_a/results && python3 score_from_raw.py          # mAcc-3 / mAcc-7 per model
```

Headline (mAcc-7 / mAcc-3): Claude Sonnet 5 0.289 / 0.504 · Gemini 3.5 Flash 0.276 / 0.531 ·
GPT-5.5 0.251 / 0.480 · Llama 4 Maverick 0.240 / 0.471.

## Experiment B (GPU)

See **[experiment_b/HOW_TO_RUN_EXP_B.md](experiment_b/HOW_TO_RUN_EXP_B.md)** for the full guide.
In short:

```bash
cd experiment_b
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python download_sam.py                      # SAM 2 (ungated) + SAM 3 (gated: hf auth login)
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GEMINI_API_KEY=... OPENROUTER_API_KEY=...
python experiment_b_run_v2.py --mode sam2_area --K 3 --device cuda \
  --models gpt_5_5,claude_sonnet_5,gemini_3_5_flash,llama_4_maverick
```

Needs **both** a GPU (SAM) and the 4 API keys (the VLMs do the affordance labelling).

---

## Citation

```bibtex
@inproceedings{affbench2026,
  title     = {Can Frontier Vision-Language Models Reason About the Interactable World?
               A Structured Affordance Benchmark},
  author    = {TBD},
  booktitle = {ECCV 2026 Workshop on Visual Perception and Reasoning in the
               Interactable World (X-Reason)},
  year      = {2026}
}
```
