# AffBench: Structured Affordance Reasoning with Frontier VLMs

**ECCV 2026 Workshop X-Reason** — *Visual Perception and Reasoning in the Interactable World*

This repository evaluates whether frontier Vision-Language Models (VLMs) can reason about **affordances** — what actions objects support, and why certain actions are forbidden (physically, socially, or for safety reasons). We combine [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) for automatic instance discovery with a structured 7-way affordance taxonomy queried across GPT-4.1, Claude Sonnet 4.6, and Gemini 2.5 Pro.

---

## Affordance Taxonomy

| Code | Category | Description |
|------|----------|-------------|
| 0 | **Positive** | Action is appropriate and feasible |
| 1 | **Firmly Negative** | Action is clearly inappropriate |
| 2 | **Object Non-functional** | Object condition prevents the action |
| 3 | **Physical Obstacle** | Scene geometry prevents the action |
| 4 | **Socially Awkward** | Action is contextually inappropriate |
| 5 | **Socially Forbidden** | Action violates social/legal norms |
| 6 | **Dangerous** | Action poses a physical safety risk |

Exception categories (2–6) require the model to supply a one-sentence explanation and consequence grounded in the image.

---

## Repository Structure

```
aff_reason_llm/
├── experiment_a/               # Text-only evaluation on ADE-Affordance GT
│   ├── eval_experiment_a.py
│   ├── llm_clients.py
│   ├── metrics_relationship.py
│   ├── metrics_caption.py
│   └── configs/llms.json
│
├── experiment_b/               # Vision pipeline: SAM → VLM
│   ├── experiment_b_run.py         # Main runner
│   ├── experiment_b_run_dataset.py # Dataset-scale runner
│   ├── experiment_b_eval_metrics.py
│   ├── experiment_b_eval_paper_tables.py
│   ├── vision_llm_clients.py
│   ├── ooal_saliency_adapter.py    # OOAL affordance saliency wrapper
│   ├── ooal/                       # OOAL model (DINOv2 + CLIP + SegDecoder)
│   └── configs/llms.json
│
├── images/                     # Test images
├── out_b/                      # Experiment B outputs
├── cache_b/                    # LLM response cache (hash-based)
├── paper/                      # LaTeX paper (X-Reason workshop)
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
```

Download the SAM ViT-H checkpoint:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---

## Running Experiments

### Experiment B: Vision Pipeline

**SAM only (area-ranked selection):**
```bash
python3 experiment_b/experiment_b_run.py \
  --images_dir ./images/ \
  --outdir ./out_b \
  --llms ./experiment_b/configs/llms.json \
  --mode sam \
  --sam_ckpt ./sam_vit_h_4b8939.pth \
  --device cpu \
  --Ks 5 10 20 \
  --actions sit_on hold carry cut throw ride
```

**SAM + Saliency (OOAL-guided selection):**
```bash
python3 experiment_b/experiment_b_run.py \
  --images_dir ./images/ \
  --outdir ./out_b \
  --llms ./experiment_b/configs/llms.json \
  --mode sam_saliency \
  --adapter ./experiment_b/ooal_saliency_adapter.py \
  --seen_ckpt ./ooal_models_amar/seen_best \
  --unseen_ckpt ./ooal_models_amar/unseen_best \
  --sam_ckpt ./sam_vit_h_4b8939.pth \
  --device cpu \
  --Ks 5 10 20 \
  --actions sit_on hold carry cut throw ride
```

**Evaluate and generate paper tables:**
```bash
python3 experiment_b/experiment_b_eval_paper_tables.py --outdir ./out_b --write_latex 1
```

### Experiment A: Text-Only (ADE-Affordance)

```bash
python3 experiment_a/eval_experiment_a.py \
  --data_dir /path/to/ade_affordance \
  --llms ./experiment_a/configs/llms.json
```

---

## Action Vocabulary

Actions must match OOAL `SEEN_AFF` names when using `sam_saliency` mode.
The default set covers diverse interaction types:

| Action | Type |
|--------|------|
| `sit_on` | Whole-body posture |
| `hold` | Fine manipulation |
| `carry` | Transport |
| `cut` | Tool use |
| `throw` | Ballistic / sport |
| `ride` | Vehicle / animal |

---

## Models Evaluated

| Name | Provider | Model ID |
|------|----------|----------|
| GPT-4.1-mini | OpenAI | `gpt-4.1-mini` |
| Claude Sonnet 4.6 | Anthropic | `claude-sonnet-4-6` |
| Gemini 2.5 Pro | Google | `gemini-2.5-pro` |

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
