# Experiment B — run guide (lab PC, GPU)

GT-free **typed-affordance agreement** across four frontier VLMs over a SAM segmentation pipeline.
Two selection strategies are compared: **SAM 2 area-ranked** (baseline — segment everything, keep
the top-K by area) vs **SAM 3 concept-targeted** (text-prompt each action's object concepts). SAM
runs **once per image**, so all four VLMs judge the *same* regions → their agreement is the signal.
Results fill paper tables `tab:main` (per-model consensus + 4-way agreement) and `tab:selection`
(area vs concept).

> **You need BOTH:** an NVIDIA **GPU** (for SAM) **and** the **4 cloud API keys** (the VLMs do the
> actual affordance labelling — SAM only proposes regions).

## What's already in the repo
- `../experiment_b_bundle/images/` — **200 ADE20K validation scenes** (committed, distinct from Exp A).
- `configs/action_concepts.json` — per-action object concepts (the SAM 3 prompt vocabulary).
- `configs/llms.json` — the 5-model lineup (GPT-5.5 / Sonnet 5 / Gemini 3.5 / Llama 4 / o4-mini).
- `experiment_b_run_v2.py` — runner (`sam2_area`, `sam3_concept`, `mock`).
- `experiment_b_agreement.py` — inter-model agreement / consensus scorer (pure stdlib).
- `download_sam.py`, `vision_llm_clients.py`, `requirements.txt`.

## One-time setup
```bash
cd experiment_b

# 1) Python deps. Install torch/torchvision matching your CUDA FIRST:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2) SAM weights (auto-download by name via ultralytics).
python download_sam.py                 # SAM 2 (ungated) + SAM 3
#   SAM 3 (sam3.pt) is GATED on HuggingFace: request access at
#   https://huggingface.co/facebook/sam3 , then:
hf auth login                          # paste your HF token, then: python download_sam.py --only sam3

# 3) API keys (same as Exp A). Gemini uses its FREE tier (rate-limited, ~$0).
export OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...  GEMINI_API_KEY=...  OPENROUTER_API_KEY=...
```

## Smoke tests (do these first)
```bash
# a) No GPU / no keys — exercises crop -> query -> cache -> output with mocks:
python experiment_b_run_v2.py --mode mock --dry_run --limit_images 2

# b) Real SAM + real VLMs on 2 images — verifies masks come back and keys work (a few cents):
STD=gpt_5_5,claude_sonnet_5,gemini_3_5_flash,llama_4_maverick
python experiment_b_run_v2.py --mode sam2_area --K 3 --limit_images 2 --models $STD --device cuda
```
The SAM provider wrappers (`_Sam2`, `_Sam3`) are verified on this first real run — check that masks
are returned before launching the full 200.

## Full run (default: 4 models, K=3, all 200 images)
Run **area-mode first** (needed for both tables) — check the bill — then concept-mode.
```bash
STD=gpt_5_5,claude_sonnet_5,gemini_3_5_flash,llama_4_maverick

# 1) area-ranked baseline (SAM 2)  ~$46 with GPT-5.5 Flex
python experiment_b_run_v2.py --mode sam2_area   --K 3 --models $STD --device cuda

# 2) concept-targeted (SAM 3)      ~$27
python experiment_b_run_v2.py --mode sam3_concept --K 3 --models $STD --device cuda

# 3) score agreement for each mode -> fills tab:main and tab:selection
python experiment_b_agreement.py --mode sam2_area    --K 3 --models $STD --out agree_area_K3.json
python experiment_b_agreement.py --mode sam3_concept --K 3 --models $STD --out agree_concept_K3.json
```

## Cost (4 models, Gemini free, GPT-5.5 on Flex tier ≈ ~$0.013 / region-action pair)
GPT-5.5 is ~half the bill; it runs on OpenAI's **Flex tier** (`service_tier:"flex"` in
`configs/llms.json`) → ~50% off, best-effort latency (the client retries the occasional Flex 429).
Prices checked Jul 2026; treat as ±30%.

| images | K=3 (both modes) | K=5 (both modes) |
|---|---|---|
| 50  | ~$18  | ~$30  |
| 100 | ~$37  | ~$61  |
| **200** | **~$73** (area ~$46 + concept ~$27) | ~$122 |

Dial with `--limit_images` and `--K`. Gemini's free tier is load-bearing — if it rate-limits you
onto paid, add ~$22 for the 200-image run. Remove the `service_tier` line from `configs/llms.json`
for standard-tier speed (2× the GPT-5.5 cost).

## Incremental / modular runs (top up later if budget allows)
Output is **append-and-dedup** keyed by `(image, region_id, action)`, and every VLM call is cached
under `cache_b/`. So you can start small and grow the run without repeat spend or data loss:
```bash
python experiment_b_run_v2.py --mode sam2_area --K 3 --limit_images 50  --models $STD   # cheap first pass
python experiment_b_run_v2.py --mode sam2_area --K 3 --limit_images 200 --models $STD   # resumes: first 50 free, adds 150
```
`--limit_images N` takes the first N images (nested: 50 ⊂ 200). Growing K reuses area-mode's cached
regions. Re-score with `experiment_b_agreement.py` anytime — it reads whatever has accumulated.

## Notes
- `--workers` controls VLM concurrency (default 4; lower if rate-limited).
- Predictions stream to `../experiment_b_bundle/out/<model>_<mode>_K<K>.jsonl` (git-ignored).
- **No GPU?** `--mode mock --dry_run` exercises everything except SAM.
- o4-mini is not needed here (it's the Exp A reasoning model); the four `$STD` models drive agreement.
