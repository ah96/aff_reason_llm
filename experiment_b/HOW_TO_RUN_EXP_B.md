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
- `configs/llms.json` — the 5-model lineup (GPT-5.5 on Flex tier / Sonnet 5 / Gemini free / Llama / o4-mini).
- `experiment_b_run_v2.py` — runner (`sam2_area`, `sam3_concept`, `mock`); per-call progress bar; resumable.
- `experiment_b_agreement.py` — inter-model agreement / consensus scorer (pure stdlib).
- `download_sam.py`, `vision_llm_clients.py`, `requirements.txt`.

## One-time setup
```bash
cd experiment_b

# 1) Python deps. Install torch/torchvision matching your CUDA FIRST:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2) SAM weights. SAM 2 auto-downloads by name via ultralytics; SAM 3 is GATED on HuggingFace.
python download_sam.py                 # SAM 2 (ungated) + SAM 3 (facebook/sam3)
#   For SAM 3: request access at https://huggingface.co/facebook/sam3 , then `hf auth login`
#   (or `export HF_TOKEN=hf_xxx`). The runner resolves the bare `sam3.pt` to the HF cache path.

# 2b) SAM 3 ONLY: its text encoder needs ultralytics' CLIP fork (a *callable* tokenizer). That fork's
#     pip build is broken (installs an empty 'UNKNOWN' wheel), so install the module files directly:
pip install --user git+https://github.com/openai/CLIP.git          # creates the clip/ package dir
git clone --depth 1 https://github.com/ultralytics/CLIP.git /tmp/ultra_clip
cp -r /tmp/ultra_clip/clip/* "$(python3 -c 'import clip,os;print(os.path.dirname(clip.__file__))')"/
python3 -c "from clip.simple_tokenizer import SimpleTokenizer as T; assert callable(T()); print('CLIP ok')"

# 3) API keys (same as Exp A). Gemini uses its FREE tier (rate-limited, ~$0).
export OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...  GEMINI_API_KEY=...  OPENROUTER_API_KEY=...

# 4) Shortcut for the 4-model lineup used below:
STD=gpt_5_5,claude_sonnet_5,gemini_3_5_flash,llama_4_maverick
```

## Step 1 — Smoke tests (do these first, ~cents each)
```bash
# a) No GPU / no keys — mocks the whole flow (crop -> query -> cache -> output):
python experiment_b_run_v2.py --mode mock --dry_run --limit_images 2

# b) SAM 2 real, 2 images — verifies segment-everything + all 4 VLMs (already passed):
python experiment_b_run_v2.py --mode sam2_area   --K 3 --limit_images 2 --models $STD --device cuda

# c) SAM 3 real, 2 images — verifies the GATED SAM 3 concept path (different wrapper + weights):
python experiment_b_run_v2.py --mode sam3_concept --K 3 --limit_images 2 --models $STD --device cuda
```
For (b)/(c): check the run finishes with `Wrote N rows ... 0 errors` and no traceback. If SAM 3 errors
on load, it's the gated checkpoint — re-do `download_sam.py --only sam3` after `hf auth login`.

## Step 2 — Full run (4 models, K=3, all 200 images)
Run **area-mode first** (both tables need it), then concept. `--workers 8` speeds the paid models.
Interruptible: Ctrl+C any time, re-run the SAME command to resume (cached calls are skipped, $0).

> **GPU:** run the two modes **sequentially, not at once** — SAM 2 and SAM 3 together exceed a 16 GB
> card (CUDA OOM silently drops regions). Let the area run finish, then start concept.
```bash
# area-ranked baseline (SAM 2)   ~$46 with GPT-5.5 Flex
python experiment_b_run_v2.py --mode sam2_area    --K 3 --models $STD --device cuda --workers 8

# concept-targeted (SAM 3)       ~$27
python experiment_b_run_v2.py --mode sam3_concept --K 3 --models $STD --device cuda --workers 8
```
The per-call bar shows `cached / live / err / exc` and `img i/200` as it goes.

## Step 3 — Score both modes
```bash
python experiment_b_agreement.py --mode sam2_area    --K 3 --models $STD --out agree_area_K3.json
python experiment_b_agreement.py --mode sam3_concept --K 3 --models $STD --out agree_concept_K3.json
```
Each prints `agreement_Nway`, `agreement_pairwise`, per-model `consensus_acc`, and `exception_rate`
(a 2-2 tie has no majority and is excluded — reported as `n_majority`). Transcribe:
- `agree_area_K3.json`  → `tab:main` (per-model consensus + 4-way agreement) and the SAM 2 row of `tab:selection`.
- `agree_concept_K3.json` → the SAM 3 row of `tab:selection`.

## Step 4 — Commit & push after the run
The **code** was already pushed; after the run you only version the small **result summaries**. The raw
predictions and cache are git-ignored (archive them separately, like Exp A).
```bash
git add experiment_b/agree_area_K3.json experiment_b/agree_concept_K3.json
git commit -m "Exp B agreement results (area + concept, K=3)"
git push
```
**Git-ignored — do NOT commit (archive separately):** `../experiment_b_bundle/out/` (raw per-model
`*.jsonl`) and `cache_b/`. Nothing else needs committing unless you changed code.

## How long does it take?
Full run ≈ **14.4k area + ~8–14k concept ≈ 20–28k VLM calls**, plus fast SAM (~10–40 min total).
- **Compute-bound (paid models, `--workers 8`):** a few hours — roughly **3–6 h** for both modes.
- **Real bottleneck = Gemini's free tier**, which is rate-limited (as in Exp A, which capped at 97%).
  Expect Gemini to gate wall-clock and possibly spread over **1–3 days**. The run is resumable, so just
  re-run the same command (e.g. daily) — cached calls are instant and free; only the un-done Gemini
  calls proceed. To avoid this, either enable billing on Gemini (~+$22, removes the cap → finishes in an
  afternoon) or let the 3 paid models complete first and let Gemini catch up over subsequent days.

## Notes
- `--workers` = VLM concurrency (default 4; use 8–12 for the full run; lower if rate-limited).
- Predictions stream to `../experiment_b_bundle/out/<model>_<mode>_K<K>.jsonl` (git-ignored, resumable).
- **No GPU?** `--mode mock --dry_run` exercises everything except SAM.
- Grow later for free: `--limit_images 50` now, `--limit_images 200` later — output append-and-dedups,
  cached calls are skipped. o4-mini is NOT used here (it's the Exp A reasoning model).
