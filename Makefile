# AffBench — convenience targets (no models needed for smoke/paper targets)

IMAGES_DIR   ?= ./images
OUTDIR       ?= ./out_b
LLMS_B       := ./experiment_b/configs/llms.json
SAM_CKPT     ?= ./sam_vit_h_4b8939.pth
SEEN_CKPT    ?= ./ooal_models_amar/seen_best
UNSEEN_CKPT  ?= ./ooal_models_amar/unseen_best
ADAPTER      := ./experiment_b/ooal_saliency_adapter.py
DEVICE       ?= cpu
KS           ?= 5 10 20
ACTIONS      ?= sit_on hold carry cut throw ride

# ── Smoke test (no models required) ─────────────────────────────────────────
.PHONY: smoke
smoke:
	python3 smoke_test.py

# ── Experiment B: SAM only ──────────────────────────────────────────────────
.PHONY: run_b_sam
run_b_sam:
	python3 experiment_b/experiment_b_run.py \
	  --images_dir $(IMAGES_DIR) \
	  --outdir $(OUTDIR) \
	  --llms $(LLMS_B) \
	  --mode sam \
	  --sam_ckpt $(SAM_CKPT) \
	  --device $(DEVICE) \
	  --Ks $(KS) \
	  --actions $(ACTIONS)

# ── Experiment B: SAM + Saliency ────────────────────────────────────────────
.PHONY: run_b_saliency
run_b_saliency:
	python3 experiment_b/experiment_b_run.py \
	  --images_dir $(IMAGES_DIR) \
	  --outdir $(OUTDIR) \
	  --llms $(LLMS_B) \
	  --mode sam_saliency \
	  --adapter $(ADAPTER) \
	  --seen_ckpt $(SEEN_CKPT) \
	  --unseen_ckpt $(UNSEEN_CKPT) \
	  --sam_ckpt $(SAM_CKPT) \
	  --device $(DEVICE) \
	  --Ks $(KS) \
	  --actions $(ACTIONS)

# ── Evaluate and generate paper tables ──────────────────────────────────────
.PHONY: tables
tables:
	python3 experiment_b/experiment_b_eval_paper_tables.py \
	  --outdir $(OUTDIR) \
	  --write_latex 1

# ── Compile the paper ────────────────────────────────────────────────────────
.PHONY: paper
paper:
	$(MAKE) -C paper

# ── Clean build artefacts ────────────────────────────────────────────────────
.PHONY: clean
clean:
	$(MAKE) -C paper clean

.PHONY: help
help:
	@echo "Targets:"
	@echo "  smoke           Run smoke tests (no models needed)"
	@echo "  run_b_sam       Experiment B: SAM-only selection"
	@echo "  run_b_saliency  Experiment B: SAM+OOAL saliency selection"
	@echo "  tables          Generate paper CSV + LaTeX tables from out_b/"
	@echo "  paper           Compile LaTeX paper in paper/"
	@echo ""
	@echo "Overrideable vars: IMAGES_DIR OUTDIR SAM_CKPT SEEN_CKPT UNSEEN_CKPT DEVICE KS ACTIONS"
