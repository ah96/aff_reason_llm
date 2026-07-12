"""
Pre-download the SAM weights Experiment B needs, so the first run doesn't stall.

Both SAM providers run through **ultralytics**, which auto-downloads weights by name on first
use — this script just triggers those downloads now and gives a clear message for the gated
SAM 3 checkpoint. You normally run it once after `pip install -r requirements.txt`.

    python download_sam.py                 # both SAM 2 and SAM 3
    python download_sam.py --only sam2     # just the area-ranked baseline weight
    python download_sam.py --only sam3     # just the concept-targeted weight

SAM 2 (`sam2.1_l.pt`) is ungated. SAM 3 (`sam3.pt`) is **gated** on HuggingFace
(facebook/sam3): request access there, then `hf auth login` (paste your HF token), then re-run.
Downloaded *.pt files are git-ignored.
"""
import sys
import argparse

SAM2_NAME = "sam2.1_l.pt"      # SAM 2.1 large — segment-everything baseline (ungated)
SAM3_NAME = "sam3.pt"          # SAM 3 — concept-targeted (gated: facebook/sam3)


def fetch_via_ultralytics(name: str) -> None:
    """Instantiating ultralytics SAM(name) downloads + caches the weight file."""
    from ultralytics import SAM
    print(f"[download_sam] fetching {name} via ultralytics ...", flush=True)
    SAM(name)
    print(f"[download_sam] OK: {name}", flush=True)


def fetch_sam3() -> None:
    """SAM 3 first via ultralytics; on failure fall back to the gated HF repo."""
    try:
        fetch_via_ultralytics(SAM3_NAME)
        return
    except Exception as e1:
        print(f"[download_sam] ultralytics fetch of {SAM3_NAME} failed ({e1}); trying HuggingFace ...",
              file=sys.stderr, flush=True)
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id="facebook/sam3", filename=SAM3_NAME)
        print(f"[download_sam] OK via HuggingFace -> {p}", flush=True)
    except Exception as e2:
        print("\n[download_sam] Could not download SAM 3.", file=sys.stderr)
        print("SAM 3 is GATED on HuggingFace. Steps:", file=sys.stderr)
        print("  1) request access: https://huggingface.co/facebook/sam3", file=sys.stderr)
        print("  2) hf auth login          (paste your HF token)", file=sys.stderr)
        print("  3) re-run: python download_sam.py --only sam3", file=sys.stderr)
        print(f"     (underlying error: {e2})", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-download SAM 2 / SAM 3 weights for Experiment B.")
    ap.add_argument("--only", choices=["sam2", "sam3"], default=None,
                    help="download just one of the two (default: both)")
    args = ap.parse_args()

    if args.only in (None, "sam2"):
        fetch_via_ultralytics(SAM2_NAME)
    if args.only in (None, "sam3"):
        fetch_sam3()
    print("[download_sam] done.", flush=True)


if __name__ == "__main__":
    main()
