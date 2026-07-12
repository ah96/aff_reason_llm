"""
Experiment B — inter-model agreement / consensus from the per-model prediction jsonls.

Reads outdir/<model>_<mode>_K<K>.jsonl (written by experiment_b_run_v2.py), aligns predictions
by (image, region, action) across the standard models, and reports the GT-free reliability
signals: N-way agreement, mean pairwise agreement, per-model consensus accuracy, exception rate.
Compare two runs (sam2_area vs sam3_concept) to get the selection ablation.

    python3 experiment_b_agreement.py --mode sam2_area --K 10 \
        --models gpt_5_5,claude_sonnet_5,gemini_3_5_flash,llama_4_maverick --out agree_sam2.json
"""
import os
import json
import glob
import argparse
import itertools
from collections import defaultdict, Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="../experiment_b_bundle/out")
    ap.add_argument("--mode", default="sam2_area")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--models", default=None, help="Comma list; default = all found for this mode/K.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    want = set(args.models.split(",")) if args.models else None
    suffix = f"_{args.mode}_K{args.K}.jsonl"
    data = defaultdict(dict)                       # (image, region, action) -> {model: rel}
    found = []
    for f in glob.glob(os.path.join(args.outdir, f"*{suffix}")):
        name = os.path.basename(f)[:-len(suffix)]
        if want and name not in want:
            continue
        found.append(name)
        for line in open(f):
            r = json.loads(line)
            data[(r["image"], r["region_id"], r["action"])][name] = r["relationship_id"]
    M = sorted(set(found))
    if not M:
        print(f"no prediction files for mode={args.mode} K={args.K} in {args.outdir}"); return

    # keep only (image, region, action) items predicted by ALL models
    items = [v for v in data.values() if all(m in v for m in M)]
    n = len(items)
    nway = sum(1 for v in items if len({v[m] for m in M}) == 1) / max(1, n)

    pair_tot = pair_agree = 0
    for v in items:
        for a, b in itertools.combinations(M, 2):
            pair_tot += 1
            pair_agree += (v[a] == v[b])
    pairwise = pair_agree / max(1, pair_tot)

    # Consensus vs. majority is only well-defined when a STRICT majority exists. A tie for the
    # top label (e.g. a 2-2 split among 4 models) has NO majority; resolving it via
    # Counter.most_common would break the tie by insertion/model order, unfairly crediting whichever
    # models happen to hold the alphabetically-first value. We therefore exclude no-majority items
    # from consensus accuracy and the exception rate, and report how many there were. N-way and
    # pairwise agreement above are tie-free and need no such handling.
    consensus_hits = {m: 0 for m in M}
    exc = 0
    n_maj = 0
    for v in items:
        top = Counter(v[m] for m in M).most_common()
        if len(top) > 1 and top[1][1] == top[0][1]:
            continue                              # no strict majority (tie) -> undefined, skip
        n_maj += 1
        maj = top[0][0]
        if 2 <= maj <= 6:
            exc += 1
        for m in M:
            consensus_hits[m] += (v[m] == maj)

    summary = {
        "mode": args.mode, "K": args.K, "models": M,
        "n_items": n, "n_majority": n_maj,
        "agreement_Nway": round(nway, 4),
        "agreement_pairwise": round(pairwise, 4),
        "exception_rate": round(exc / max(1, n_maj), 4),
        "consensus_acc": {m: round(consensus_hits[m] / max(1, n_maj), 4) for m in M},
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
