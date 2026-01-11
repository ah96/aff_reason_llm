from typing import List, Dict, Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Optional CIDEr support
try:
    from pycocoevalcap.cider.cider import Cider
    _HAS_CIDER = True
except Exception:
    _HAS_CIDER = False


def bleu4(candidate: str, references: List[str]) -> float:
    refs_tok = [r.split() for r in references]
    cand_tok = candidate.split()
    smooth = SmoothingFunction().method1
    return float(sentence_bleu(refs_tok, cand_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))


def meteor(candidate: str, references: List[str]) -> float:
    # NLTK meteor_score expects list of references (strings) + candidate string
    return float(meteor_score(references, candidate))


def rouge_l(candidate: str, references: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for r in references:
        s = scorer.score(r, candidate)["rougeL"].fmeasure
        scores.append(float(s))
    return float(sum(scores) / max(1, len(scores)))


def cider(candidate: str, references: List[str]) -> Optional[float]:
    if not _HAS_CIDER:
        return None
    # pycocoevalcap expects dict {id: [refs]} and {id: [hyp]}
    gts = {0: references}
    res = {0: [candidate]}
    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return float(score)


def compute_caption_metrics(candidate: str, references: List[str]) -> Dict[str, float]:
    out = {
        "BLEU-4": bleu4(candidate, references),
        "METEOR": meteor(candidate, references),
        "ROUGE-L": rouge_l(candidate, references),
    }
    c = cider(candidate, references)
    out["CIDEr"] = float(c) if c is not None else float("nan")
    return out
