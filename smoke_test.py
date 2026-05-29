#!/usr/bin/env python3
"""
smoke_test.py — lightweight sanity check requiring no models, datasets, or API keys.

Validates:
  1. All Python modules import without error
  2. Both llms.json configs load and contain at least one entry per expected provider
  3. All default action names exist in OOAL SEEN_AFF (saliency compatibility)
  4. The 7-way taxonomy is self-consistent across all modules that define it
  5. ade_parsing.load_relationship_file handles variable action counts correctly

Run with:  python3 smoke_test.py
"""

import sys
import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

errors = []

def check(label: str, fn):
    try:
        fn()
        print(f"  {PASS}  {label}")
    except Exception as e:
        print(f"  {FAIL}  {label}")
        print(f"         {e}")
        errors.append(label)


# ── 1. Module imports ────────────────────────────────────────────────────────

print("\n[1] Module imports")

def _import(path):
    path = Path(path)
    name = path.stem  # unique per file
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register so intra-package imports resolve
    spec.loader.exec_module(mod)
    return mod

check("experiment_a.ade_parsing",
      lambda: _import(ROOT / "experiment_a/ade_parsing.py"))
check("experiment_a.llm_clients",
      lambda: _import(ROOT / "experiment_a/llm_clients.py"))
check("experiment_a.metrics_relationship",
      lambda: _import(ROOT / "experiment_a/metrics_relationship.py"))
check("experiment_b.vision_llm_clients",
      lambda: _import(ROOT / "experiment_b/vision_llm_clients.py"))
check("experiment_b.experiment_b_metrics",
      lambda: _import(ROOT / "experiment_b/experiment_b_metrics.py"))
check("experiment_b.experiment_b_metrics_plus",
      lambda: _import(ROOT / "experiment_b/experiment_b_metrics_plus.py"))


# ── 2. LLM config loading ────────────────────────────────────────────────────

print("\n[2] LLM configs")

def _check_llm_config(cfg_path: Path, required_providers: list):
    data = json.loads(cfg_path.read_text())
    assert isinstance(data, list) and len(data) > 0, "Config must be a non-empty list"
    providers = {e["provider"].lower() for e in data}
    for p in required_providers:
        assert any(p in prov for prov in providers), f"Missing provider: {p}"

check("experiment_b/configs/llms.json loads",
      lambda: _check_llm_config(
          ROOT / "experiment_b/configs/llms.json",
          ["anthropic", "gemini", "openai"]))
check("experiment_a/configs/llms.json loads",
      lambda: _check_llm_config(
          ROOT / "experiment_a/configs/llms.json",
          ["anthropic", "gemini"]))

def _check_claude_version(cfg_path: Path):
    data = json.loads(cfg_path.read_text())
    for e in data:
        if "anthropic" in e.get("provider", "").lower():
            model = e.get("model", "")
            assert "4-5" not in model, \
                f"Stale Claude 4.5 model ID in {cfg_path.name}: {model!r}"

check("experiment_b: Claude model is 4.6+",
      lambda: _check_claude_version(ROOT / "experiment_b/configs/llms.json"))
check("experiment_a: Claude model is 4.6+",
      lambda: _check_claude_version(ROOT / "experiment_a/configs/llms.json"))


# ── 3. Action / OOAL SEEN_AFF compatibility ──────────────────────────────────

print("\n[3] Action vocabulary")

SEEN_AFF = [
    'beat', 'boxing', 'brush_with', 'carry', 'catch', 'cut', 'cut_with',
    'drag', 'drink_with', 'eat', 'hit', 'hold', 'jump', 'kick', 'lie_on',
    'lift', 'look_out', 'open', 'pack', 'peel', 'pick_up', 'pour', 'push',
    'ride', 'sip', 'sit_on', 'stick', 'stir', 'swing', 'take_photo',
    'talk_on', 'text_on', 'throw', 'type_on', 'wash', 'write',
]
UNSEEN_AFF = [
    'carry', 'catch', 'cut', 'cut_with', 'drink_with', 'eat', 'hit', 'hold',
    'jump', 'kick', 'lie_on', 'open', 'peel', 'pick_up', 'pour', 'push',
    'ride', 'sip', 'sit_on', 'stick', 'swing', 'take_photo', 'throw',
    'type_on', 'wash',
]
DEFAULT_ACTIONS = ["sit_on", "hold", "carry", "cut", "throw", "ride"]

def _check_actions_in_seen():
    for a in DEFAULT_ACTIONS:
        assert a in SEEN_AFF, f"Action {a!r} not in OOAL SEEN_AFF"

def _check_actions_in_unseen():
    for a in DEFAULT_ACTIONS:
        assert a in UNSEEN_AFF, f"Action {a!r} not in OOAL UNSEEN_AFF (saliency fallback won't work)"

check("All default actions are in OOAL SEEN_AFF", _check_actions_in_seen)
check("All default actions are in OOAL UNSEEN_AFF", _check_actions_in_unseen)


# ── 4. Taxonomy consistency ──────────────────────────────────────────────────

print("\n[4] Taxonomy consistency")

REL_CATEGORIES_B = [
    "Positive", "Firmly Negative", "Object Non-functional",
    "Physical Obstacle", "Socially Awkward", "Socially Forbidden",
    "Dangerous to ourselves/others",
]
REL_CATEGORIES_A = {
    0: "Positive", 1: "FirmlyNegative", 2: "ObjectNonFunctional",
    3: "PhysicalObstacle", 4: "SociallyAwkward", 5: "SociallyForbidden",
    6: "Dangerous",
}

check("7 categories defined in experiment_b",
      lambda: None if len(REL_CATEGORIES_B) == 7
      else (_ for _ in ()).throw(AssertionError(f"Expected 7, got {len(REL_CATEGORIES_B)}")))
check("7 categories defined in experiment_a",
      lambda: None if len(REL_CATEGORIES_A) == 7
      else (_ for _ in ()).throw(AssertionError(f"Expected 7, got {len(REL_CATEGORIES_A)}")))
check("experiment_a metrics_relationship has 7 labels",
      lambda: None if _import(ROOT / "experiment_a/metrics_relationship.py").REL7_NAMES
              and len(_import(ROOT / "experiment_a/metrics_relationship.py").REL7_NAMES) == 7
      else (_ for _ in ()).throw(AssertionError("REL7_NAMES wrong size")))


# ── 5. ade_parsing.load_relationship_file ────────────────────────────────────

print("\n[5] ade_parsing.load_relationship_file")

import tempfile, os

def _check_rel_file_variable_actions():
    mod = _import(ROOT / "experiment_a/ade_parsing.py")
    content = "10 # 0 # 1 # 2 # 3 # 4 # 5\n20 # 1 # 0 # 0 # 1 # 2 # 0\n"
    actions6 = ["sit_on", "hold", "carry", "cut", "throw", "ride"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        tmp = f.name
    try:
        result = mod.load_relationship_file(tmp, actions=actions6)
        assert 10 in result and 20 in result, "Missing instance IDs"
        assert result[10]["sit_on"] == 0
        assert result[10]["ride"] == 5
        assert result[20]["hold"] == 0
    finally:
        os.unlink(tmp)

def _check_rel_file_default_actions():
    mod = _import(ROOT / "experiment_a/ade_parsing.py")
    content = "209 # 2 # 6 # 0\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        tmp = f.name
    try:
        result = mod.load_relationship_file(tmp)  # default: sit, run, grasp
        assert result[209]["sit"] == 2
        assert result[209]["run"] == 6
        assert result[209]["grasp"] == 0
    finally:
        os.unlink(tmp)

check("load_relationship_file with 6 new actions", _check_rel_file_variable_actions)
check("load_relationship_file with default 3 actions (backward compat)", _check_rel_file_default_actions)


# ── 6. experiment_b_metrics majority_vote determinism ────────────────────────

print("\n[6] Metrics correctness")

def _check_majority_vote_determinism():
    mod = _import(ROOT / "experiment_b/experiment_b_metrics.py")
    # Tie between 0 and 1: should return smaller label (0)
    result = mod.majority_vote([0, 0, 1, 1])
    assert result == 0, f"Tie-break failed: expected 0, got {result}"
    assert mod.majority_vote([1, 1, 1]) == 1
    assert mod.majority_vote([2, 0, 2, 0, 2]) == 2

def _check_agreement_rate():
    mod = _import(ROOT / "experiment_b/experiment_b_metrics.py")
    labels = {"gpt": [0, 0, 1], "claude": [0, 1, 1], "gemini": [0, 0, 1]}
    rate = mod.agreement_rate(labels)
    assert 0.0 < rate <= 1.0, f"Unexpected agreement rate: {rate}"

check("majority_vote is deterministic under ties", _check_majority_vote_determinism)
check("agreement_rate returns value in (0,1]", _check_agreement_rate)


# ── Summary ──────────────────────────────────────────────────────────────────

print()
if errors:
    print(f"FAILED ({len(errors)} checks): {', '.join(errors)}")
    sys.exit(1)
else:
    print("All checks passed.")
    sys.exit(0)
