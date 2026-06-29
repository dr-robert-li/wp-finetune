# Phase 8: Reward Infrastructure - Pattern Map

**Mapped:** 2026-06-19
**Files analyzed:** 7 new/modified files
**Analogs found:** 7 / 7

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `scripts/reward_pipeline.py` | signal-scoring module + normalization math | batch (group) + per-sample | `eval/rubric_scorer.py` + `eval/eval_gate.py` | role-match (per-sample signal scoring + gate logic) |
| `eval/eval_judge.py` (refactor only) | service — per-sample judge wrapper extraction | request-response | `eval/eval_judge.py` itself (internal `_judge_create` + `parse_judge_response`) | exact (surgical extraction within same file) |
| `scripts/build_antihack_set.py` | data-construction script | batch + agent-orchestration | `scripts/generate_judge_batch.py` + `wp-finetune:run-data-pipeline` SKILL.md | role-match |
| `tests/test_reward_pipeline.py` | unit test | — | `tests/test_protected_mask.py` + `tests/test_bootstrap_ci.py` | exact (same class-based pytest convention) |
| `tests/test_reward_pipeline_integration.py` | integration test | — | `tests/test_pipeline_integration.py` | exact (patch-based integration test pattern) |
| `tests/test_antihack.py` | CI-aware gate test | — | `tests/test_bootstrap_ci.py` (`TestJaccardDisposition`) | exact (CI gate assertion pattern) |
| `tests/conftest.py` | shared fixtures | — | `tests/phase4_4/conftest.py` | role-match (session-scoped path fixtures + synthetic response builders) |

---

## Pattern Assignments

### `scripts/reward_pipeline.py` (signal-scoring module, batch data flow)

**Primary analog:** `eval/rubric_scorer.py` (per-sample scoring + dataclass output contract)
**Secondary analog:** `eval/eval_gate.py` (threshold/gate logic pattern)

**Imports pattern** — copy from `eval/rubric_scorer.py` lines 16–34:
```python
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import openai

from eval.rubric_scorer import score_code, RubricScore, POSITIVE_CHECK_IDS, NEGATIVE_CHECK_IDS
from eval.rubric_definitions import CRITICAL_FLOOR_RULES, DIMENSION_WEIGHTS
from eval.eval_judge import _judge_create, parse_judge_response
from scripts.compute_concentration import bootstrap_ci
```

**Module-level constant loading** — pattern from `eval/eval_gate.py` lines 26–38 (fallback thresholds loaded once at module level):
```python
# Load recalibration offset at module init — never hardcode 3.58 here.
# Source of truth: output/eval_reasoning_v4_winner/judge_recalibration.json
_RECALIB_PATH = Path("output/eval_reasoning_v4_winner/judge_recalibration.json")

def _load_score_offset() -> float:
    data = json.loads(_RECALIB_PATH.read_text())
    return float(data["score_offset"])

_SCORE_OFFSET: float = _load_score_offset()   # module-level singleton
```

**Output contract dataclasses** — copy dataclass style from `eval/rubric_scorer.py` lines 68–97 (`RubricScore` with `to_dict()`):
```python
@dataclass
class RewardBreakdown:
    # Pre-normalization (raw signal values)
    phpcs_raw: float              # rubric_scorer overall (0-100)
    verpo_raw: float              # VeRPO weighted pass fraction (0-1)
    judge_raw: Optional[float]    # raw wp_judge overall_score or None on parse failure
    judge_offset_applied: float   # judge_raw + _SCORE_OFFSET, clipped [0, 100]
    security_fail: bool           # whether CRITICAL_FLOOR_RULE for D2_security triggered
    # Post-normalization
    phpcs_norm: float
    verpo_norm: float
    judge_norm: float
    # Composite pre-gate
    composite_pre_gate: float
    # VeRPO per-check data
    check_pass_rates: dict        # {check_id: pass_rate_across_group}
    check_difficulties: dict      # {check_id: difficulty_weight}
    # Group stats
    group_size: int
    group_phpcs_mean: float
    group_phpcs_std: float
    group_judge_mean: float
    group_judge_std: float
    # Parse failure metadata
    judge_parse_failure: bool = False
    judge_imputed_from_group: bool = False

    def to_dict(self) -> dict:
        # mirrors RubricScore.to_dict() pattern
        ...

@dataclass
class RewardResult:
    scalar: float           # final reward (0.0 if security gate, else composite)
    breakdown: RewardBreakdown
```

**MO-GRPO normalization helper** — no direct analog; implement using numpy, pattern from `scripts/compute_concentration.py` lines 59–72 (epsilon guard style):
```python
_EPSILON = 1e-8  # matches D-08-04 / compute_concentration rng pattern

def _mo_grpo_norm(values: np.ndarray) -> np.ndarray:
    """Within-group standardization: (x - mu) / (sigma + epsilon)."""
    mu = values.mean()
    sigma = values.std()
    return (values - mu) / (sigma + _EPSILON)
```

**Security hard gate** — terminal override AFTER composite; pattern referenced in `eval/eval_gate.py` lines 60–80 (threshold check then result):
```python
# CRITICAL: gate applied AFTER normalize+combine, not before.
# D-08-05: fire condition is any CRITICAL_FLOOR_RULE for D2_security triggering,
# NOT floor_rules_applied being non-empty.
# See apply_floor_rules() in rubric_scorer.py lines 605-628:
#   a rule triggers when any(check_hits.get(cid, False) for cid in trigger_checks)
#   BUT floor_rules_applied only appends when current > cap — so check_hits is
#   the reliable signal, not floor_rules_applied. Gate must inspect check_hits
#   directly against D2_security entries in CRITICAL_FLOOR_RULES.
def _security_fail(rubric: RubricScore) -> bool:
    for rule in CRITICAL_FLOOR_RULES:
        dim_key = rule[0] if isinstance(rule, (list, tuple)) else rule["dimension"]
        if dim_key != "D2_security":
            continue
        trigger_checks = rule[2] if isinstance(rule, (list, tuple)) else rule["triggers"]
        # check_hits reconstructed from triggered_checks (all fired check IDs)
        all_triggered = {cid for ids in rubric.triggered_checks.values() for cid in ids}
        if any(cid in all_triggered for cid in trigger_checks):
            return True
    return False

# In compute_group_rewards() final step:
final_scalar = 0.0 if breakdown.security_fail else breakdown.composite_pre_gate
```

**Public API signatures**:
```python
def compute_reward(
    php_code: str,
    group_signals: "GroupSignals",   # pre-computed group normalization params
    judge_client: openai.OpenAI,
    judge_model: str,
) -> RewardResult:
    """Score a single generation given pre-computed group normalization params."""

def compute_group_rewards(
    php_codes: list[str],
    judge_client: openai.OpenAI,
    judge_model: str,
) -> list[RewardResult]:
    """Two-pass: collect raw → group stats → normalize → gate → return list."""
```

**LLM checks guard** — add at module top (Pitfall 6):
```python
# Ensure deterministic reward compute — do NOT run LLM checks in training.
os.environ.pop("RUBRIC_USE_LLM_CHECKS", None)
```

**Judge parse failure fallback** — impute from group mean (D-08-07):
```python
judge_scores = [judge_score_single(code, client, model) for code in php_codes]
valid_scores = [s for s in judge_scores if s is not None]
group_judge_mean = float(np.mean(valid_scores)) if valid_scores else 0.0
parse_fail_rate = sum(1 for s in judge_scores if s is None) / len(judge_scores)
if parse_fail_rate > 0.10:
    import warnings
    warnings.warn(f"judge parse failure rate {parse_fail_rate:.1%} exceeds 10% threshold")
# impute None entries with group mean before applying offset/clip/normalize
judge_scores_imputed = [s if s is not None else group_judge_mean for s in judge_scores]
```

---

### `eval/eval_judge.py` — refactor: extract `judge_score_single()` (service, request-response)

**Analog:** `eval/eval_judge.py` itself — the existing `_judge_create` (lines 46–75) and `parse_judge_response` (lines 154–207) are the pattern to wrap.

**Critical constraint — must call `_judge_create`, NOT `client.chat.completions.create` directly** (RC-A guard, lines 46–75):
```python
# eval/eval_judge.py lines 46-75 — MUST reuse this wrapper:
def _judge_create(client, *, model, messages, max_tokens=1024, temperature=0.0):
    """Query vLLM with enable_thinking=False (RC-A fix).
    Graceful fallback if template rejects kwarg, but WARNS loudly."""
    global _thinking_kwarg_supported
    if _thinking_kwarg_supported:
        try:
            return client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
                temperature=temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as e:
            emsg = str(e).lower()
            if "enable_thinking" in emsg or "chat_template" in emsg or "template" in emsg:
                _thinking_kwarg_supported = False
                print("WARNING [eval_judge RC-A]: ...", file=sys.stderr, flush=True)
            else:
                raise
    return client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
```

**New function to add to `eval/eval_judge.py`** (after `parse_judge_response`, before `_extract_code_from_judge_prompt`):
```python
def judge_score_single(
    php_code: str,
    client: openai.OpenAI,
    model: str,
    max_tokens: int = 512,
) -> Optional[float]:
    """Invoke wp_judge on a single PHP code string.

    Returns raw overall_score (0-100) or None on parse failure.
    MUST use _judge_create (not client.chat.completions.create) for RC-A guard.
    """
    messages = [{"role": "user", "content": f"<wp_judge> Evaluate this WordPress code:\n\n{php_code}"}]
    resp = _judge_create(client, model=model, messages=messages,
                         max_tokens=max_tokens, temperature=0.0)
    raw_text = resp.choices[0].message.content
    parsed = parse_judge_response(raw_text)
    if parsed is None:
        return None
    overall = parsed.get("overall_score")
    return float(overall) if isinstance(overall, (int, float)) else None
```

**Client setup pattern** — copy from `eval/eval_judge.py` lines 363–368:
```python
import os
resolved_base_url = base_url or os.environ.get("EVAL_JUDGE_BASE_URL")
if not resolved_base_url:
    from scripts.dgx_toolbox import get_toolbox
    resolved_base_url = get_toolbox().vllm_endpoint()
client = openai.OpenAI(base_url=resolved_base_url, api_key="none")
```

---

### `scripts/build_antihack_set.py` (data-construction script, batch + agent-orchestration)

**Primary analog:** `scripts/generate_judge_batch.py` lines 1–80 (CLI structure, PROJECT_ROOT, checkpoint pattern, argparse)
**Secondary analog:** `wp-finetune:run-data-pipeline` SKILL.md lines 43–55 (Agent background invocation pattern)

**File header + imports pattern** — copy from `generate_judge_batch.py` lines 1–32:
```python
#!/usr/bin/env python3
"""Build adversarial anti-hack eval set for D-11 reward-pipeline regression gate.

Perturbs real gen+judge JSONL outputs along three axes:
  1. Verbose padding (inert comments/docblocks)
  2. Template-critique collapse (boilerplate critique phrases)
  3. Self-preference swap (judge evaluates its own training target)

Usage:
    python -m scripts.build_antihack_set \\
        --source-jsonl output/eval_reasoning_v4_winner/eval_judge_results.pairs.jsonl \\
        --output-dir output/antihack_validation/ \\
        --cases-per-axis 15
"""
import argparse
import json
import random
from pathlib import Path

from scripts.compute_concentration import bootstrap_ci
from scripts.reward_pipeline import compute_reward   # available after Wave 2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

**Agent spawn pattern for scoring** — from SKILL.md lines 43–55 (Agent with run_in_background=true):
```
# Claude Code orchestrator spawns scoring agents, NOT direct Anthropic API calls.
# Pattern (D-08-03 / SKILL.md):
Agent(
  model="sonnet",
  description="Score antihack batch: axis={axis} cases={batch_ids}",
  prompt="Score each PHP case in {batch_file} using reward_pipeline.compute_reward().
  Write results to {output_file} as JSONL: {case_id, scalar, breakdown_dict}.
  Use judge endpoint from EVAL_JUDGE_BASE_URL env or DGX toolbox.",
  run_in_background=True
)
# Wait for all agents, then collect results.
```

**Source filtering — only perturb MEDIUM-HIGH quality originals** (Pitfall 7):
```python
# Filter source records to rubric_score.overall >= 65.0 before perturbation.
def _load_source_records(path: Path, min_score: float = 65.0) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("rubric_overall", 0.0) >= min_score:
                records.append(rec)
    return records
```

**Bootstrap CI gate pattern** — from `scripts/compute_concentration.py` lines 42–72:
```python
from scripts.compute_concentration import bootstrap_ci
import numpy as np

lo_perturbed, hi_perturbed = bootstrap_ci(np.array(perturbed_rewards), n_boot=1000)
lo_clean, hi_clean = bootstrap_ci(np.array(clean_rewards), n_boot=1000)
gate_pass = hi_perturbed < lo_clean  # D-09 CI-aware disposition
# ALWAYS report all 4 bounds, not just pass/fail.
report = {
    "axis": axis_name,
    "gate_pass": gate_pass,
    "perturbed_ci": [lo_perturbed, hi_perturbed],
    "clean_ci": [lo_clean, hi_clean],
}
```

---

### `tests/test_reward_pipeline.py` (unit test)

**Analog:** `tests/test_protected_mask.py` (class-based pytest, GPU-free, synthetic data) and `tests/test_bootstrap_ci.py` (CI-gate assertion style)

**File header + import pattern** — copy from `tests/test_protected_mask.py` lines 1–18:
```python
"""Unit tests for scripts/reward_pipeline.py.

Tests are GPU-free: all external service calls mocked.
Covers GRPO-01..04, SC2 (security gate), RLEV-02 (breakdown contract).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.reward_pipeline import (
    RewardBreakdown, RewardResult,
    compute_group_rewards, compute_reward,
    _mo_grpo_norm, _security_fail, _load_score_offset,
)
```

**Class-based test grouping** — copy from `tests/test_protected_mask.py` lines 27–60 (one class per logical unit):
```python
class TestMOGRPONorm:
    def test_zero_variance_epsilon(self):
        """All-identical group -> sigma=0 -> epsilon floor prevents NaN."""
        values = np.ones(5) * 42.0
        result = _mo_grpo_norm(values)
        assert not np.any(np.isnan(result)), "NaN with zero-variance group (epsilon missing)"
        assert np.allclose(result, 0.0), "Zero-variance group must normalize to all-zeros"

    def test_mean_centered(self):
        """Normalized values must have zero mean."""
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = _mo_grpo_norm(values)
        assert abs(result.mean()) < 1e-6

class TestSecurityGate:
    def test_security_fail_overrides_to_zero(self):
        """Security-failing code -> final scalar == 0.0 regardless of other signals."""
        ...

    def test_gate_applied_after_normalization(self):
        """Security override must be TERMINAL — composite computed first, then zeroed."""
        ...
```

**Mock pattern for judge** — copy from `tests/test_eval_judge.py` function-level style:
```python
def test_judge_score_single_returns_float(monkeypatch):
    import eval.eval_judge as ej
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps({"overall_score": 82})
    monkeypatch.setattr(ej, "_judge_create", lambda *a, **kw: mock_resp)
    result = ej.judge_score_single("<?php echo 1;", MagicMock(), "test-model")
    assert result == 82.0
```

---

### `tests/test_reward_pipeline_integration.py` (integration test)

**Analog:** `tests/test_pipeline_integration.py` (patch-based integration test with `unittest.mock`)

**File structure pattern** — copy from `tests/test_pipeline_integration.py` lines 1–22:
```python
"""Integration tests for reward_pipeline.py — 50-case known-good/bad suite.

Tests exercise compute_group_rewards() end-to-end with fixture PHP files.
External services (vLLM judge endpoint, PHPCS) mocked where unavailable in CI.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

**Fixture loading pattern** — adapt from `test_pipeline_integration.py` checkpoint dict pattern:
```python
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "reward_integration_cases"
KNOWN_GOOD_DIR = FIXTURE_DIR / "known_good_php"
KNOWN_BAD_DIR = FIXTURE_DIR / "known_bad_php"
SC2_FILE = FIXTURE_DIR / "secure_fail_high_quality.php"

def test_sc2_security_fail_scores_zero():
    """SC2: high-quality but security-failing code -> reward_result.scalar == 0.0."""
    php_code = SC2_FILE.read_text()
    with patch("scripts.reward_pipeline.judge_score_single", return_value=95.0):
        results = compute_group_rewards([php_code] * 4, MagicMock(), "test-model")
    assert results[0].scalar == 0.0, "SC2 security failure must override to 0.0"
    assert results[0].breakdown.security_fail is True
```

---

### `tests/test_antihack.py` (CI-aware gate test)

**Analog:** `tests/test_bootstrap_ci.py` (`TestJaccardDisposition` — same CI-lower-bound gate pattern)

**Import + class pattern** — copy from `tests/test_bootstrap_ci.py` lines 1–18:
```python
"""Tests for D-11 anti-hack eval set CI-aware gate.

Tests are GPU-free: uses synthetic reward arrays.
Verifies bootstrap CI gate: perturbed CI upper < clean CI lower.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.compute_concentration import bootstrap_ci
```

**Gate assertion style** — mirror `TestJaccardDisposition.test_ci_lower_below_threshold_fails_even_if_point_above` (lines 93–104):
```python
class TestAntihackCIGate:
    def test_perturbed_below_clean_passes(self):
        """hi_perturbed < lo_clean -> gate PASS (adversarial case is detectably worse)."""
        clean = np.array([0.75] * 15)
        perturbed = np.array([0.30] * 15)
        lo_p, hi_p = bootstrap_ci(perturbed, n_boot=1000)
        lo_c, hi_c = bootstrap_ci(clean, n_boot=1000)
        assert hi_p < lo_c, "Gate should PASS when perturbed CI upper < clean CI lower"

    def test_ci_aware_not_bare_point(self):
        """D-09: gate based on CI bounds, not bare point estimate."""
        # Even if perturbed mean < clean mean, if CIs overlap gate must FAIL
        ...

    def test_all_axes_report_four_ci_bounds(self):
        """Acceptance report must publish lo/hi for both perturbed and clean CIs."""
        ...
```

---

### `tests/conftest.py` (shared fixtures, new at top-level tests/)

**Analog:** `tests/phase4_4/conftest.py` lines 1–60 (session-scoped path fixtures + synthetic builder helpers)

**Pattern to copy:**
```python
"""Shared fixtures for Phase 8 reward pipeline tests."""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def recalib_json(tmp_path_factory):
    """Synthetic judge_recalibration.json for tests that call _load_score_offset()."""
    d = tmp_path_factory.mktemp("recalib")
    p = d / "judge_recalibration.json"
    p.write_text(json.dumps({"score_offset": 3.58, "ci_95": [1.24, 6.09], "rank_invariant": True}))
    return p


@pytest.fixture(scope="session")
def php_fixture_dir():
    return PROJECT_ROOT / "tests" / "fixtures" / "reward_integration_cases"
```

---

## Shared Patterns

### RubricScore Signal Extraction
**Source:** `eval/rubric_scorer.py` lines 68–97 (`RubricScore` dataclass) + lines 723–768 (`score_code()`)
**Apply to:** `reward_pipeline.py` — all 70% verifiable signal extraction
```python
from eval.rubric_scorer import score_code, RubricScore, POSITIVE_CHECK_IDS, NEGATIVE_CHECK_IDS

result: RubricScore = score_code(php_code)
# PHPCS signal: result.overall (0-100)
# Security gate signal: result.triggered_checks (dim_key -> [check_ids])
#   floor_rules_applied is NOT the gate signal — see security gate note below.
# VeRPO input: result.triggered_checks across D1_wpcs + WP-standards dims
#   POSITIVE check fired = pass; NEGATIVE check fired = fail (Pitfall 5)
```

**CRITICAL NOTE on `floor_rules_applied` vs trigger-check membership:**
`apply_floor_rules()` (lines 594–628) only appends to `floor_rules_applied` when
`current > cap`. If D2_security is already at/below the cap when a CRITICAL_FLOOR_RULE
trigger fires, `floor_rules_applied` stays empty despite the rule triggering.
The security gate (D-08-05) must check trigger-check membership directly:
```python
all_triggered = {cid for ids in rubric.triggered_checks.values() for cid in ids}
security_fail = any(
    any(cid in all_triggered for cid in (rule[2] if isinstance(rule, (list,tuple)) else rule["triggers"]))
    for rule in CRITICAL_FLOOR_RULES
    if (rule[0] if isinstance(rule, (list,tuple)) else rule["dimension"]) == "D2_security"
)
```

### Judge Client Setup
**Source:** `eval/eval_judge.py` lines 363–368
**Apply to:** `reward_pipeline.py` and `tests/test_reward_pipeline.py` (mock point)
```python
import os
resolved_base_url = base_url or os.environ.get("EVAL_JUDGE_BASE_URL")
if not resolved_base_url:
    from scripts.dgx_toolbox import get_toolbox
    resolved_base_url = get_toolbox().vllm_endpoint()
client = openai.OpenAI(base_url=resolved_base_url, api_key="none")
```

### Bootstrap CI (D-09 CI-aware gate)
**Source:** `scripts/compute_concentration.py` lines 42–72
**Apply to:** `scripts/build_antihack_set.py` (gate computation) + `tests/test_antihack.py`
```python
from scripts.compute_concentration import bootstrap_ci
import numpy as np

lo, hi = bootstrap_ci(np.array(values), n_boot=1000, alpha=0.05)
# Gate: use CI bound, never bare point estimate (D-09)
```

### JSONL Per-Example Output
**Source:** `eval/eval_gen.py` / `eval/eval_judge.py` per-example JSONL logging (EVAL-06)
**Apply to:** `scripts/build_antihack_set.py` output + `reward_pipeline.py` breakdown logging
```python
# Each result written as one JSON object per line
with output_path.open("a") as f:
    f.write(json.dumps(result_dict) + "\n")
```

### Env-var LLM check suppression
**Source:** `eval/rubric_scorer.py` (RUBRIC_USE_LLM_CHECKS opt-in)
**Apply to:** `scripts/reward_pipeline.py` module top
```python
# Suppress LLM checks in reward compute — deterministic signals only (Pitfall 6)
os.environ.pop("RUBRIC_USE_LLM_CHECKS", None)
```

---

## No Analog Found

No files are without an analog. All seven files have close matches.

---

## Key Architecture Notes for Planner

1. **`eval/eval_judge.py` refactor scope:** One new public function `judge_score_single()` added after line 207 (after `parse_judge_response`). It calls `_judge_create` which is already defined in the same file (line 46). No structural changes to `run_eval()`.

2. **Security gate signal:** Use `triggered_checks` intersection with CRITICAL_FLOOR_RULES trigger-check IDs, not `floor_rules_applied`. The `apply_floor_rules()` function only records `floor_rules_applied` when capping occurs, not when the trigger fires on an already-low score.

3. **VeRPO scope (D-08-06):** Applies to WP-standards subset (`D1_wpcs` + WP-specific sniffs) per locked decision. `POSITIVE_CHECK_IDS` and `NEGATIVE_CHECK_IDS` from `rubric_scorer.py` line 39–40 are the canonical source for check polarity.

4. **Anti-hack scoring agents:** Follow SKILL.md `Agent(run_in_background=True)` pattern — agents call `reward_pipeline.compute_reward()` against local vLLM. No external Anthropic API in reward compute path.

5. **Test run commands:**
   - Unit: `pytest tests/test_reward_pipeline.py -x -q`
   - Integration: `pytest tests/test_reward_pipeline_integration.py -x -q`
   - Anti-hack: `pytest tests/test_antihack.py -x -q`
   - Full suite: `pytest tests/ -x -q`

---

## Metadata

**Analog search scope:** `eval/`, `scripts/`, `tests/`, `.claude/skills/wp-finetune:run-data-pipeline/`
**Files scanned:** 12 source files read + SKILL.md
**Pattern extraction date:** 2026-06-19
