# Phase 10: RL Comparative Evaluation - Pattern Map

**Mapped:** 2026-06-21
**Files analyzed:** 6 (4 new, 1 modified/extended, 1 orchestration-layer)
**Analogs found:** 5 / 6 (1 derived — no direct analog; see No Analog Found)

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `scripts/bootstrap_gate.py` | utility / gate | transform (CI resampling) | `scripts/build_antihack_set.py::compute_axis_gate` | exact role-match |
| `scripts/rlev02_report.py` | utility / report | batch + transform | `scripts/build_antihack_set.py::build_fixture_acceptance_report` + `score_and_gate` | role-match |
| `tests/test_bootstrap_gate.py` | test | — | `tests/test_bootstrap_ci.py` + `tests/test_antihack.py` | exact |
| `tests/test_rlev02_report.py` | test | — | `tests/test_antihack.py::TestAntihackAxisGate` + `TestAntihackAcceptanceReport` | exact |
| `.claude/skills/wp-finetune:run-evaluation/SKILL.md` | orchestration config | — | itself (no code pattern to copy) | — |
| `bootstrap_spearman_improvement()` in `scripts/bootstrap_gate.py` | utility / gate | transform (pair-bootstrap correlation) | derived from `eval/eval_judge.py::_safe_spearman` + scipy convention | derived — no direct analog |

---

## Pattern Assignments

### `scripts/bootstrap_gate.py` (utility, transform)

**Primary analog:** `scripts/build_antihack_set.py` — `compute_axis_gate` (lines 385–439)

**Imports pattern** (build_antihack_set.py lines 46–54 + lazy imports inside functions):
```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

# Note: numpy and bootstrap_ci are lazy-imported inside functions
# so --help and unit-test imports remain infra-free
```

**Core CI gate pattern — `compute_axis_gate`** (build_antihack_set.py lines 385–439):
```python
def compute_axis_gate(
    perturbed_rewards: list[float],
    clean_rewards: list[float],
    axis_name: str,
    n_boot: int = 1000,
) -> dict:
    """Gate passes when hi_perturbed < lo_clean (D-09 CI-aware)."""
    if not perturbed_rewards:
        raise ValueError(
            f"compute_axis_gate: perturbed_rewards is empty for axis '{axis_name}'. "
            "Cannot compute CI on an empty list."
        )
    # ... identical guard for clean_rewards ...

    # Lazy import keeps --help and test imports free of judge-artifact deps
    import numpy as np
    from scripts.compute_concentration import bootstrap_ci

    lo_p, hi_p = bootstrap_ci(np.array(perturbed_rewards, dtype=float), n_boot=n_boot)
    lo_c, hi_c = bootstrap_ci(np.array(clean_rewards, dtype=float), n_boot=n_boot)
    gate_pass = bool(hi_p < lo_c)

    return {
        "axis": axis_name,
        "gate_pass": gate_pass,
        "lo_perturbed": float(lo_p),
        "hi_perturbed": float(hi_p),
        "lo_clean": float(lo_c),
        "hi_clean": float(hi_c),
        "perturbed_mean": float(sum(perturbed_rewards) / len(perturbed_rewards)),
        "clean_mean": float(sum(clean_rewards) / len(clean_rewards)),
        "n_perturbed": len(perturbed_rewards),
        "n_clean": len(clean_rewards),
        # Full CI bounds reported for auditability
        "perturbed_ci": [float(lo_p), float(hi_p)],
        "clean_ci": [float(lo_c), float(hi_c)],
    }
```

**Adaptation for `check_dim_regression` (D-10-01 mean gate):** Replace the `hi_perturbed < lo_clean` shape with `lo_candidate >= baseline_mean`. The baseline side is a POINT estimate (the v1.2 SFT per-example dim mean), not a CI. Only the candidate side needs bootstrap resampling. Return shape should match the gate_row convention in `eval/eval_gate.py::check_gates` — add `"dim"` key, rename `"gate_pass"` to `"passed"`.

**Project constant to reuse:** `bootstrap_ci` signature from `scripts/compute_concentration.py` lines 42–72:
```python
def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    # rng = np.random.default_rng()
    # boot_means[i] = rng.choice(values, size=n, replace=True).mean()
    # returns (lo, hi) = alpha/2 and 1-alpha/2 percentiles of boot_means
```

**wp-bench aggregate gate (`check_wpbench_gate`, D-10-03):** Same shape as `check_dim_regression` but inputs are per-task binary scores (0.0/1.0) from the JSONL sidecar. The CI lower bound of the candidate's per-task mean must be >= 0.4616 (baseline `reasoning_score` from `output/04.4_wp_bench_results.json`). Sub-type floor checks are plain comparisons — no bootstrap needed (knowledge >= 0.45, execution >= 0.375 on the aggregate sub-type score).

**Module structure pattern** (from `scripts/rl_train.py` lines 16–35):
```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)
```

**Error handling pattern** (`compute_axis_gate` lines 406–415): validate-before-compute with explicit ValueError on empty inputs. Return dicts always include all 4 CI bounds for audit, even when not used in the gate predicate.

**Output file:** `output/rl_eval/{checkpoint}/bootstrap_gate_result.json` — single dict with per-dim gate rows plus overall `all_dims_passed: bool`. Mirror the `report` assembly in `build_fixture_acceptance_report` lines 512–528.

---

### `scripts/rlev02_report.py` (utility / report, batch + transform)

**Primary analog:** `scripts/build_antihack_set.py` — `build_fixture_acceptance_report` (lines 447–528) and `score_and_gate` (lines 536–560+)

**Report assembly pattern** (build_antihack_set.py lines 499–528):
```python
axis_results = []
all_gates_pass = True

for axis_name in PERTURBATION_AXES:
    gate_result = compute_axis_gate(perturbed, clean, axis_name, n_boot=n_boot)
    axis_results.append(gate_result)
    if not gate_result["gate_pass"]:
        all_gates_pass = False

report = {
    "report_type": "fixture_backed",          # → "live" for rlev02
    "all_axes_pass": all_gates_pass,           # → "all_gates_passed"
    "gate_criterion": "hi_perturbed < lo_clean (D-09 CI-aware)",
    "n_boot": n_boot,
    "axes": {r["axis"]: r for r in axis_results},
}

output_dir.mkdir(parents=True, exist_ok=True)
report_path = output_dir / "acceptance_report.json"
report_path.write_text(json.dumps(report, indent=2))
return report_path
```

**Adaptation for rlev02:** Replace `axes` dict with five named gate sections. Replace `all_axes_pass` with `all_gates_passed` (conjunctive). Add a `gates` list (like `gate_rows` in `eval_gate.py::check_gates`) for structured pass/fail per sub-gate. Final report must be written with `json.dumps(report, indent=2)` to match project convention.

**JSONL reader pattern** (`scripts/build_antihack_set.py::_load_source_records`, lines 63+):
```python
def _load_source_records(path: Path, ...) -> list[dict]:
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
```
Use this pattern verbatim for reading `rl_metrics.jsonl` — one dict per step.

**Five-part conjunctive gate output shape** (derived from `eval/eval_gate.py::check_gates` return shape, lines 96–170):
```python
# Each sub-gate is a dict with standardised keys:
gate_row = {
    "gate": str,       # e.g. "judge_spearman_improvement"
    "passed": bool,
    # gate-specific numeric evidence keys (CI bounds, point estimates, etc.)
}
gates = [gate_row_1, gate_row_2, gate_row_3, gate_row_4, gate_row_5]
all_gates_passed = all(g["passed"] for g in gates)
```

**rl_metrics.jsonl field names** (verified in `output/rl_checkpoints/metrics/rl_metrics.jsonl` dry-run; `scripts/rl_train.py::_log_step`):
- Reward convergence: `step`, `reward_mean`, `reward_breakdown` (dict with `wp_gen`, `wp_judge` sub-keys — A1 ASSUMED pending live run)
- Router-shift / KL stability (gate #5): `kl_sample_train_v1`, `e_frac_with_tokens_mean`, `halt_reason`
- Protected-expert Jaccard per step (gate #4): `jaccard_protected`
- Hard collapse thresholds (from `checkpoint_manifest.json::run_args`): `kl_hard=0.3`, `efrac_hard=0.5`

**No-routing-collapse gate output format** (D-10-04 #5, from RESEARCH.md verified against manifest):
```python
{
    "gate": "no_routing_collapse",
    "n_steps": int,
    "any_halt": bool,
    "max_kl": float,
    "min_efrac": float,
    "kl_breach_steps": [],      # steps where kl_sample_train_v1 >= 0.3
    "efrac_breach_steps": [],   # steps where e_frac_with_tokens_mean < 0.5
    "soft_kl_steps": [],        # advisory: kl >= 0.1
    "soft_efrac_steps": [],     # advisory: efrac < 0.7
    "passed": bool,
}
```

**Anti-hack gate pattern** (gate #3, D-10-04): Re-use `compute_axis_gate` from `build_antihack_set.py` directly — do not re-implement. Call it with RL model's perturbed/clean reward arrays and compare result CI bounds. Gate passes when `hi_perturbed_rl < lo_clean_v12` where `lo_clean_v12` comes from a live v1.2 SFT anti-hack scoring run (NOT the fixture values in `acceptance_report.json`).

**Module project-root pattern:** Same as `scripts/build_antihack_set.py` line 56:
```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

**Output file:** `output/rl_eval/rlev02_report.json` — written with `json.dumps(report, indent=2)` + `Path.write_text`.

---

### `tests/test_bootstrap_gate.py` (test)

**Primary analog:** `tests/test_bootstrap_ci.py` (full file) and `tests/test_antihack.py` (lines 1–145)

**File header and import pattern** (test_bootstrap_ci.py lines 1–11, test_antihack.py lines 1–41):
```python
"""Unit tests for scripts/bootstrap_gate.py — CI-aware dim regression gate.

Tests are GPU-free: uses synthetic numpy arrays.
Covers D-10-01 (per-dim CI gate), D-10-01 Spearman improvement,
D-10-03 wp-bench aggregate gate, and D-10-03 per-task floor checks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.compute_concentration import bootstrap_ci

PROJECT_ROOT = Path(__file__).resolve().parents[1]
```

**Class-per-requirement test grouping pattern** (test_antihack.py structure):
```python
class TestBootstrapGateDimRegression:
    """D-10-01 CI-aware dimension regression gate."""

    def test_no_regression_passes(self):
        """lo_candidate >= baseline_mean -> PASS."""
        ...

    def test_real_regression_fails(self):
        """lo_candidate < baseline_mean -> FAIL (real regression detected)."""
        ...

    def test_within_noise_dip_passes(self):
        """High-variance candidate whose lo_candidate >= baseline_mean -> PASS
        (within-noise dip does not fail the gate — D-10-01)."""
        ...

class TestBootstrapGateSpearmanImprovement:
    """Judge Spearman improvement beyond noise (D-10-01 judge gate, D-10-04 #1)."""
    ...

class TestBootstrapGateWpbench:
    """D-10-03 wp-bench hard gate: aggregate CI lower bound + sub-type floors."""
    ...
```

**Seed pattern** (test_bootstrap_ci.py line 22, test_antihack.py line 67): Use `np.random.seed(N)` before bootstrap calls in tests requiring reproducible CI results. Do NOT use `np.random.default_rng()` in tests — that's the production path; tests seed the global RNG for stability.

**Constant array CI is tight** pattern (test_bootstrap_ci.py lines 30–34): Build passing gate tests with constant arrays (zero variance → CI collapses to point → deterministic pass/fail).

**skip pattern for missing on-disk artifacts** (test_antihack.py lines 163–165):
```python
if not SOURCE_JSONL.exists():
    pytest.skip("Source JSONL not available in this environment")
```
Apply to any test reading `output/rl_eval/*/eval_gen_results.jsonl` — skip when not present (Wave 0; live data in Wave 1+).

**Import-importability guard** (test_antihack.py lines 49–51):
```python
def test_bootstrap_gate_importable():
    from scripts.bootstrap_gate import check_dim_regression, check_wpbench_gate
    assert callable(check_dim_regression)
    assert callable(check_wpbench_gate)
```

---

### `tests/test_rlev02_report.py` (test)

**Primary analog:** `tests/test_antihack.py` — `TestAntihackAxisGate` and `TestAntihackAcceptanceReport` classes (lines 155+)

**Report schema verification pattern** (test_antihack.py lines 126–145):
```python
def test_all_axes_report_four_ci_bounds(self):
    report = {
        "perturbed_ci": [float(lo_p), float(hi_p)],
        "clean_ci": [float(lo_c), float(hi_c)],
        "gate_pass": bool(hi_p < lo_c),
    }
    assert len(report["perturbed_ci"]) == 2
    assert len(report["clean_ci"]) == 2
    assert "gate_pass" in report
    assert isinstance(report["gate_pass"], bool)
```
Adapt: verify `all_gates_passed`, `gates` list, each gate has `"gate"` + `"passed"` keys, numeric evidence keys present.

**Fixture-based JSONL test** — build a synthetic `rl_metrics.jsonl` in `tmp_path` and pass its path to `rlev02_report.py::build_report`. Same pattern as conftest.py's `recalib_json` fixture (lines 23–41):
```python
@pytest.fixture
def rl_metrics_jsonl(tmp_path):
    lines = [
        json.dumps({"step": 1, "reward_mean": 0.52, "kl_sample_train_v1": 0.05,
                    "e_frac_with_tokens_mean": 0.72, "halt_reason": None,
                    "jaccard_protected": 0.91}),
        json.dumps({"step": 2, "reward_mean": 0.58, "kl_sample_train_v1": 0.07,
                    "e_frac_with_tokens_mean": 0.75, "halt_reason": None,
                    "jaccard_protected": 0.93}),
    ]
    p = tmp_path / "rl_metrics.jsonl"
    p.write_text("\n".join(lines))
    return p
```

**Conjunctive gate test** pattern (from `eval_gate.py` test idiom, test_eval_gate.py lines 60–76):
```python
def test_conjunctive_gate_all_pass(rl_metrics_jsonl, tmp_path):
    from scripts.rlev02_report import build_report
    report = build_report(rl_metrics_jsonl=rl_metrics_jsonl, ...)
    assert report["all_gates_passed"] is True
    assert all(g["passed"] for g in report["gates"])

def test_conjunctive_gate_one_fail_fails_all(rl_metrics_jsonl, tmp_path):
    """Single sub-gate failure must set all_gates_passed=False."""
    ...
    assert report["all_gates_passed"] is False
```

---

## Shared Patterns

### Bootstrap CI (cross-cutting — all gate scripts)
**Source:** `scripts/compute_concentration.py` lines 42–72
**Apply to:** `scripts/bootstrap_gate.py` (mean-CI gate), `scripts/rlev02_report.py` (anti-hack re-run)
**Import:** `from scripts.compute_concentration import bootstrap_ci`
**Lazy import inside functions** (from `build_antihack_set.py` line 417–419): wrap in function body to keep module-level imports infra-free.

### Project Root Resolution
**Source:** `scripts/rl_train.py` lines 31–33 / `scripts/build_antihack_set.py` line 56
**Apply to:** all new `scripts/` files
```python
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
```
(Use `parents[1]` from `scripts/`; use `parents[2]` from `tests/`.)

### JSON Report Write
**Source:** `scripts/build_antihack_set.py` lines 525–527
**Apply to:** `scripts/bootstrap_gate.py`, `scripts/rlev02_report.py`
```python
output_dir.mkdir(parents=True, exist_ok=True)
report_path = output_dir / "rlev02_report.json"
report_path.write_text(json.dumps(report, indent=2))
return report_path
```

### Per-dimension Spearman Point Estimate
**Source:** `eval/eval_judge.py` lines 26–27, 326–343
**Apply to:** `scripts/bootstrap_gate.py::bootstrap_spearman_improvement` for the project-standard scipy import and `result.statistic` attribute access pattern:
```python
from scipy.stats import spearmanr

def _safe_spearman(xs, ys) -> dict:
    result = spearmanr(xs, ys)
    return {
        "corr": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_pairs": len(xs),
    }
```
The pair-bootstrap for the improvement check uses `result.statistic` (not `.correlation` — note the attribute name in the project's eval_judge.py).

### Error Handling / Input Validation
**Source:** `scripts/build_antihack_set.py` lines 406–415
**Apply to:** all gate functions in `scripts/bootstrap_gate.py`
```python
if not candidate_scores:
    raise ValueError(
        f"check_dim_regression: candidate_scores is empty for dim '{dim_key}'. "
        "Cannot compute CI on an empty list."
    )
```

### JSONL Reading
**Source:** `scripts/build_antihack_set.py::_load_source_records` lines 63–90 (pattern: open + strip + json.loads per line)
**Apply to:** `scripts/rlev02_report.py` for reading `rl_metrics.jsonl`

### Test File Preamble
**Source:** `tests/test_bootstrap_ci.py` lines 1–12
**Apply to:** both new test files
```python
"""..."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
```
Plus `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` is NOT needed — `conftest.py` already inserts project root at session start.

---

## No Analog Found

| File / Function | Role | Data Flow | Reason |
|-----------------|------|-----------|--------|
| `bootstrap_spearman_improvement()` in `scripts/bootstrap_gate.py` | gate (correlation) | transform (pair bootstrap) | No existing function resamples `(pred, GT)` pairs to bootstrap a Spearman correlation difference. `bootstrap_ci` handles mean-of-values only. The RESEARCH.md Pattern 1b (lines 206–241) is the specification to implement. The scipy call site in `eval/eval_judge.py::_safe_spearman` (lines 326–343) provides the `spearmanr` import convention and `.statistic` attribute to reuse — but the pair-bootstrap loop itself is novel to this project. |
| `.claude/skills/wp-finetune:run-evaluation/SKILL.md` | orchestration config | — | Orchestration-layer markdown consumed by the skill dispatcher. No Python code pattern to copy. Phase 10 extends the skill by adding RL-specific eval steps to the process narrative, but this is prose documentation, not code. The skill's process flow (Step 0 inventory → Step 1 profiling → Step 2 eval → triage) provides the ordering reference for how Phase 10's eval run should be sequenced in a plan action, but there is no copyable code excerpt. |

---

## Critical Anti-Patterns (Planner Guard)

1. **`bootstrap_ci(corr_array)` for the Spearman gate**: `bootstrap_ci` resamples the MEAN of a 1-D values array. Spearman correlation is not a mean — feeding a scalar correlation array into it is mathematically incorrect. The Spearman improvement gate MUST use pair-level resampling (resample `(pred_score, gt_score)` pairs, recompute `spearmanr` on each resample).

2. **Reading `eval_gen_results.json` for per-example scores**: `bootstrap_gate.py` must read `eval_gen_results.jsonl` (the per-example sidecar) to get the `dimension_scores` array for resampling — not the aggregate `.json` summary file.

3. **Using fixture anti-hack CI bounds as the D-10-04 #3 baseline**: `output/antihack_validation/acceptance_report.json` is `report_type: fixture_backed` with synthetic reward values. Wave 1 must score the v1.2 SFT model against the real anti-hack JSONLs to establish a live baseline before comparing RL model results.

4. **Using `eval/eval_gate.py::run_gate()` for CI-aware gates**: `run_gate()` is a point-threshold comparator (`actual >= target`). It has no bootstrap CI logic. Use it only as a helper for point-estimate checks; implement all D-10-01/03 CI logic in `scripts/bootstrap_gate.py`.

---

## Metadata

**Analog search scope:** `scripts/`, `eval/`, `tests/`, `.claude/skills/`
**Files read for pattern extraction:** `scripts/compute_concentration.py`, `scripts/build_antihack_set.py`, `scripts/rl_train.py`, `eval/eval_gate.py`, `eval/eval_judge.py`, `tests/test_bootstrap_ci.py`, `tests/test_antihack.py`, `tests/test_eval_gate.py`, `tests/conftest.py`, `.claude/skills/wp-finetune:run-evaluation/SKILL.md`
**Pattern extraction date:** 2026-06-21
