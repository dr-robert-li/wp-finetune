# Phase 7: Router Profiling & Protected Expert Set - Pattern Map

**Mapped:** 2026-06-14
**Files analyzed:** 11 (3 new scripts, 1 new skill, 5 new test files, 2 modified files)
**Analogs found:** 8 / 11 (3 no-analog items: cumulative coverage, layer-depth skew, bootstrap CI)

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `scripts/profile_merged_model.py` (new) | script (profiler) | batch (GPU forward pass) | `scripts/profile_base_model.py` | exact |
| `scripts/compute_concentration.py` (new) | utility (analysis) | transform (JSONL → metrics) | `scripts/_04.4_dit02_expert_delta_norms.py` | role-match (formulas same, I/O differs) |
| `scripts/extract_protected_mask.py` (new) | utility (analysis) | transform (counts → mask) | `scripts/_04.4_dit02_expert_delta_norms.py` | partial (decision-logic structure only) |
| `.claude/skills/wp-finetune:run-profiling/SKILL.md` (new) | skill (DGX orchestration) | request-response | `.claude/skills/wp-finetune:run-evaluation/SKILL.md` | exact |
| `tests/test_routing_collector.py` (new) | test | — | `tests/test_eeff.py` | exact |
| `tests/test_jaccard_stability.py` (new) | test | — | `tests/test_eeff.py` + `tests/phase4_4/test_revl08_reasoning_length.py` | role-match |
| `tests/test_concentration.py` (new) | test | — | `tests/test_eeff.py` + `scripts/_04.4_dit02_expert_delta_norms.py` | role-match |
| `tests/test_protected_mask.py` (new) | test | — | `tests/test_eeff.py` | role-match |
| `tests/test_bootstrap_ci.py` (new) | test | — | `tests/phase4_4/test_revl08_reasoning_length.py` | partial (statistical helper pattern) |
| `scripts/profile_base_model.py` (modified) | script (profiler) | batch (GPU forward pass) | self | — |
| `tests/test_eeff.py` (modified) | test | — | self | — |

**Modified files detail:**
- `scripts/profile_base_model.py`: `write_profiling_jsonl` must gain a `model_tag: str = "base"` parameter (Adaptation Note 1). One-line signature change; backward-compatible.
- `tests/test_eeff.py`: `test_model_field_is_base` (lines 199-207) tests the hardcoded `"base"` string. Must be updated to pass `model_tag="base"` explicitly once `write_profiling_jsonl` is parameterized.

---

## Pattern Assignments

### `scripts/profile_merged_model.py` (script, batch/GPU)

**Analog:** `scripts/profile_base_model.py`

**Reuse scope:** This file is a thin adaptation of `profile_base_model.py`. The core classes (`RoutingCollector`, `compute_eeff`, `set_token_types`, `write_profiling_jsonl`, `discover_dataset_dirs`) are imported unchanged. Only the CLI defaults, the `model` field value in output, and the adapter-loading path differ.

**Imports pattern** (lines 16-28 of analog):
```python
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
```

**Stimulus loading — verbatim reuse** (lines 42-60 of analog, `discover_dataset_dirs`):
```python
def discover_dataset_dirs(final_dataset_dir: Path) -> dict[str, str]:
    """Auto-discover dataset directories containing openai_train.jsonl."""
    results = {}
    root_train = final_dataset_dir / "openai_train.jsonl"
    if root_train.exists():
        results["current"] = str(root_train)
    if final_dataset_dir.exists():
        for d in sorted(final_dataset_dir.iterdir()):
            if d.is_dir():
                train_file = d / "openai_train.jsonl"
                if train_file.exists():
                    results[d.name] = str(train_file)
    return results
```
D-05 (amended) uses `discover_dataset_dirs(project_root / "data" / "final_dataset")` then filters to `ratio_30_70` via `--ratio ratio_30_70`. This is identical to the base profiler's pattern (lines 630-640 of analog).

**Hook registration — verbatim reuse** (lines 454-459 of analog):
```python
base = model.get_base_model() if hasattr(model, "get_base_model") else model
hooks = []
for i, layer in enumerate(base.model.layers):
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
        h = layer.mlp.gate.register_forward_hook(collector.make_hook(i))
        hooks.append(h)
```
Log hook count at startup: `logger.info(f"Registered {len(hooks)} hooks (expected 48)")`.

**Model loading — adaptation point** (lines 643-655 of analog):
```python
# ADAPTATION: merged model — no --adapter, no PeftModel wrapper.
# The hasattr(model, 'get_base_model') guard already handles this gracefully.
model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    dtype=torch.bfloat16,
    device_map="auto",
)
# No PeftModel.from_pretrained() call. No adapter path arg.
```

**Output model field — adaptation point** (line 307 of analog, `write_profiling_jsonl`):
```python
# ORIGINAL (base profiler):
"model": "base",

# ADAPTED (merged profiler) — parameterize the field:
"model": model_tag,  # pass "reasoning-merged-v4" as CLI arg default
```
Note: `test_eeff.py::test_model_field_is_base` (line 199-207) tests the hardcoded `"base"` string. The adapted script must parameterize this field and the planner must update that test assertion.

**CUDA guard — verbatim reuse** (lines 615-620 of analog):
```python
if not torch.cuda.is_available() and not args.allow_cpu:
    print("ERROR: torch.cuda.is_available() is False — refusing to run a 30B forward "
          "pass on CPU. ...")
    sys.exit(2)
```

**Batch processing loop — verbatim reuse** (lines 485-517 of analog):
```python
for batch_start in range(0, len(subsample), batch_size):
    batch = subsample[batch_start: batch_start + batch_size]
    texts = []
    for ex in batch:
        messages = ex.get("messages", [])
        if messages:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            text = ex.get("text", "")
        texts.append(text)
    enc = tokenizer(texts, return_tensors="pt", max_length=max_seq_len,
                    truncation=True, padding=True)
    input_ids = enc["input_ids"]
    collector.set_token_types(input_ids)
    with torch.no_grad():
        model(input_ids=input_ids.to(model.device))
```

**Hook cleanup — verbatim reuse** (lines 547-549 of analog):
```python
finally:
    for h in hooks:
        h.remove()
```

**CLI output directory** — change default from `output/profiling` to `output/profiling/reasoning-merged-v4` to avoid overwriting the baseline `base_model_eeff.jsonl`.

---

### `scripts/compute_concentration.py` (utility, transform)

**Analog:** `scripts/_04.4_dit02_expert_delta_norms.py`

**Reuse scope:** Copy the formula implementations only. The I/O structure is different: this script reads the JSONL output of `profile_merged_model.py` (per-layer expert count dicts), not adapter tensors. The formulas for CV, entropy, top-K mass, and median aggregation are direct copies.

**Normalized entropy formula** (lines 71-77 of analog):
```python
def _entropy_norm(norms: list[float]) -> float:
    s = sum(norms)
    if s <= 0:
        return 1.0
    ps = [n / s for n in norms if n > 0]
    H = -sum(p * math.log(p) for p in ps)
    return H / math.log(N_EXPERTS)  # normalized to [0,1]; 1.0 = uniform
```
Adapt: replace `norms` with `expert_count_array` (numpy array from JSONL `expert_counts_*` dict).

**Top-K mass formula** (lines 80-84 of analog):
```python
def _topk_mass(norms: list[float], k: int) -> float:
    s = sum(norms)
    if s <= 0:
        return 0.0
    return sum(sorted(norms, reverse=True)[:k]) / s
```

**CV and max-over-mean** (lines 99-108 of analog):
```python
mean = statistics.fmean(norms)
std = statistics.pstdev(norms)
cv = std / mean if mean else 0.0
max_over_mean = max(norms) / mean if mean else 0.0
```
In the Phase 7 script, use `numpy` instead of `statistics` since counts arrive as numpy arrays: `cv = counts.std() / counts.mean()` if `counts.mean() > 0` else `0.0`.

**Median aggregation** (lines 112-114 of analog):
```python
med_ent = statistics.median(p["norm_entropy"] for p in per_layer)
med_cv  = statistics.median(p["cv"] for p in per_layer)
```

**E_eff delta vs baseline** — new code, no codebase analog. Load `output/profiling/base_model_eeff.jsonl` filtered to `ratio=30_70`, join on `layer_idx`, subtract `eeff_total` values. Report as `eeff_delta = merged_eeff - base_eeff` per layer.

**CRITICAL — ratio-key join seam (D-08):** `base_model_eeff.jsonl` records use `"ratio": "30_70"` (confirmed by inspecting live file). `discover_dataset_dirs()` keys on the directory name, producing `"ratio_30_70"` (prefixed). The merged profiler must write `"ratio": "30_70"` (or normalize at join time) to prevent the D-08 delta join silently yielding zero matches. Add an explicit normalization step: `ratio_key = ratio_dir_name.lstrip("ratio_")` before writing output records, matching the base file's convention.

**Cumulative coverage curve** — no codebase analog. Sort expert counts descending, compute cumsum/total per layer. New code.

**Layer-depth skew** — no codebase analog. Compute mean concentration metric for early (layers 0-15) vs late (layers 32-47) layers and report ratio. New code.

**Output schema** — follow the same `json.dumps(result, indent=2)` + `Path.write_text()` pattern as the analog (lines 147 of analog):
```python
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2))
```

---

### `scripts/extract_protected_mask.py` (utility, transform)

**Analog:** `scripts/_04.4_dit02_expert_delta_norms.py` (decision-logic structure only)

**Reuse scope:** The pre-committed threshold + verdict structure from the analog is the closest structural match. The mask logic itself is new (D-03 co-activation rule). Copy the JSON output pattern and the threshold-before-looking discipline.

**Co-activation mask logic** — new code implementing D-03 (from RESEARCH.md Pattern 5):
```python
def extract_protected_mask(
    counts_wp_gen: np.ndarray,   # [n_layers, n_experts]
    counts_wp_judge: np.ndarray, # [n_layers, n_experts]
) -> np.ndarray:
    """Bool mask [n_layers, n_experts]. Expert protected iff above per-layer
    mean for BOTH wp_gen AND wp_judge (D-03 conservative co-activation)."""
    mean_gen   = counts_wp_gen.mean(axis=1, keepdims=True)
    mean_judge = counts_wp_judge.mean(axis=1, keepdims=True)
    return (counts_wp_gen > mean_gen) & (counts_wp_judge > mean_judge)
```

**Sensitivity table** — three threshold variants alongside the chosen mask (D-04):
```python
# mean threshold (D-03 conservative): expert > layer mean in BOTH splits
# median threshold: expert > layer median in BOTH splits
# top-K intersection: expert in top-K by count in BOTH splits (K=16 or top_k*2)
sensitivity = {
    "mean_threshold": {"mask_size_per_layer": [...], "total_protected": int},
    "median_threshold": {"mask_size_per_layer": [...], "total_protected": int},
    "topk_intersection_k16": {"mask_size_per_layer": [...], "total_protected": int},
}
```

**Mask export pattern** — new code, no analog:
```python
np.save(out / "protected_expert_mask.npy", mask)
sidecar = {
    str(layer_idx): [int(e) for e in np.where(mask[layer_idx])[0]]
    for layer_idx in range(mask.shape[0])
}
with open(out / "protected_expert_mask.json", "w") as f:
    json.dump(sidecar, f, indent=2)
```

**JSON result output** — follow analog structure (lines 129-147):
```python
result = {
    "analysis": "protected_expert_mask_extraction",
    "model": "reasoning-merged-v4",
    "stimulus": "data/final_dataset/ratio_30_70/openai_train.jsonl",
    "n_layers": 48, "n_experts": 128, "top_k": 8,
    "rule": "D-03_conservative_co_activation",
    "total_protected": int(mask.sum()),
    "mean_protected_per_layer": float(mask.sum(axis=1).mean()),
    "sensitivity_table": sensitivity,
    "per_layer": [{"layer_idx": i, ...} for i in range(48)],
}
OUT.write_text(json.dumps(result, indent=2))
```

---

### `.claude/skills/wp-finetune:run-profiling/SKILL.md` (skill, DGX orchestration)

**Analog:** `.claude/skills/wp-finetune:run-evaluation/SKILL.md`

**Reuse scope:** Mirror the run-evaluation skill's structure exactly. Copy: Step 0 DGX readiness checks, idempotency `.complete` marker pattern, observe-evaluation telemetry embed decision (lightweight vs full agent team), and the CLI reference table format.

**Step 0: DGX readiness checks** (lines 52-63 of analog):
```bash
nvidia-smi 2>&1 | head -1
python3 -c "import torch; print(torch.cuda.is_available())"
ls models/qwen3-30b-wp-30_70-reasoning-merged-v4/config.json 2>/dev/null
ls output/profiling/base_model_eeff.jsonl 2>/dev/null  # baseline must exist for D-08 delta
```

**Telemetry embed decision** (lines 19-28 of analog — copy verbatim, adapt output path):
```
Default: Lightweight monitor only. The observe-evaluation agent team (3 agents, ~1.2 GB overhead)
should NOT be spawned during profiling if memory headroom <25 GB. Profiling loads the 30B model
directly (no vLLM), consuming ~60 GB VRAM — even tighter than evaluation.
```

**Lightweight monitor adapt** — change the file-existence checks in the monitor loop (lines 84-104 of analog) from `eval_triage` outputs to profiling outputs:
```bash
# Replace eval-specific checks with profiling-specific:
profiling_jsonl_done=$([[ -f output/profiling/reasoning-merged-v4/routing_report.jsonl ]] && echo "true" || echo "false")
concentration_done=$([[ -f output/profiling/reasoning-merged-v4/concentration_report.json ]] && echo "true" || echo "false")
mask_done=$([[ -f output/profiling/reasoning-merged-v4/protected_expert_mask.npy ]] && echo "true" || echo "false")
```

**Idempotency `.complete` marker pattern** (lines 457-462 of analog):
```
- output/profiling/reasoning-merged-v4/.profile_complete   — routing pass done
- output/profiling/reasoning-merged-v4/.concentration_complete — concentration.py done
- output/profiling/reasoning-merged-v4/.mask_complete      — mask extraction done
Re-running the skill resumes from the last incomplete step. Use --force to re-run everything.
```

**DGX container invocation** — the profiling script runs INSIDE the `eval_toolbox` container (not from HOST like eval), because it requires CUDA directly. This differs from run-evaluation where the orchestrator runs on HOST:
```bash
# Run inside container — DGX Toolbox manages lifecycle
bash deps/dgx-toolbox/containers/ngc-pytorch.sh python3 -m scripts.profile_merged_model \
  --model-path models/qwen3-30b-wp-30_70-reasoning-merged-v4 \
  --ratio ratio_30_70 \
  --output-dir output/profiling/reasoning-merged-v4
```

**Trigger:** User says: "run profiling", "profile the merged model", "run router profiling", "/run-profiling"

---

## Test File Patterns

### `tests/test_routing_collector.py` (covers PROF-01, PROF-02)

**Analog:** `tests/test_eeff.py`

**Class structure pattern** (lines 57-336 of analog):
```python
import json, math, tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from scripts.profile_base_model import (
    RoutingCollector, WP_GEN_ID, WP_JUDGE_ID, compute_eeff, write_profiling_jsonl,
)

class TestRoutingCollectorHookAccumulates:
    """PROF-01: hook fires on gate, counts accumulate correctly per layer."""

    def _collector(self):
        return RoutingCollector(n_layers=2, n_experts=8, top_k=2, pad_token_id=151643)

    def test_hook_accumulates_per_layer(self):
        """make_hook accumulates router_indices into correct layer bucket."""
        collector = self._collector()
        collector._current_token_types = ["wp_gen", "wp_gen"]
        router_indices = torch.tensor([[0, 1], [2, 3]])
        mock_outputs = (None, None, router_indices)
        hook_fn = collector.make_hook(layer_idx=0)
        hook_fn(None, None, mock_outputs)
        assert collector._counts_total[0].get(0, 0) > 0
        assert collector._n_tokens_total[0] == 2
```
The mock hook output pattern `(None, None, router_indices)` is established in `test_eeff.py` lines 321-325 — copy it exactly.

**Token attribution test class** — copy `TestTagTokenTypes` from `test_eeff.py` lines 94-153 verbatim; it already covers PROF-02 set_token_types behavior.

**model field adaptation test** — update `test_model_field_is_base` to expect `"reasoning-merged-v4"` when `model_tag` arg is set:
```python
def test_model_field_is_parameterized(self):
    """JSONL 'model' field reflects the model_tag argument."""
    # ... write_profiling_jsonl called with model_tag="reasoning-merged-v4"
    for rec in records:
        assert rec["model"] == "reasoning-merged-v4"
```

---

### `tests/test_jaccard_stability.py` (covers PROF-03)

**Analog:** `tests/test_eeff.py` (class structure) + `tests/phase4_4/test_revl08_reasoning_length.py` (statistical helper asserted on known distribution)

**Pattern — statistical helper on known values** (lines 12-24 of `test_revl08_reasoning_length.py`):
```python
class TestPercentile:
    def test_empty_returns_zero(self):
        assert _percentile([], 0.95) == 0.0
    def test_single_element(self):
        assert _percentile([42], 0.95) == 42.0
    def test_known_distribution(self):
        vals = list(range(1, 101))
        assert 94.0 <= _percentile(vals, 0.95) <= 96.0
```
Apply same pattern to Jaccard: test on a known top-K overlap, assert exact value:
```python
class TestJaccardStability:
    def test_identical_profiles_jaccard_one(self):
        """Identical full and subsample profiles -> Jaccard=1.0 per layer."""
        counts = np.zeros((2, 8))
        counts[0, [0, 1, 2]] = [100, 80, 60]
        counts[1, [0, 1]] = [100, 50]
        j = compute_jaccard_stability(counts, counts, top_k=2)
        assert np.all(j == 1.0)

    def test_disjoint_profiles_jaccard_zero(self):
        """Completely disjoint top-K sets -> Jaccard=0.0."""
        full = np.zeros((1, 8)); full[0, [0, 1]] = 100
        sub  = np.zeros((1, 8)); sub[0,  [4, 5]] = 100
        j = compute_jaccard_stability(full, sub, top_k=2)
        assert j[0] == 0.0

    def test_gate_passes_at_threshold(self):
        """Jaccard >= 0.94 gate passes when all layers meet threshold."""
        j = np.array([0.94, 0.96, 0.95])
        assert np.all(j >= 0.94)
```

---

### `tests/test_concentration.py` (covers PROF-04)

**Analog:** `tests/test_eeff.py` (GPU-free, mock data pattern) + `_04.4_dit02_expert_delta_norms.py` (formulas)

**Key assertions — known-value tests on formulas:**
```python
class TestConcentrationMetrics:
    def test_cv_uniform_distribution(self):
        """Uniform counts -> CV ~= 0."""
        counts = np.ones(128) * 100.0
        cv = counts.std() / counts.mean()
        assert cv < 0.01

    def test_eeff_uniform_is_128(self):
        """Uniform expert counts -> E_eff ~= 128."""
        counts = {i: 100 for i in range(128)}
        assert abs(compute_eeff(counts) - 128.0) < 0.5

    def test_cumulative_coverage_sums_to_one(self):
        """Cumulative coverage at 128 experts = 1.0."""
        counts = np.random.randint(1, 100, 128).astype(float)
        sorted_desc = np.sort(counts)[::-1]
        cumsum = np.cumsum(sorted_desc) / counts.sum()
        assert abs(cumsum[-1] - 1.0) < 1e-9

    def test_eeff_delta_direction(self):
        """E_eff delta = merged - base; positive means more diffuse after fine-tuning."""
        base_eeff = 45.0
        merged_eeff = 42.0
        delta = merged_eeff - base_eeff
        assert delta == -3.0  # more concentrated = negative delta
```

---

### `tests/test_protected_mask.py` (covers D-03)

**Analog:** `tests/test_eeff.py` (class structure, mock collector data)

**Key assertions — co-activation rule:**
```python
class TestProtectedMask:
    def test_co_activation_rule_flags_dual_purpose(self):
        """Expert above mean in BOTH splits is flagged."""
        gen   = np.zeros((1, 4)); gen[0]   = [200, 50, 50, 50]   # expert 0 above mean
        judge = np.zeros((1, 4)); judge[0] = [200, 50, 50, 50]   # expert 0 above mean
        mask = extract_protected_mask(gen, judge)
        assert mask[0, 0] == True

    def test_single_split_above_mean_not_flagged(self):
        """Expert above mean in only ONE split is NOT flagged."""
        gen   = np.zeros((1, 4)); gen[0]   = [200, 50, 50, 50]   # expert 0 above mean gen
        judge = np.zeros((1, 4)); judge[0] = [50, 200, 50, 50]   # expert 1 above mean judge
        mask = extract_protected_mask(gen, judge)
        assert mask[0, 0] == False   # high gen, not high judge
        assert mask[0, 1] == False   # high judge, not high gen

    def test_mask_shape(self):
        """Output mask shape is [n_layers, n_experts]."""
        gen   = np.ones((48, 128))
        judge = np.ones((48, 128))
        mask = extract_protected_mask(gen, judge)
        assert mask.shape == (48, 128)
        assert mask.dtype == bool
```

---

### `tests/test_bootstrap_ci.py` (covers D-09)

**Analog:** `tests/phase4_4/test_revl08_reasoning_length.py` (statistical helper structure)

**No existing bootstrap CI in codebase.** The `run_grid_eval.py` `ci_lower` field (lines 215, 264-267) reads a pre-computed value from `eval_judge.py` output — `_safe_spearman` uses `scipy.stats.spearmanr` with no bootstrap. D-09 bootstrap CI is entirely new code.

**Pattern — test statistical helper on known distribution:**
```python
class TestBootstrapCI:
    def test_known_distribution_ci_contains_true_mean(self):
        """Bootstrap CI contains true mean for a known distribution."""
        np.random.seed(42)
        values = np.array([1.0] * 50 + [2.0] * 50)  # true mean = 1.5
        lo, hi = bootstrap_ci(values, n_boot=500, alpha=0.05)
        assert lo < 1.5 < hi

    def test_constant_array_ci_is_tight(self):
        """Constant array bootstrap CI collapses to the constant."""
        values = np.ones(100) * 5.0
        lo, hi = bootstrap_ci(values, n_boot=200, alpha=0.05)
        assert abs(lo - 5.0) < 0.01
        assert abs(hi - 5.0) < 0.01

    def test_ci_lower_used_for_gate_disposition(self):
        """CI-aware gate: only passes when lower bound clears threshold (D-09)."""
        # Mirrors run_grid_eval.py lines 264-267 ci_lower mode
        threshold = 0.94
        lo, hi = 0.91, 0.97   # lower bound below threshold
        gate_passes = lo >= threshold
        assert gate_passes is False  # lower bound fails even though point > threshold
```

---

## Shared Patterns

### CUDA Guard
**Source:** `scripts/profile_base_model.py` lines 615-620
**Apply to:** `profile_merged_model.py`
```python
if not torch.cuda.is_available() and not args.allow_cpu:
    print("ERROR: torch.cuda.is_available() is False ...")
    sys.exit(2)
```

### Hook Cleanup (finally block)
**Source:** `scripts/profile_base_model.py` lines 465, 547-549
**Apply to:** `profile_merged_model.py`
```python
try:
    # ... profiling loop
finally:
    for h in hooks:
        h.remove()
```

### NaN-to-null serialization
**Source:** `scripts/profile_base_model.py` lines 250-254 (`_nan_to_null`)
**Apply to:** `profile_merged_model.py`, `compute_concentration.py` output
```python
def _nan_to_null(value: float):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
```

### CI-aware gate disposition
**Source:** `scripts/run_grid_eval.py` lines 264-267
**Apply to:** Jaccard stability gate, E_eff delta gate, protected-expert cutoff gate in `compute_concentration.py` and `extract_protected_mask.py`
```python
# "point" mode: metric >= bar (unsafe — D-09 says use CI-aware)
# "ci_lower" mode: lower_bound >= bar (D-09 compliant)
gate_passes = (ci_lower >= threshold)  # NOT bare_metric >= threshold
```

### Idempotency `.complete` marker
**Source:** `.claude/skills/wp-finetune:run-evaluation/SKILL.md` lines 457-462
**Apply to:** `run-profiling` skill
```
# Write on step completion, check before re-running:
Path("output/profiling/reasoning-merged-v4/.profile_complete").touch()
```

### JSON output with `indent=2` + `mkdir(parents=True, exist_ok=True)`
**Source:** `scripts/_04.4_dit02_expert_delta_norms.py` lines 146-147 and `scripts/profile_base_model.py` lines 279-280
**Apply to:** All new post-processing scripts
```python
out_path = Path(out_path)
out_path.parent.mkdir(parents=True, exist_ok=True)
# ...
OUT.write_text(json.dumps(result, indent=2))
```

---

## No Analog Found

| File/Feature | Role | Data Flow | Reason |
|---|---|---|---|
| `scripts/compute_concentration.py` cumulative coverage curve | utility | transform | No cumsum/coverage-curve computation exists in codebase |
| `scripts/compute_concentration.py` layer-depth skew | utility | transform | No layer-depth stratification exists in codebase |
| `scripts/extract_protected_mask.py` mask export (`.npy` + JSON sidecar) | utility | file-I/O | No numpy boolean mask export exists in codebase |
| D-09 bootstrap CI computation | utility | transform | `eval_judge.py::_safe_spearman` uses `scipy.stats.spearmanr` directly (no bootstrap); `run_grid_eval.py` reads a pre-computed `ci_lower` field. No bootstrap resampling exists anywhere in codebase |

For these, implement from RESEARCH.md Patterns 4-6 directly (the Jaccard, mask extraction, and bootstrap_ci pseudo-code there is specification-ready).

---

## Key Adaptation Notes for Planner

1. **`write_profiling_jsonl` hardcodes `"model": "base"` at line 307 of `scripts/profile_base_model.py`.** Add `model_tag: str = "base"` parameter. Backward-compatible (existing callers get `"base"` by default). `profile_merged_model.py` passes `model_tag="reasoning-merged-v4"`. `tests/test_eeff.py::test_model_field_is_base` (lines 199-207) must be updated to pass `model_tag="base"` explicitly.

2. **Ratio-key join seam (D-08 delta will silently yield zero matches if not handled).** `output/profiling/base_model_eeff.jsonl` stores `"ratio": "30_70"` (confirmed). `discover_dataset_dirs()` keys on directory name, producing `"ratio_30_70"`. The merged profiler's output records must normalize the ratio key to `"30_70"` (strip the `ratio_` prefix), or `compute_concentration.py` must normalize at join time. Either is fine; the planner must specify which. Failure mode: D-08 delta table is empty at runtime, no error surfaced.

3. **Single-ratio profiling.** The base profiler loops over all discovered ratios. The merged profiler passes `--ratio ratio_30_70` (D-01) to restrict to one. This uses the existing `args.ratio` filter already in `profile_base_model.py` lines 635-640.

4. **Output directory collision risk.** The base profiler writes to `output/profiling/base_model_eeff.jsonl`. The merged profiler MUST write to `output/profiling/reasoning-merged-v4/routing_report.jsonl` — never the base path. Gate this in the script with an explicit path check.

5. **No test infrastructure gap for routing_collector.** `test_eeff.py` already covers `RoutingCollector.make_hook` padding exclusion (lines 313-335) and `set_token_types` (lines 94-153). `test_routing_collector.py` should import these existing tests (or extend the existing `test_eeff.py`) rather than re-implement them.

6. **Observe telemetry embed.** Follow `run-evaluation` SKILL.md Step 2b pattern exactly (lines 172-177): spawn a background telemetry agent with `run_in_background=true`. The describe/prompt template adapts the output paths to `output/profiling/reasoning-merged-v4/`.

7. **Container vs host execution.** `profile_merged_model.py` runs INSIDE the DGX `ngc-pytorch` container (CUDA required). The run-profiling skill orchestration layer (`.complete` checks, monitor, reporting) runs on HOST. This mirrors the eval flow (eval harness inside container; orchestrator on host) but the split is at a different step — profiling has no vLLM intermediary.

---

## Metadata

**Analog search scope:** `scripts/`, `tests/`, `eval/`, `.claude/skills/`, `.planning/phases/04.4-*/`
**Files scanned:** ~15 source files read in full or partial
**Pattern extraction date:** 2026-06-14
