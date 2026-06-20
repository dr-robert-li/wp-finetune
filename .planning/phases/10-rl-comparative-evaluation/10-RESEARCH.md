# Phase 10: RL Comparative Evaluation - Research

**Researched:** 2026-06-21
**Domain:** RL checkpoint evaluation, comparative eval harness extension, CI-aware gate wiring
**Confidence:** HIGH (all key claims verified against on-disk code and artifacts)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-10-01 — CI-aware per-dimension regression gate**
No dimension regression = no statistically real regression under the project CI-aware noise-band disposition (D-09, inherited). A dimension counts as regressed only if its bootstrap CI shows a real drop below the v1.2 baseline, measured identically on baseline + RL candidate; within-noise dips PASS. Judge Spearman must improve beyond noise (the primary RL target) — a flat/within-noise judge is a soft fail to surface, not a silent pass. Per-dimension (not aggregate-only) so a real gen-dim drop cannot hide behind a judge gain.

**D-10-02 — Checkpoint selection: best-by-reward + final, head-to-head**
Export and evaluate two RL checkpoints — the best-by-reward-convergence checkpoint AND the final-step checkpoint — run both through the full eval, pick the winner against the baseline. The selected winner is the canonical RL model handed to Phase 11.

**D-10-03 — wp-bench hard gate: aggregate CI-aware + per-task floor**
The wp-bench HARD GATE passes when the RL model's aggregate wp-bench bootstrap lower bound >= the v1.2 baseline aggregate point estimate (0.4616 from `output/04.4_wp_bench_results.json`), AND no individual wp-bench task catastrophically regresses (a hard per-task floor). Concrete floor value is Claude's Discretion — derived below. [CORRECTED per plan-checker D-10-03 BLOCKER fix: 0.4616 is wp-bench's DETERMINISTIC WEIGHTED overall (metadata.scores.overall), not a per-task mean, so the gate is a DIRECT point comparison candidate_overall >= 0.4616 — NOT a flat-array bootstrap. A simple-mean bootstrap vs a weighted baseline is apples-to-oranges. Per-dim + jaccard gates keep bootstrap_ci; sub-type floors unchanged.]

**D-10-04 — RLEV-02 five-part conjunctive gate + human sign-off**
Sign-off to v3.0 (gating MoE-Sieve) requires ALL of:
1. Judge Spearman improvement beyond noise (primary RL target)
2. wp-bench HARD GATE pass (D-10-03)
3. Anti-hack pass rate >= Phase 8 anti-hack baseline (CI-aware: hi_perturbed < lo_clean)
4. Protected-expert retention >= Phase 7 baseline (CI-aware vs `protected_expert_mask`)
5. Router-shift / KL stability log shows no routing collapse over the run

A human review checkpoint presents the full v1.2-SFT-vs-RL comparison table before the v3.0 gate is declared pass/fail.

### Claude's Discretion
- Exact bootstrap method, N resamples, and CI level (reuse the project's existing CI-aware gate implementation in `eval/eval_gate.py` — do not invent a new one)
  > **Research annotation:** `eval/eval_gate.py` is a point-threshold comparator (no bootstrap). The actual `bootstrap_ci` function lives in `scripts/compute_concentration.py` (N=1000, alpha=0.05). The discretion is to reuse that function — not `eval_gate.py`'s threshold logic — for D-10-01/03 bounds. See Eval Infra Inventory.
- Concrete numeric value of per-task "catastrophic regression" floor (D-10-03) — derived below
- Eval-harness wiring, serving venue plumbing, report layout/format, telemetry embedding
- Whether the judge-recalibration +3.58 offset (D-V4-09) applies identically to the RL judge component — confirmed below

### Deferred Ideas (OUT OF SCOPE)
- Fresh RL-policy routing re-profiling for sieve selection (Phase 11)
- MoE-Sieve / merge / pruning (Phases 11/13)
- Auto-retrain on regression (surface + suggest fix; do not loop)
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RLEV-01 | RL model evaluated against v1.2 SFT baseline on wp-bench and all 9 eval dimensions — no dimension regression permitted; judge Spearman improvement expected (primary RL target) | Eval infra inventory (eval_gen.py + eval_judge.py + eval_gate.py), bootstrap_ci wiring, D-10-01/03 gate derivation, serving pipeline correction |
| RLEV-02 | RL evaluation report includes reward metric convergence curves, router-shift stability log (per-step shift ratios), protected expert retention rate vs Phase 7 baseline, gen/judge quality delta, and anti-hack eval results | rl_metrics.jsonl field names confirmed, anti-hack acceptance_report.json schema, protected_expert_mask location, checkpoint_manifest.json structure |
</phase_requirements>

---

## Summary

Phase 10 re-runs the existing `eval/` 9-dim + wp-bench harness on the Phase 9 GSPO RL checkpoint(s) and diffs against the v1.2 SFT baseline (`output/merge_v4_winner`). It is an evaluation-only phase — no retraining. The infrastructure is largely complete: eval_gen.py, eval_judge.py, eval_gate.py, rubric_scorer.py, and the wp-bench Docker grader are all operational from Phase 4.4. The plan needs to: (1) add a bootstrap CI wrapper (`scripts/bootstrap_gate.py`) because eval_gate.py is a point-threshold comparator with no bootstrap logic, (2) implement four RL-specific report sections that pull from `rl_metrics.jsonl`, and (3) wire the Tinker LoRA export through merge + vLLM serve before eval can run.

The primary blocker is Phase 9's live Tinker RL run (credential-gated). All wiring code, fixture-backed tests, and gate/report scripts can be written and validated before the run completes; only the final comparison numbers are blocked.

The ROADMAP contains two stale references that the plan must correct: (a) `dgx.execute("eval_toolbox", ...)` — the actual pattern is vLLM served directly in Docker on DGX, accessed via `http://localhost:8020/v1`; (b) `Agent(run_in_background=true)` as Python-side judge dispatch — the actual pattern is `scripts/claude_agent.py` subprocess (`claude --print --no-session-persistence --tools "" --model <model>`). The `Agent(run_in_background=true)` construct exists only at the skill-orchestrator layer, not inside Python evaluation scripts.

**Primary recommendation:** Write a single 3-wave plan. Wave 0: bootstrap_gate.py + RLEV-02 report skeleton + fixture harness. Wave 1: RL checkpoint merge + serve + eval run (blocked on live Tinker run). Wave 2: gate evaluation, five-part conjunctive check, human sign-off table.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Tinker LoRA export + merge | Build/CI (Tinker cloud) | DGX host (merge script) | RL training is Tinker-native; merge runs on DGX before serving |
| vLLM model serving | DGX container | Host orchestrator | Same pattern as Phase 4.4; Docker container exposes :8020 |
| Generation eval (9 dims) | DGX host (eval_gen.py) | vLLM endpoint | eval_gen.py is an HTTP client to vLLM; rubric_scorer runs locally |
| Judge eval (Spearman) | DGX host (eval_judge.py) | claude_agent.py subprocess | eval_judge.py calls claude CLI subprocess, not Anthropic API |
| Bootstrap CI gate | DGX host (bootstrap_gate.py) | — | New script; wraps bootstrap_ci from compute_concentration.py |
| wp-bench scoring | DGX host (wp-bench harness) | Docker grader container | wp-bench runner orchestrates Docker grader for execution tasks |
| RL metrics / report | DGX host (report script) | rl_metrics.jsonl (Tinker output) | Read from file; no live Tinker connection needed post-export |
| Anti-hack eval | DGX host | vLLM endpoint + eval_judge.py | Re-run Phase 8 antihack JSONLs through RL model judge |
| Human sign-off table | Human + report script | — | Gate is conjunctive; human approval is the final gate |

---

## Standard Stack

### Core (all already installed / operational)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `eval/eval_gen.py` | project-local | 9-dim generation scoring via rubric_scorer | Operational since Phase 4.4 |
| `eval/eval_judge.py` | project-local | Judge Spearman eval, gt_mode=calibrated_canonical | Operational since Phase 4.4; RC-A fix in place |
| `eval/eval_gate.py` | project-local | Point-threshold gate logic | Operational; does NOT contain bootstrap CI |
| `scripts/compute_concentration.py` | project-local | `bootstrap_ci(values, n_boot=1000, alpha=0.05)` | The authoritative bootstrap function; used by Phase 7 + Phase 8 |
| `scripts/claude_agent.py` | project-local | `generate(prompt, system, model, timeout, max_retries)` → str | Subprocess dispatch to `claude` CLI; used in eval_judge.py |
| `scripts/build_antihack_set.py` | project-local | Anti-hack CI gate logic | Phase 8 gate: `hi_perturbed < lo_clean` using bootstrap_ci |
| vLLM | Docker-served | Model serving for eval | Phase 4.4 re-bench pattern; Docker image on DGX |
| wp-bench | `wp-bench/` submodule | Hard gate scoring | Docker grader at `ghcr.io/wordpress/wp-bench-grader:latest` |

### New (to be created in Wave 0)

| Script | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `scripts/bootstrap_gate.py` | CI-aware dim regression gate + aggregate gate | `eval_gen_results.json`, `eval_judge_results.json`, baseline values | `bootstrap_gate_result.json` per dim, aggregate CI lower bound |
| `scripts/rlev02_report.py` | RLEV-02 four-section report | `rl_metrics.jsonl`, `checkpoint_manifest.json`, `antihack_result.json`, `jaccard_stability.json` | `output/rl_eval/rlev02_report.json` |

**Version verification:** All listed project-local scripts confirmed present on disk via `ls` + `Read`. [VERIFIED: on-disk]

---

## Package Legitimacy Audit

No new external packages are introduced in this phase. All tooling reuses existing project dependencies. Phase 10 installs nothing new.

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

---

## Architecture Patterns

### Corrected Pipeline (ROADMAP vs Reality)

```
ROADMAP (STALE):           dgx.execute("eval_toolbox", ...) + Agent(run_in_background=true)
ACTUAL PIPELINE:

[Tinker cloud]
  └── rl_train.py --live-run  →  checkpoint_manifest.json + rl_metrics.jsonl
  └── export tinker:// → HF archive  →  output/rl_checkpoints/{best,final}/

[DGX host — orchestrator runs from host]
  └── scripts/merge_tinker_v3.py  →  output/rl_merged_{best,final}/
  └── docker run vllm --model output/rl_merged_{best}/  →  localhost:8020
  └── eval/eval_gen.py  →  output/rl_eval/{best}/eval_gen_results.json
  └── eval/eval_judge.py (gt_mode=calibrated_canonical)  →  eval_judge_results.json
       └── LLM dispatch: scripts/claude_agent.py subprocess
           (claude --print --no-session-persistence --tools "" --model <model>)
  └── scripts/bootstrap_gate.py  →  dim regression check vs v1.2 baseline
  └── wp-bench run (Docker grader)  →  wp_bench_results.json + per-task JSONL
  └── bootstrap lower bound check vs 0.4616 baseline point estimate  # CORRECTED: direct weighted-overall point comparison (metadata.scores.overall >= 0.4616), not a per-task bootstrap
  └── scripts/rlev02_report.py  →  rlev02_report.json + human sign-off table
```

**Critical correction — serving:** The RL model is a Tinker LoRA (rank 32, not bf16-merged in the same pipeline as v1.2). vLLM cannot serve a raw LoRA for this model. The RL checkpoint must be merged (LoRA → full weights) before vLLM can serve it. Reuse `scripts/merge_tinker_v3.py` MoE-only path. This merge step is MANDATORY and must be Wave 1 Task 1 before any eval can run. [VERIFIED: on-disk — `ls scripts/merge*` confirmed `scripts/merge_tinker_v3.py` exists; docstring confirms Tinker MoE LoRA tensor convention from `checkpoint.tar` format]

### Recommended Output Structure
```
output/
└── rl_eval/
    ├── best_checkpoint/
    │   ├── eval_gen_results.json
    │   ├── eval_gen_results.jsonl      # per-example
    │   ├── eval_judge_results.json
    │   ├── wp_bench_results.json
    │   └── bootstrap_gate_result.json
    ├── final_checkpoint/
    │   └── (same structure)
    ├── winner/                          # symlink/copy of winning checkpoint
    └── rlev02_report.json              # four-section conjunctive gate report
```

### Pattern 1: bootstrap_gate.py — CI-Aware Dimension Regression

```python
# Source: scripts/compute_concentration.py (bootstrap_ci) + scripts/build_antihack_set.py
from scripts.compute_concentration import bootstrap_ci
import numpy as np

def check_dim_regression(
    baseline_scores: list[float],   # per-example dim scores from baseline eval_gen JSONL
    candidate_scores: list[float],  # per-example dim scores from RL candidate eval_gen JSONL
    dim_key: str,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> dict:
    """
    Gate passes if candidate bootstrap CI lower bound >= baseline mean.
    Uses same bootstrap_ci(values, n_boot, alpha) as Phase 7 + Phase 8.
    """
    lo_cand, hi_cand = bootstrap_ci(np.array(candidate_scores), n_boot=n_boot, alpha=alpha)
    baseline_mean = np.mean(baseline_scores)
    passed = lo_cand >= baseline_mean  # no statistically real regression
    return {
        "dim": dim_key,
        "baseline_mean": baseline_mean,
        "candidate_ci_lower": lo_cand,
        "candidate_ci_upper": hi_cand,
        "passed": passed,
    }
```

[VERIFIED: on-disk — `bootstrap_ci` signature confirmed in `scripts/compute_concentration.py` lines 42-68: `bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]`]
### Pattern 1b: bootstrap_gate.py — Judge Spearman Improvement (DISTINCT from mean-bootstrap)

**Critical:** `bootstrap_ci(values)` computes the CI of the **mean of a 1-D array**. You cannot feed it a Spearman correlation directly. The judge-Spearman gate (D-10-01 judge improvement, D-10-04 #1) requires resampling `(pred, GT)` score *pairs* and recomputing `spearmanr` on each resample.

```python
# Source: Pattern derived from scipy.stats.spearmanr + bootstrap pairing
# DIFFERENT from bootstrap_ci — handles correlation, not mean
from scipy.stats import spearmanr
import numpy as np

def bootstrap_spearman_improvement(
    baseline_pairs: list[tuple[float, float]],   # (model_score, gt_score) for v1.2 SFT
    candidate_pairs: list[tuple[float, float]],  # (model_score, gt_score) for RL model
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> dict:
    """
    Compute bootstrap CI of Spearman correlation difference (rho_RL - rho_v12).
    Both models must be scored on the same val set (shared GT pairs).
    Improvement is beyond noise when CI lower bound of difference > 0.
    """
    assert len(baseline_pairs) == len(candidate_pairs), "Must score same examples"
    n = len(baseline_pairs)
    rng = np.random.default_rng()
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bl_pred = [baseline_pairs[j][0] for j in idx]
        bl_gt   = [baseline_pairs[j][1] for j in idx]
        cand_pred = [candidate_pairs[j][0] for j in idx]
        cand_gt   = [candidate_pairs[j][1] for j in idx]
        rho_bl   = spearmanr(bl_pred, bl_gt).statistic  # .statistic per project convention (eval_judge.py::_safe_spearman); .correlation is deprecated in modern scipy
        rho_cand = spearmanr(cand_pred, cand_gt).statistic  # .statistic per project convention (eval_judge.py::_safe_spearman); .correlation is deprecated in modern scipy
        diffs[i] = rho_cand - rho_bl
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    rho_bl_point   = spearmanr([p[0] for p in baseline_pairs], [p[1] for p in baseline_pairs]).statistic  # .statistic per project convention (eval_judge.py::_safe_spearman); .correlation is deprecated in modern scipy
    rho_cand_point = spearmanr([p[0] for p in candidate_pairs], [p[1] for p in candidate_pairs]).statistic  # .statistic per project convention (eval_judge.py::_safe_spearman); .correlation is deprecated in modern scipy
    return {
        "rho_baseline": rho_bl_point,
        "rho_candidate": rho_cand_point,
        "diff_point": rho_cand_point - rho_bl_point,
        "diff_ci_lower": lo,
        "diff_ci_upper": hi,
        "improved_beyond_noise": lo > 0,
    }
```

**Per-dimension Spearman check:** Same pattern per dim, but with smaller `n_pairs` -> wider CIs. A per-dim judge correlation that is flat or within noise is a soft fail (surface to user); a statistically real *regression* in judge dim correlation hard-fails.

**Input sourcing:** Both baseline_pairs and candidate_pairs come from paired `(judge_output_score, gt_score)` records from `eval_judge.py` run on the SAME val set. The val set, GT mode, and grading prompt must be identical across both runs.


### Pattern 2: eval_judge.py — Two-GT Modes

```python
# Source: eval/eval_judge.py (verified on-disk)
# REVL-01A (primary): gt_mode="calibrated_canonical"
result = run_eval(
    dataset_path="data/reasoning_dataset/openai_val.jsonl",
    limit=None,
    output_path="output/rl_eval/best_checkpoint/eval_judge_results.json",
    model="openai/qwen3-wp-rl",
    base_url="http://localhost:8020/v1",
    output_format="jsonl",
    gt_mode="calibrated_canonical",   # HARD: vs calibrated GT scores
    responses_jsonl=None,             # live endpoint (not offline)
)
# result["overall_spearman"] -> {corr, p_value, n_pairs}
# primary check: result["overall_spearman"]["corr"] > v1.2 SFT Spearman (beyond noise)
```

**RC-A fix is in place:** `_judge_create()` in eval_judge.py passes `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` — prevents unclosed `<think>` blocks from causing parse failures. [VERIFIED: on-disk, confirmed in prior session]

### Pattern 3: wp-bench Run

```python
# config/wp-bench.yaml (verified on-disk)
# models: api_base: "http://localhost:8020/v1" (overridden at runtime)
# output: output/wp-bench-results.json + output/wp-bench-results.jsonl (per-task)
# grader: kind: docker, wp_env_dir: ./wp-bench/runtime
# suite: wp-core-v1, 344 tasks (320 knowledge + 24 execution)
```

### Anti-Patterns to Avoid

- **Using `Agent(run_in_background=true)` in Python eval scripts:** This is a skill-orchestrator pattern. Python-side LLM dispatch uses `scripts/claude_agent.py` subprocess. Mixing them causes silent failures.
- **Serving raw LoRA via vLLM:** vLLM requires a merged model. Attempting to serve the Tinker LoRA directly will fail. Merge first.
- **Using point-threshold eval_gate.py for CI-aware D-10-01/03:** eval_gate.py's `run_gate()` is a simple `actual >= target` comparator. It has no bootstrap CI. Treat it as a threshold helper only; implement CI logic in bootstrap_gate.py.
- **Applying +3.58 offset to Spearman gate:** The offset is rank-invariant. The primary judge gate (D-10-01 Spearman) is unaffected. Only absolute-threshold score gates need offset consideration.
- **Trusting `checkpoint_manifest.json` for live checkpoint paths before run completes:** Current manifest only has `dry-run-step-0`. Live paths only exist after the Tinker run completes.

---

## Eval Infra Inventory

### `eval/eval_gen.py`

**Function:** `run_eval(dataset_path, limit, output_path, model, base_url) -> dict`

**Output schema (confirmed):**
```json
{
  "total": int,
  "overall_mean": float,
  "overall_median": float,
  "grade_distribution": {...},
  "per_dimension": {
    "D1_wpcs": {"mean": float, "pass_rate_8": float, "pass_rate_8_inclusive": float, "na_count": int, "na_rate": float},
    "D2_security": {...},
    ...
  },
  "floor_rules": {...},
  "n_applicable_dims_mean": int,
  "phpcs_pass_rate": float,
  "security_pass_rate": float
}
```

**Per-example JSONL:** `eval_gen_results.jsonl` with `{example_idx, prompt, response, extracted_code, overall, grade, dimension_scores, dimension_na, floor_rules_applied, triggered_checks}`

**Note:** eval_gen does NOT use enable_thinking guard (generation mode; thinking stripped in `_extract_php_code`).

### `eval/eval_judge.py`

**Function:** `run_eval(dataset_path, limit, output_path, model, base_url, output_format, gt_mode, responses_jsonl) -> dict`

**Two modes:**
- `gt_mode="dataset"` — legacy, uses dataset assistant-target as GT (REVL-01B SOFT)
- `gt_mode="calibrated_canonical"` — uses calibrated GT scores (REVL-01A HARD, primary for Phase 10)

**Output schema:**
```json
{
  "overall_spearman": {"corr": float, "p_value": float, "n_pairs": int},
  "per_dimension": {
    "D1_wpcs": {"corr": float, "p_value": float, "n_pairs": int},
    ...all 9 dims...
  },
  "skipped": int,
  "total": int,
  "spearman_corr": float,
  "p_value": float,
  "total_pairs": int
}
```

**Offline mode:** `responses_jsonl` param allows pre-captured responses (Tinker-captured) instead of live endpoint.

**RC-A fix in place:** `_judge_create()` passes `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.

### `eval/eval_gate.py`

**Function:** `run_gate(results_dir, config_path) -> (passed: bool, gate_rows: list[dict])`

**What it does:** Simple threshold comparator — `actual >= target`. Each gate_row: `{gate: str, target: float, actual: float, passed: bool}`.

**What it does NOT do:** No bootstrap CI, no resampling, no lower bound logic.

**Fields read from gen results:** `overall_mean`, `per_dimension[dim]["pass_rate_8"]`, `phpcs_pass_rate`, `security_pass_rate`

**Fields read from judge results:** `overall_spearman` (dict → extracts `.corr`), `per_dimension[dim]["corr"]`, `spearman_corr`

**Phase 10 use:** Use `check_gates()` as a helper for point-estimate thresholds only. D-10-01/03 CI logic must be in `scripts/bootstrap_gate.py`.

### `scripts/compute_concentration.py` — `bootstrap_ci`

**Canonical bootstrap function for the project:**
```python
def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    # Uses np.random.default_rng() + np.empty + resample loop
    # Returns (lo, hi) = (alpha/2 percentile, 1-alpha/2 percentile) of boot_means
```
[VERIFIED: on-disk — lines 42-75 of `scripts/compute_concentration.py`]

**All CI-aware gates in the project use this function:**
- Phase 7 PROF-03 Jaccard gate: `extract_protected_mask.py` → `bootstrap_ci(mask_sizes)`
- Phase 8 anti-hack gate: `build_antihack_set.py` → `bootstrap_ci(perturbed_rewards)`, `bootstrap_ci(clean_rewards)`, gate = `hi_perturbed < lo_clean`
- Phase 10 (new): `bootstrap_gate.py` → same function, `lo_candidate >= baseline_mean`

### `eval/rubric_definitions.py` — Method Classification

**193 total checks.** Method breakdown by dimension:
[VERIFIED: on-disk via `python3 -c "from eval.rubric_definitions import CHECK_REGISTRY; ..."`]

| Dimension | phpcs | regex | regex+ | llm | phpstan | file | Total LLM checks |
|-----------|-------|-------|--------|-----|---------|------|-----------------|
| D1_wpcs | 26 | 5 | 1 | 0 | 0 | 0 | 0 |
| D2_security | 13 | 10 | 1 | 9 | 0 | 0 | 9 |
| D3_sql | 8 | 13 | 2 | 3 | 0 | 0 | 3 |
| D4_perf | 3 | 15 | 1 | 6 | 0 | 0 | 6 |
| D5_wp_api | 15 | 9 | 0 | 3 | 0 | 0 | 3 |
| D6_i18n | 3 | 17 | 0 | 3 | 0 | 0 | 3 |
| D7_a11y | 0 | 19 | 0 | 6 | 0 | 0 | 6 |
| D8_errors | 2 | 13 | 0 | 4 | 4 | 0 | 4 |
| D9_structure | 2 | 16 | 0 | 7 | 0 | 2 | 7 |
| **Total** | 72 | 117 | 5 | **41** | 4 | 2 | **41** |

**Dimension classification for eval purposes:**

- **D1_wpcs:** Fully deterministic (phpcs + regex only). No LLM calls.
- **D2_security through D9_structure:** Hybrid — rubric_scorer runs ALL checks (deterministic + LLM sub-checks batched via `llm_checks.py`). The LLM sub-checks use `llm_checks.py`'s backend (env-selectable: `claude` subprocess or local vLLM at `:8000`). These are RUBRIC scoring calls, not "judge eval" calls.
- **Judge eval (eval_judge.py):** Separate from rubric scoring. Tests whether the model's own numeric judge outputs match calibrated GT scores. Uses `scripts/claude_agent.py` subprocess.

**Key distinction:** The 9 dims in eval_gen.py are ALL scored deterministically by rubric_scorer (which internally may call llm_checks.py for its 41 LLM sub-checks, but this is transparent to eval_gen). The "LLM-judged" layer is only the Spearman correlation eval in eval_judge.py.

### `eval/llm_checks.py`

**Function:** `run_llm_checks(code: str) -> dict`

Batched YES/NO inference: all 41 LLM checks evaluated in a single call.

**Backend selection:** `LLM_BACKEND` env var:
- `"claude"` (default): Claude Code agent via `claude` CLI subprocess (subscription, $0)
- `"vllm"`: local vLLM at `LLM_VLLM_BASE_URL` (default `http://localhost:8000/v1`)

**Important:** At eval scale, `LLM_BACKEND=vllm` is strongly preferred (Phase 1 used it for 20K+ samples). At Phase 10 eval scale (~few hundred examples), `claude` backend is viable but slower.

### wp-bench Schema (verified on-disk)

**Config:** `config/wp-bench.yaml`
```yaml
dataset: {source: local, name: wp-core-v1}
models: [{name: openai/qwen3-wp, api_base: http://localhost:8020/v1}]
grader: {kind: docker, wp_env_dir: ./wp-bench/runtime}
run: {suite: wp-core-v1, limit: null, concurrency: 4}
output: {path: output/wp-bench-results.json, jsonl_path: output/wp-bench-results.jsonl}
```

**Result schema (344 tasks):**
```json
{
  "metadata": {
    "suite": "wp-core-v1",
    "scores": {"knowledge": float, "correctness": float, "quality": null, "overall": float}
  },
  "results": [
    {"test_id": "k-abilities-004", "type": "knowledge", "prompt_hash": "...", "answer": "C", "correct": false, "score": 0.0},
    {"test_id": "c-...", "type": "execution", "code": "...", "result": "...", "correctness": bool, "quality": null}
  ]
}
```

**Task types and schemas:**
- `knowledge` (320 tasks): `{test_id, type, prompt_hash, answer, correct, score}` — score is 0.0 or 1.0
- `execution` (24 tasks): `{test_id, type, prompt_hash, code, result, stdout, stderr, correctness, quality}` — no `score` field; correctness is bool

**Aggregate scoring formula:** wp-bench computes `knowledge=0.4906`, `correctness=0.4375`, `overall=0.4603` (per detailed re-bench metadata) using its own weighted formula (not a simple mean of per-task scores). The D-10-03 baseline is the **aggregate `overall` field** from `output/04.4_wp_bench_results.json` = **0.4616** (reasoning_score) — use this value per CONTEXT.md, noting the 0.0013 discrepancy vs re-bench metadata.

---

## Artifact Schemas

### `output/04.4_wp_bench_results.json` — D-10-03 Baseline
[VERIFIED: on-disk]
```json
{
  "gate": "REVL-04",
  "baseline_score": 0.4286,
  "reasoning_score": 0.4616,
  "meets_baseline": true,
  "pass": true
}
```

**D-10-03 baseline value = `reasoning_score` = 0.4616** (the v1.2 SFT score). `baseline_score` = 0.4286 is the pre-SFT baseline — do NOT use this as the comparison target.

**Source discrepancy note:** `output/04.4_wp_bench_results.json` reports `reasoning_score=0.4616`. The detailed summary at `output/eval_reasoning_v4_winner/revl04_rebench/summary.json` and the re-bench result metadata both report `overall=0.4603`. Difference = 0.0013. CONTEXT.md explicitly names `output/04.4_wp_bench_results.json` as the D-10-03 reference — use **0.4616**. Planner should note this discrepancy; the 0.0013 gap may reflect rounding or a version difference in the wp-bench grader.

The matching detailed file is `output/eval_reasoning_v4_winner/revl04_rebench/reasoning_merged/wp_bench_results_20260613_214919.json` which contains all 344 per-task results used to derive the catastrophic-regression floor.

### `output/eval_reasoning_v4_winner/judge_recalibration.json`
[VERIFIED: on-disk]
```json
{
  "score_offset": 3.58,
  "applies_to": "wp_judge overall score",
  "rank_invariant": true,
  "ci_95": [1.24, 6.09],
  "se": 1.25,
  "n": 118,
  "method": "paired mean (sampler - merged) over n=118 common val pairs; bf16-merge inference-path calibration artifact",
  "note": "Does NOT affect rank metrics (REVL-01A Spearman). Under MO-GRPO within-group normalization a uniform offset largely cancels; apply for absolute-threshold consistency.",
  "recalibrated_threshold_equivalent": 66.42
}
```

### `output/rl_checkpoints/checkpoint_manifest.json`
[VERIFIED: on-disk — only dry-run data]
```json
{
  "checkpoints": [
    {"name": "dry-run-step-0", "sampler_path": "/dry-run/sampler", "saved_at": "2026-06-20T10:17:38Z"}
  ],
  "run_args": {
    "model_id": "Qwen/Qwen3-30B", "lora_rank": 32, "use_gspo": true,
    "dry_run": true, "total_steps": 1, "checkpoint_every": 50,
    "jaccard_every": 20, "kl_soft": 0.1, "kl_hard": 0.3,
    "efrac_soft": 0.7, "efrac_hard": 0.5,
    "mask_path": "output/profiling/reasoning-merged-v4/protected_expert_mask.npy"
  }
}
```

**After live run:** checkpoints at every 50 steps. D-10-02 best-by-reward checkpoint = max `reward_mean` across all steps; final checkpoint = last entry.

### `output/rl_checkpoints/metrics/rl_metrics.jsonl`
[VERIFIED: on-disk — dry-run step-0 only]

Per-step fields:
```
step, reward_mean, reward_breakdown: {n_samples, reward_min, reward_max,
  <wp_gen>: {mean, min, max}, <wp_judge>: {mean, min, max}},
kl_sample_train_v1, kl_sample_train_v2,
e_frac_with_tokens_mean, e_max_violation_mean, e_max_violation_max,
jaccard_protected, halt_reason, use_gspo, model_id, ts
```

**RLEV-02 report fields consumed from this file:**
- `reward_mean` over steps → convergence curve
- `reward_breakdown.<wp_gen>.mean` + `reward_breakdown.<wp_judge>.mean` → dual-mode breakdown
- `kl_sample_train_v1` + `e_frac_with_tokens_mean` → router-shift / stability log
- `jaccard_protected` per step → protected-expert retention monitor

**Note:** `reward_breakdown.<wp_gen>/<wp_judge>` are PENDING confirmation via Phase 9 UAT test #3. Dry-run has sparse breakdown. Plan should use fixture/placeholder for Wave 0, confirmed on live data in Wave 1. [ASSUMED for sub-field structure; top-level fields [VERIFIED: on-disk]]

### `output/antihack_validation/acceptance_report.json` — Phase 8 Anti-Hack Baseline
[VERIFIED: on-disk]
```json
{
  "report_type": "fixture_backed",
  "fixture_note": "Synthetic reward arrays to prove CI gate logic. Live scoring is follow-up.",
  "all_axes_pass": true,
  "gate_criterion": "hi_perturbed < lo_clean (D-09 CI-aware)",
  "n_boot": 1000,
  "axes": {
    "verbose_padding":            {"lo_perturbed": 0.332, "hi_perturbed": 0.378, "lo_clean": 0.666, "hi_clean": 0.710, "perturbed_mean": 0.356, "clean_mean": 0.689, "n_perturbed": 15, "n_clean": 15},
    "template_critique_collapse": {"lo_perturbed": 0.282, "hi_perturbed": 0.326, "lo_clean": 0.685, "hi_clean": 0.746, "perturbed_mean": 0.304, "clean_mean": 0.716},
    "self_preference_swap":       {"lo_perturbed": 0.324, "hi_perturbed": 0.363, "lo_clean": 0.646, "hi_clean": 0.689, "perturbed_mean": 0.344, "clean_mean": 0.667}
  }
}
```

**Critical note:** `report_type: fixture_backed` — these are SYNTHETIC baseline values from Wave 0 scaffolding. The "Known Follow-Up" from `08-04-SUMMARY.md` states live scoring against vLLM judge endpoint is needed. **The D-10-04 #3 anti-hack gate must re-run Phase 8's `build_antihack_set.py` anti-hack JSONLs through the RL model's judge endpoint** and compare the RL anti-hack CI result vs the v1.2 SFT anti-hack CI result (not the fixture values). The fixture report is only useful as a structural reference for the gate schema.

**Anti-hack set files (real data):**
- `output/antihack_validation/antihack_verbose_padding.jsonl` — 50.4K
- `output/antihack_validation/antihack_template_critique_collapse.jsonl` — 48.5K
- `output/antihack_validation/antihack_self_preference_swap.jsonl` — 48.3K

45 cases total (15 per axis), each with clean + perturbed variants.

### Phase 7 Protected-Expert Baseline
[VERIFIED: on-disk]
- **Mask:** `output/profiling/reasoning-merged-v4/protected_expert_mask.npy` — [48, 128] bool, 1,480 experts
- **Stability gate:** `output/profiling/reasoning-merged-v4/concentration_report.json` — `jaccard_ci_lower=0.9426` (bootstrap CI lower bound of per-layer Jaccard mean across 48 layers; PROF-03 gate; from `jaccard_stability.json` + `compute_concentration.py`)
- **Per-step monitor:** `jaccard_protected` in `rl_metrics.jsonl` — Jaccard of active experts vs this mask at each RL step

**D-10-04 #4 referent mismatch — see Open Questions:** `jaccard_ci_lower=0.9426` is the CI lower bound of *cross-run per-layer Jaccard stability* from Phase 7 SFT profiling (48-layer Jaccards resampled). The RL signal `jaccard_protected` is *per-step fraction of protected experts that are active during RL training*. These are different quantities. No pre-existing "RL retention" bar exists. See Open Questions #4 for the recommended approach.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Bootstrap CI | Custom resampling | `scripts/compute_concentration.py::bootstrap_ci` | Same function used in Phase 7 + Phase 8; N=1000, alpha=0.05, 95% CI |
| Anti-hack gate shape | Custom adversarial logic | `scripts/build_antihack_set.py::compute_axis_gate` | Phase 8 gate: `hi_perturbed < lo_clean`, already debugged |
| vLLM serving for merged MoE | Ad-hoc docker commands | Phase 4.4 re-bench serving pattern (SKILL.md) | Concurrency, timeout, health-check patterns already established |
| LoRA merge for vLLM | Ad-hoc weight addition | `scripts/merge_tinker_v3.py` MoE-only path | Handles MoE architecture correctly; Phase 4.3 proved it |
| LLM judge dispatch | Anthropic API calls | `scripts/claude_agent.py` subprocess | Uses Claude subscription quota; RC-A enable_thinking fix already in place |
| Per-example Spearman | Manual rank computation | `eval/eval_judge.py::run_eval` with `gt_mode="calibrated_canonical"` | Two-GT logic, parse strategies, and retry already implemented |
| Spearman improvement CI | Applying `bootstrap_ci(corr_array)` | Pattern 1b: resample `(pred, GT)` pairs, recompute `spearmanr` per resample | Correlations not additive means; pair-level bootstrap is the correct approach |

**Key insight:** All custom CI / gate / serving / merge / dispatch infrastructure already exists and has been battle-tested through Phases 4–9. Phase 10 extends, never rebuilds.

---

## +3.58 Judge-Recalibration Offset — Phase 10 Application

[VERIFIED: `output/eval_reasoning_v4_winner/judge_recalibration.json`, `rank_invariant: true`]

**Primary gate (D-10-01 Spearman):** The +3.58 offset is rank-invariant. Spearman correlation is computed on ranks, not absolute scores. The offset DOES NOT AFFECT the primary judge improvement check. No action needed.

**Asymmetry consideration [ASSUMED]:** The offset was derived as `mean(sampler_score - merged_score)` over n=118 pairs for the v1.2 bf16-merge inference path. The RL model is a Tinker LoRA merged via a different path. The calibration offset for the RL model may differ. Since:
1. The primary gate uses Spearman (rank-invariant), offset is moot
2. MO-GRPO within-group normalization during training caused uniform offsets to largely cancel (per the json note)
3. Absolute-threshold judge gates are advisory, not primary

**Recommendation (Claude's Discretion):** Do NOT apply the +3.58 offset to RL judge absolute scores without re-deriving it for the RL model. For Phase 10, rely exclusively on rank-based metrics (Spearman) for the primary judge gate. If an absolute-score judge gate is needed (e.g., confusion verdict threshold), flag it as requiring RL-path re-calibration. This is safe because the recalibration.json itself says `rank_invariant: true` and the MO-GRPO normalization mitigates uniform offsets.

---

## D-10-03 Per-Task Catastrophic-Regression Floor (Claude's Discretion)

[VERIFIED: on-disk — computed from `output/eval_reasoning_v4_winner/revl04_rebench/reasoning_merged/wp_bench_results_20260613_214919.json`]

**v1.2 SFT wp-bench per-task score statistics (n=344 tasks):**

| Stat | Knowledge (n=320) | Execution (n=24) | Combined |
|------|-------------------|------------------|----------|
| mean | 0.4906 | 0.4375 | ~0.4869 |
| min | 0.0 | 0.0 | 0.0 |
| unique values | {0.0, 1.0} | {0.0, 1.0} | binary |

**Key finding:** wp-bench tasks are **binary scored** (0.0 or 1.0 per task). There is no continuous per-task score to derive a spread from. The aggregate score (0.4616) comes from wp-bench's own weighted formula across knowledge and execution sub-categories.

**Catastrophic-regression floor recommendation:**

Given binary task scores, "catastrophic regression" must be defined at the aggregate-sub-type level, not per individual task. Proposed floor:

- **Knowledge sub-score floor:** RL knowledge sub-score >= 0.45 (vs v1.2 knowledge = 0.4906, floor = ~0.04 below, 1 sigma buffer)
- **Execution sub-score floor:** RL execution sub-score >= 0.375 (vs v1.2 execution = 0.4375, floor = 0.0625 below — only 24 tasks so variance is higher)
- **Overall aggregate WEIGHTED overall (metadata.scores.overall) >= 0.4616 [CORRECTED: direct point comparison, not a per-task bootstrap — see D-10-03 BLOCKER note above]** (D-10-03 primary CI gate, already locked)

These floors serve as catastrophe protection against a model that somehow collapses on a whole sub-type. A model that passes the aggregate CI gate but scores 0.0 on all execution tasks would be caught by the execution floor.

**Alternative (simpler):** Set per-task floor as "RL correct on any task where v1.2 was correct" = RL must not regress to 0.0 on any task where v1.2 scored 1.0. This is verifiable via set-diff of `correct=true` tasks between baseline and candidate. Recommend this as a secondary check (flag for human review but not a hard gate — single-task binary noise is high).

**Final recommendation:** Planner should define D-10-03 per-task floor as: aggregate WEIGHTED overall (metadata.scores.overall) >= 0.4616 (HARD, direct point comparison — NOT a per-task bootstrap; D-10-03 BLOCKER fix), plus sub-type floors (knowledge >= 0.45, execution >= 0.375) as HARD, plus flag-for-human-review if any task regresses from correct to incorrect. This is the most defensible derivation given binary scoring.

---

## Phase 9 Dependency — Real-On-Disk vs Post-Live-Run

[VERIFIED: `output/rl_checkpoints/checkpoint_manifest.json` + `09-HUMAN-UAT.md`]

### On Disk Now (no live run needed)
- `scripts/rl_train.py` — complete, fixed (commit `06dcba7`)
- `output/rl_checkpoints/checkpoint_manifest.json` — exists, dry-run-step-0 only
- `output/rl_checkpoints/metrics/rl_metrics.jsonl` — exists, step-0 dry-run entry only
- `output/profiling/reasoning-merged-v4/protected_expert_mask.npy` — complete, immutable
- `output/antihack_validation/*.jsonl` — 45 anti-hack cases, complete
- `eval/eval_gen.py`, `eval/eval_judge.py`, `eval/eval_gate.py` — operational
- `scripts/claude_agent.py` — operational
- All eval dependencies

### Only After Tinker Live Run Completes
- Real checkpoint files at `output/rl_checkpoints/{best_checkpoint,final_checkpoint}/`
- Real `rl_metrics.jsonl` entries (steps 1..N with full reward_breakdown)
- Real `jaccard_protected` per-step values (current = 0.002 fixture artifact)
- Real `kl_sample_train_v1` convergence trace

### Implications for Planning
- **Wave 0:** All wiring code, bootstrap_gate.py, rlev02_report.py skeleton, fixture-backed tests → can be done NOW
- **Wave 1:** Merge + serve + eval run → BLOCKED on live Tinker run
- **Wave 2:** Gate evaluation, conjunctive check, human sign-off → BLOCKED on Wave 1

Wave 0 tasks must use dry-run fixture data and synthetic baselines for all integration tests.

---

## Common Pitfalls

### Pitfall 1: Confusing the Two Baseline Scores in 04.4_wp_bench_results.json
**What goes wrong:** Using `baseline_score=0.4286` (pre-SFT baseline) instead of `reasoning_score=0.4616` (v1.2 SFT) as the D-10-03 gate bar.
**Why it happens:** The field name `baseline_score` sounds like "the baseline to compare against."
**How to avoid:** D-10-03 explicitly says "v1.2 SFT baseline aggregate point estimate." That is `reasoning_score=0.4616`.
**Warning signs:** If the gate bar is 0.4286 anywhere in the plan, it is wrong.

### Pitfall 2: Serving Raw LoRA via vLLM
**What goes wrong:** Attempting `docker run vllm --model output/rl_checkpoints/best/` without merging first — vLLM errors out or loads base weights only.
**Why it happens:** The checkpoint manifest points to LoRA adapter files, not merged weights.
**How to avoid:** Wave 1 Task 1 must be `merge_tinker_v3.py` MoE-only merge → `output/rl_merged_{best,final}/`. Only then serve via vLLM.
**Warning signs:** vLLM warning about "adapter not applied" or serving responding correctly on base prompts but ignoring WordPress patterns.

### Pitfall 3: Using Phase 8 Fixture Anti-Hack Report as Baseline
**What goes wrong:** Treating `output/antihack_validation/acceptance_report.json`'s `lo_clean=0.666` etc. as the v1.2 SFT baseline for D-10-04 #3.
**Why it happens:** It's labeled `all_axes_pass: true` and has gate criterion.
**How to avoid:** `report_type: fixture_backed` — these are synthetic values. Phase 10 must score the v1.2 model against antihack JSONLs via vLLM, get real CI bounds, then compare RL model CI bounds vs those real v1.2 CI bounds.
**Warning signs:** Any plan that says "compare RL anti-hack to 0.666" is using the fixture number.

### Pitfall 4: Missing eval_gen per-example JSONL for CI Resampling
**What goes wrong:** bootstrap_gate.py tries to resample per-example scores but eval_gen results only has aggregate stats.
**Why it happens:** eval_gen produces both `eval_gen_results.json` (aggregate) and `eval_gen_results.jsonl` (per-example). CI resampling needs the JSONL.
**How to avoid:** bootstrap_gate.py must read `eval_gen_results.jsonl` to get per-example `dimension_scores` arrays for resampling, not the aggregate JSON.
**Warning signs:** bootstrap_gate.py reading only `.json` not `.jsonl`.

### Pitfall 5: +3.58 Offset Applied to Spearman Gate
**What goes wrong:** Adjusting the v1.2 Spearman score by +3.58/100 before comparing to RL model Spearman.
**Why it happens:** The recalibration offset is documented and visible.
**How to avoid:** `rank_invariant: true` — the offset does not affect Spearman. Apply it only if comparing absolute judge scores on a percentage scale, and only after confirming the RL model needs the same offset.

### Pitfall 6: Jaccard Dry-Run Value as Retention Baseline
**What goes wrong:** Using `jaccard_protected=0.002` from the dry-run step-0 as the D-10-04 #4 retention baseline, or mechanically applying 0.9426 (cross-run SFT stability) as the RL retention bar.
**Why it happens:** It's the only value in rl_metrics.jsonl; the 0.9426 number appears in docs.
**How to avoid:** `jaccard_protected=0.002` is a fixture artifact. `jaccard_ci_lower=0.9426` measures SFT profiling stability, not RL retention. D-10-04 #4 needs a separately-defined retention threshold — see Open Questions #4. Do not assert 0.9426 is the bar until the user confirms.

### Pitfall 7: LLM_BACKEND Not Set for rubric_scorer LLM Checks
**What goes wrong:** rubric_scorer's LLM sub-checks default to `claude` backend (slow, serial) when running at full eval scale.
**Why it happens:** Default is `claude` for safety on small smoke runs.
**How to avoid:** Set `LLM_BACKEND=vllm` + `LLM_VLLM_BASE_URL=http://localhost:8000/v1` when running batch eval. This requires a second vLLM instance serving the rubric-scoring model (Qwen3.6-35B-A3B-FP8 or equivalent), separate from the model-under-eval endpoint at :8020.

---

## D-10-04 #5 — "No Routing Collapse" Operational Definition

From `checkpoint_manifest.json` -> `run_args`, hard-halt thresholds are: `kl_hard=0.3`, `efrac_hard=0.5`.

**Gate passes when ALL THREE are clean across the entire run:**

| Signal | Collapse condition | Source field |
|--------|-------------------|--------------|
| Training halt | Any `halt_reason` is set (non-null) on any step | `rl_metrics.jsonl[*].halt_reason` |
| KL divergence breach | Any step has `kl_sample_train_v1 >= 0.3` | `rl_metrics.jsonl[*].kl_sample_train_v1` |
| Expert utilization collapse | Any step has `e_frac_with_tokens_mean < 0.5` | `rl_metrics.jsonl[*].e_frac_with_tokens_mean` |

Soft thresholds (`kl >= 0.1`, `efrac < 0.7`) are advisory signals for the RLEV-02 report but do NOT fail the hard gate.

**Report format for gate #5:**
```json
{
  "gate": "no_routing_collapse",
  "n_steps": int,
  "any_halt": bool,
  "max_kl": float,
  "min_efrac": float,
  "kl_breach_steps": [],
  "efrac_breach_steps": [],
  "soft_kl_steps": [],
  "soft_efrac_steps": [],
  "passed": bool
}
```

[VERIFIED: thresholds from `output/rl_checkpoints/checkpoint_manifest.json` -> `run_args.kl_hard=0.3`, `run_args.efrac_hard=0.5`, `run_args.kl_soft=0.1`, `run_args.efrac_soft=0.7`]

---

## Runtime State Inventory

Not a rename/refactor phase. Omitted.

---

## Validation Architecture

nyquist_validation: true (from `.planning/config.json`)

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (existing, 471 tests confirmed passing in Phase 9) |
| Config file | `pyproject.toml` or `pytest.ini` (existing) |
| Quick run command | `pytest tests/ -x -q --tb=short` |
| Full suite command | `pytest tests/ -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RLEV-01 | bootstrap_gate.py dim regression logic correct (CI lower bound vs baseline mean) | unit | `pytest tests/test_bootstrap_gate.py -x` | Wave 0 gap |
| RLEV-01 | bootstrap_gate.py Spearman improvement check correct (beyond noise) | unit | `pytest tests/test_bootstrap_gate.py::test_spearman_improvement -x` | Wave 0 gap |
| RLEV-01 | eval_gen.py + eval_judge.py produce correct output format against dry-run fixture | integration | `pytest tests/test_eval_integration.py -x` | Likely exists from Phase 4.4 |
| RLEV-01 | wp-bench D-10-03 gate: weighted overall (metadata.scores.overall) >= 0.4616 with fixture data (direct point comparison; includes discriminating case where simple-mean would falsely pass) | unit | `pytest tests/test_bootstrap_gate.py::test_wpbench_gate -x` | Wave 0 gap |
| RLEV-01 | D-10-03 per-task floor: knowledge >= 0.45, execution >= 0.375 sub-type check | unit | `pytest tests/test_bootstrap_gate.py::test_pertask_floor -x` | Wave 0 gap |
| RLEV-02 | rlev02_report.py reads rl_metrics.jsonl correctly (reward_breakdown parsing) | unit | `pytest tests/test_rlev02_report.py -x` | Wave 0 gap |
| RLEV-02 | rlev02_report.py five-part conjunctive gate aggregates correct pass/fail | unit | `pytest tests/test_rlev02_report.py::test_conjunctive_gate -x` | Wave 0 gap |
| RLEV-02 | Anti-hack gate CI comparison: hi_perturbed_rl < lo_clean_v12 | unit | `pytest tests/test_rlev02_report.py::test_antihack_gate -x` | Wave 0 gap |
| RLEV-02 | Jaccard retention: rl_jaccard_ci_lower >= 0.9426 (Phase 7 baseline) | unit | `pytest tests/test_rlev02_report.py::test_jaccard_retention -x` | Wave 0 gap |

### Sampling Rate
- **Per task commit:** `pytest tests/test_bootstrap_gate.py tests/test_rlev02_report.py -x -q`
- **Per wave merge:** `pytest tests/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_bootstrap_gate.py` — covers RLEV-01 dim CI gate, Spearman improvement, wpbench aggregate gate, per-task floor
- [ ] `tests/test_rlev02_report.py` — covers RLEV-02 report parsing, conjunctive gate, anti-hack CI, Jaccard retention
- [ ] `scripts/bootstrap_gate.py` — the CI-aware gate script itself (Wave 0 deliverable)
- [ ] `scripts/rlev02_report.py` — RLEV-02 report generator (Wave 0 deliverable)

*(Existing test infrastructure in `tests/` covers eval_gen, eval_judge, rubric_scorer — those tests are NOT gaps)*

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| vLLM (Docker) | Model serving for eval | Assumed available on DGX | Phase 4.4 used same | — |
| wp-bench Docker grader | wp-bench scoring | `ghcr.io/wordpress/wp-bench-grader:latest` | Phase 4.4 used | — |
| `merge_tinker_v3.py` | RL LoRA merge before serving | On-disk | — | No fallback; MANDATORY |
| `claude` CLI | `scripts/claude_agent.py` subprocess dispatch | Assumed available | Phase 4.4 used | No alternative |
| Tinker cloud credentials | Phase 9 live run | Credential-gated | — | No fallback (manual) |
| `scripts/compute_concentration.py` | `bootstrap_ci` function | On-disk confirmed | — | — |

**Missing dependencies with no fallback:**
- Tinker cloud credentials (blocks Wave 1 + Wave 2; Wave 0 unaffected)

**Missing dependencies with fallback:**
- None (all needed Wave 0 deps are on disk)

---

## Security Domain

security_enforcement not set to false in config.json — included.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | eval is internal localhost; no auth surface |
| V3 Session Management | no | stateless eval scripts |
| V4 Access Control | no | no user-facing endpoints |
| V5 Input Validation | yes (limited) | eval results parsed from JSON; use `json.loads` with exception handling |
| V6 Cryptography | no | no crypto operations |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Prompt injection via eval dataset | Tampering | eval_judge.py already sanitizes; `enable_thinking=False` prevents think-block leakage |
| Poisoned checkpoint from Tinker export | Tampering | Verify checkpoint hash from manifest before merge; flag if manifest shows unexpected halts |
| LLM-generated code in rubric eval triggers RCE via Docker grader | Elevation | wp-bench grader runs in Docker with WP env isolation — already mitigated |

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `dgx.execute("eval_toolbox")` | vLLM Docker serve + eval_gen/judge HTTP | Phase 4.4 | Direct HTTP; no toolbox abstraction layer |
| `Agent(run_in_background=true)` in Python | `scripts/claude_agent.py` subprocess | Phase 4.4 | $0 subscription cost, no API key needed |
| Merging via full LoRA application | `merge_tinker_v3.py` MoE-only path | Phase 4.3 | Correct handling of MoE architecture; preserves router |
| Single checkpoint eval | Two-checkpoint (best-by-reward + final) | D-10-02 (Phase 10) | Robust to late training divergence |
| Point-threshold gates | CI-aware bootstrap lower bound | Phase 7 (D-09) | No false failures from eval noise |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `reward_breakdown.<wp_gen>/<wp_judge>` sub-fields exist in live rl_metrics.jsonl (Phase 9 UAT test #3 pending) | Artifact Schemas | RLEV-02 report section 1 (reward convergence by mode) fails to parse; fallback to `reward_mean` only |
| A2 | +3.58 offset differs for RL model (Tinker LoRA merge path) vs v1.2 (bf16 merge path) | +3.58 Offset section | If offset is identical, no harm; if not assumed, might incorrectly apply it |
| A3 | `merge_tinker_v3.py` docstring states Tinker MoE tensor convention verified via `checkpoint.tar` inspection (2026-06-07). Assumes Phase 9 Tinker LoRA export uses the same format. | Architecture Patterns | LOW risk — same tool, same Tinker convention. Verify checkpoint format before merge in Wave 1. |
| A4 | `LLM_BACKEND=vllm` + separate vLLM instance at :8000 is available for rubric_scorer LLM checks (Pitfall 7) | Pitfalls | If only one GPU slot available, must use slower `claude` backend for rubric LLM checks |
| A5 | Phase 9 live run will produce checkpoints at every 50 steps as configured in `run_args.checkpoint_every` | D-10-02 checkpoint selection | If run is shorter than 50 steps or Tinker export truncates, only dry-run checkpoint available |

**Highest risk assumption:** A3 (merge_tinker_v3.py Tinker LoRA compatibility) — verify in Wave 0 by reading `merge_tinker_v3.py` and confirming its input format matches Tinker checkpoint format. This is a plan-blocking item if incompatible.

---

## Open Questions

1. **Tinker LoRA checkpoint format vs merge_tinker_v3.py input format**
   - What we know: `merge_tinker_v3.py` exists and handles MoE-only merges (Phase 4.3). Tinker exports via `tinker://` → HF archive.
   - What's unclear: Whether `merge_tinker_v3.py` expects HF-format LoRA weights or a different format.
   - Recommendation: Wave 0 Task 0 should read `merge_tinker_v3.py` and confirm input format. If incompatible, Wave 1 needs a format adapter.

2. **v1.2 SFT anti-hack CI bounds (real, not fixture)**
   - What we know: `output/antihack_validation/acceptance_report.json` is fixture-backed. Phase 8 "Known Follow-Up" says live scoring is needed.
   - What's unclear: Whether the v1.2 SFT anti-hack live scoring was ever done (not on disk).
   - Recommendation: Wave 1 must run v1.2 SFT through the anti-hack JSONLs (using `output/merge_v4_winner`) to establish the real baseline CI bounds before comparing RL model bounds.

4. **D-10-04 #4 — RL retention bar: what is the correct threshold?**
   - What we know: CONTEXT.md says "Protected-expert retention >= the Phase 7 baseline (CI-aware vs protected_expert_mask)". Phase 7 `jaccard_ci_lower=0.9426` (from `concentration_report.json`) is a cross-run SFT stability measure (per-layer Jaccards across profiling runs), not an RL retention bar.
   - What is unclear: Whether D-10-04 #4 intends (a) RL per-step `jaccard_protected` mean CI lower bound >= 0.9426, or (b) a new threshold, or (c) a monitoring-only signal for human review.
   - Recommendation: Provisional bar for planning = mean `jaccard_protected` across all live-run steps CI lower bound >= 0.85 (lower than the SFT stability bar; frozen router makes routing collapse unlikely; exact value needs user confirmation). The RLEV-02 report should present the full `jaccard_protected` trace regardless of gate disposition.

3. **Two vLLM instances: eval model (:8020) + rubric LLM checks (:8000)**
   - What we know: rubric_scorer's LLM checks can use local vLLM at :8000. The model-under-eval is served at :8020.
   - What's unclear: Whether DGX GPU memory allows two simultaneous vLLM instances (30B + 35B).
   - Recommendation: Planner should sequence evals (rubric LLM checks as part of eval_gen.py inline; model-under-eval via :8020 only). Use `LLM_BACKEND=claude` for rubric LLM checks at Phase 10 eval scale (~few hundred examples) to avoid two-GPU constraint.

---

## Sources

### Primary (HIGH confidence)
- On-disk: `scripts/compute_concentration.py` — `bootstrap_ci` signature and implementation verified
- On-disk: `output/antihack_validation/acceptance_report.json` — Phase 8 anti-hack baseline schema
- On-disk: `output/eval_reasoning_v4_winner/judge_recalibration.json` — +3.58 offset spec
- On-disk: `output/rl_checkpoints/checkpoint_manifest.json` — Phase 9 manifest schema (dry-run)
- On-disk: `output/rl_checkpoints/metrics/rl_metrics.jsonl` — per-step fields (dry-run)
- On-disk: `eval/rubric_definitions.py` — 193 checks, method breakdown per dim (verified via `python3`)
- On-disk: `eval/eval_gate.py`, `eval/eval_gen.py`, `eval/eval_judge.py` — interface specs
- On-disk: `config/wp-bench.yaml` — wp-bench config
- On-disk: `output/04.4_wp_bench_results.json` — D-10-03 baseline (0.4616)
- On-disk: `output/eval_reasoning_v4_winner/revl04_rebench/reasoning_merged/wp_bench_results_20260613_214919.json` — 344 per-task results for floor derivation
- `.planning/phases/10-rl-comparative-evaluation/10-CONTEXT.md` — all locked decisions
- `.planning/phases/09-gspo-training/09-CONTEXT.md` + `09-HUMAN-UAT.md` — Phase 9 status

### Secondary (MEDIUM confidence)
- `scripts/rl_train.py` (first 80 lines) — GSPO primary, router frozen, MANIFEST_PATH, METRICS_PATH
- `scripts/build_antihack_set.py` — bootstrap CI gate pattern in Phase 8
- `.claude/skills/wp-finetune:run-evaluation/SKILL.md` — serving + eval orchestration pattern

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Eval infra interfaces: HIGH — verified on-disk via Read + grep
- Bootstrap CI wiring: HIGH — `bootstrap_ci` confirmed in compute_concentration.py; used by Phase 7 + 8
- Artifact schemas: HIGH — all JSON files read or inspected on-disk
- RL metrics field names: MEDIUM — top-level fields verified in dry-run jsonl; sub-fields like `reward_breakdown.<wp_gen>` assumed (UAT #3 pending)
- Merge compatibility (A3): LOW — script exists, Tinker export format not confirmed compatible

**Research date:** 2026-06-21
**Valid until:** 2026-07-21 (eval infra stable; rl_metrics sub-fields may change on live run)
