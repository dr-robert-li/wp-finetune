# Phase 8: Reward Infrastructure — Research

**Researched:** 2026-06-19
**Domain:** Composite reward pipeline (GRPO/MO-GRPO + VeRPO) for WordPress PHP code generation RL training
**Confidence:** HIGH (eval harness verified from source; formulas ASSUMED — see Assumptions Log)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-08-01** — Reuse `eval/rubric_scorer.py` + `eval/llm_checks.py` (PHPCS / security / WP-standards) and `eval/eval_judge.py` for the frozen `wp_judge` invocation. No signal drift between eval and reward.

**D-08-02 / D-V4-09** — Apply `+3.58` `score_offset` to the RAW `wp_judge` overall score → clip to valid range → THEN MO-GRPO normalize. Order: offset → clip → normalize. Source of truth: `output/eval_reasoning_v4_winner/judge_recalibration.json`; never hardcode the literal in two places.

**D-08-03 / D-11** — Anti-hack eval set built by perturbing real gen+judge JSONL outputs on three fixed axes (verbose padding, template-critique collapse, self-preference swap). Claude Code agents (`Agent(run_in_background=true)` per `wp-finetune:run-data-pipeline`) score candidates during construction. Pass criterion: CI-aware — bootstrap lower bound of adversarial reward must be below clean-baseline reward.

**D-08-04** — `reward_pipeline` exposes per-sample entry point returning `(scalar, breakdown_dict)`. Breakdown carries each signal pre- and post-normalization. Call accepts rollout group for MO-GRPO variance and VeRPO difficulty. Epsilon floor on zero-variance groups.

**Validation gate hygiene** — All Phase-8 acceptance gates CI-aware: report bootstrap CIs, require lower bound to clear bar, measured identically on baseline + candidate.

**Constraint** — `scripts/reward_pipeline.py` + pytest. No new skill. Reward compute = deterministic signals + frozen local `wp_judge`. No external Anthropic API in reward compute (Claude Code agents only for anti-hack SET construction).

### Claude's Discretion

- VeRPO per-check difficulty formula (within-group pass-rate weighting) — implement per GRPO-04 semantics
- MO-GRPO epsilon floor value (suggest 1e-8; adjust if numerical issues observed)
- Bootstrap n_boot (suggest 1000, matching Phase 7 convention)
- Anti-hack case count per axis (suggest 15-20 per axis, ~50 total)
- Internal dataclass vs plain dict for breakdown (suggest dataclass for type safety)

### Deferred Ideas (OUT OF SCOPE)

- CI-as-weight-discount for judge component (option B)
- Synthesize-fresh / hand-curated anti-hack sets
- Router-shift stabilization, protected-expert routing regularizer (Phase 9)
- Dual-mode RL rewards for judge reasoning (GRPO-05/06/07, Phase 9)
- Running GSPO/GRPO RL (Phase 9)
- RL-vs-SFT evaluation (Phase 10)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GRPO-01 | Composite reward pipeline with 70% verifiable / 30% judge weighting — PHPCS pass rate, security scanner hard gate, WP-standards VeRPO partial credit, frozen wp_judge MO-GRPO normalized | `score_code()` for 70%; `_judge_create`+`parse_judge_response` for 30%; see §eval-reuse-surface |
| GRPO-02 | Security scanner hard gate — reward=0 on security failure regardless of all other scores | Hard gate as terminal override after normalization; see §security-hard-gate |
| GRPO-03 | MO-GRPO normalization on all reward signals — within-group variance normalization to prevent single-signal dominance | See §mo-grpo-normalization |
| GRPO-04 | VeRPO-style partial credit for WP-standards checks — each check weighted by difficulty (pass rate across group samples) | See §verpo-partial-credit |
</phase_requirements>

---

## Summary

Phase 8 builds `scripts/reward_pipeline.py` — a standalone Python module that maps a PHP code generation (with its rollout group) to a scalar reward and a structured breakdown dict. The pipeline reuses the existing `eval/` harness verbatim as the signal source, combining 70% verifiable deterministic signals (PHPCS pass rate, WP-standards VeRPO partial credit, security gate) with 30% frozen `wp_judge` score (offset-corrected by +3.58, then MO-GRPO normalized). A security hard gate overrides the final composite to 0 if a critical security check fires. The pipeline also produces an anti-hack eval set by perturbing real gen+judge outputs along three axes, validated via CI-aware bootstrap gates.

The eval harness needs a **small surgical refactor**: `eval_judge.run_eval` is a monolithic function that loads a dataset and queries a judge in batch — it exposes no per-sample call point. A thin wrapper must be extracted so `reward_pipeline.py` can invoke the judge on a single generation without dataset I/O. Everything else in `rubric_scorer.py` and `llm_checks.py` is already structured for per-sample use.

**Primary recommendation:** Implement in five focused tasks — (1) eval harness wrapper refactor, (2) verifiable signal component, (3) judge component with recalibration, (4) MO-GRPO+VeRPO normalization layer, (5) anti-hack set construction and CI-aware gate. Gate each with pytest before moving to the next. Phase 9 depends on this API being stable.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| PHPCS / PHPStan / regex scoring | `scripts/reward_pipeline.py` | `eval/rubric_scorer.py` (owned) | Per-sample, deterministic; `score_code()` is the callable boundary |
| Security hard gate | `scripts/reward_pipeline.py` | — | Must be applied AFTER normalization as terminal override (see §security-hard-gate) |
| Judge invocation | Thin wrapper in `eval/` (new function) | `eval/eval_judge.py` internals | `run_eval` is batch-only; per-sample wrapper must be extracted |
| Recalibration offset | `scripts/reward_pipeline.py` | `output/eval_reasoning_v4_winner/judge_recalibration.json` | Runtime read, never hardcoded |
| MO-GRPO normalization | `scripts/reward_pipeline.py` | — | Group-level computation; can only run with full rollout group |
| VeRPO difficulty weighting | `scripts/reward_pipeline.py` | — | Group-level pass-rate computation; requires rollout group |
| Bootstrap CI gate | Anti-hack construction script | `scripts/compute_concentration.bootstrap_ci` (reused) | Existing helper; no rebuild |
| Anti-hack set construction | Claude Code agents (background) | Perturbation script | Sanctioned pattern per D-11 / run-data-pipeline SKILL.md |
| Breakdown logging | `scripts/reward_pipeline.py` breakdown dict | Phase 9 GSPO trainer | RLEV-02 shape; per-sample logging mirrors existing JSONL convention |

---

## Standard Stack

### Core (all already in repo — no new packages)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `eval.rubric_scorer` | repo-local | PHPCS / PHPStan / regex scoring | D-08-01: single source of truth for deterministic signals |
| `eval.llm_checks` | repo-local | LLM-assisted check batch (opt-in) | Supplementary to PHPCS; same `score_code()` call path |
| `eval.eval_judge` | repo-local (refactor needed) | `wp_judge` frozen invocation | D-08-01; refactor to expose per-sample callable |
| `scripts.compute_concentration.bootstrap_ci` | repo-local | CI-aware gate on anti-hack set | Already exists, used in Phase 7; `(lo, hi) = bootstrap_ci(arr)` |
| `numpy` | installed | Bootstrap resampling, epsilon guard | Already a dep |
| `dataclasses` | stdlib | Reward breakdown contract | Type safety; no new dep |
| `json` | stdlib | Read `judge_recalibration.json` at load time | No new dep |

### No New External Packages Required

The locked constraint is explicit: no new packages, no external Anthropic API. All logic uses existing repo deps.

---

## Package Legitimacy Audit

**No new external packages are installed in Phase 8.** All dependencies are existing repo-local modules or stdlib. Audit section is SKIPPED (no new installs).

---

## Architecture Patterns

### System Architecture Diagram

```
Rollout group (G generations for prompt P)
        |
        v
+------------------------------------------+
|  reward_pipeline.compute_group_rewards() |
|                                          |
|  For each generation g in group:         |
|    score_code(g.php) -> RubricScore      |  <- eval/rubric_scorer.py (unchanged)
|    judge_score_single(g.php) -> float    |  <- thin wrapper (new) over eval_judge
|    apply_offset_clip(raw_judge) -> float |  <- +3.58, clip [0,100]
|                                          |
|  Per-signal within-group normalization:  |
|    phpcs_norm  = mo_grpo_norm(phpcs_arr) |
|    verpo_norm  = verpo_weight(wp_std_arr)|
|    judge_norm  = mo_grpo_norm(judge_arr) |
|                                          |
|  Composite (before gate):                |
|    raw = 0.70 * (0.5*phpcs + 0.5*verpo) |
|          + 0.30 * judge_norm             |
|                                          |
|  Security hard gate (TERMINAL override): |
|    final = 0.0 if security_fail else raw |
|                                          |
|  Return (scalar, breakdown_dict)         |
+------------------------------------------+
        |
        v
Phase 9 GSPO trainer consumes (scalar, breakdown)
Anti-hack validation reads breakdown for CI gate
```

Notes on diagram:
- The 70% split between PHPCS and VeRPO (WP-standards) within the verifiable block is a planner discretion item — shown as 50/50 here but configurable.
- Security gate is TERMINAL (after normalize+combine), not before. This is critical — see §security-hard-gate.

### Recommended Project Structure

```
scripts/
├── reward_pipeline.py      # Main deliverable: compute_reward(), compute_group_rewards()
├── build_antihack_set.py   # Anti-hack perturbation script (new, D-11)
├── compute_concentration.py  # bootstrap_ci() already here — REUSE
eval/
├── rubric_scorer.py        # Unchanged — score_code() entry point
├── llm_checks.py           # Unchanged — opt-in LLM checks
├── eval_judge.py           # Small refactor: extract judge_score_single()
tests/
├── test_reward_pipeline.py # Unit + integration tests (new)
├── test_antihack.py        # Anti-hack gate tests (new)
```

---

## Eval Reuse Surface

### 1. Verifiable Signals (70%): `eval/rubric_scorer.py`

**Primary callable:**

```python
# Source: eval/rubric_scorer.py line 723
from eval.rubric_scorer import score_code, RubricScore

result: RubricScore = score_code(php_source_code, file_path="<generated>")
```

**Key fields on `RubricScore` dataclass (verified from source):**

| Field | Type | Use in reward pipeline |
|-------|------|----------------------|
| `result.overall` | `float` (0-100) | PHPCS overall score before normalization |
| `result.dimension_scores` | `dict[str, Optional[float]]` | Per-dim scores for D1..D9 |
| `result.triggered_checks` | `dict[str, list[str]]` | Check IDs that fired, grouped by dimension |
| `result.check_evidence` | `dict[str, list[str]]` | Evidence strings per check |
| `result.floor_rules_applied` | `list[str]` | Which CRITICAL_FLOOR_RULES fired |
| `result.dimension_na` | `list[str]` | Dimensions marked N/A for this code |

**Security signal (for hard gate):** Read `result.dimension_scores.get("D2_security")`. The threshold for security gate is a distinct design choice (see §security-hard-gate Open Question). `triggered_checks.get("D2_security", [])` lists the specific SEC-N* check IDs that fired.

**WP-standards dimension keys** (for VeRPO, from `dim_map.json`):**

```python
# All 9 eval dimensions available as keys in dimension_scores:
ALL_DIMS = ["D1_wpcs", "D2_security", "D3_sql", "D4_perf",
            "D5_wp_api", "D6_i18n", "D7_a11y", "D8_errors", "D9_structure"]

# Dimension weights from eval/dim_map.json (canonical source):
DIM_WEIGHTS = {
    "D1_wpcs": 0.10, "D2_security": 0.20, "D3_sql": 0.15, "D4_perf": 0.10,
    "D5_wp_api": 0.10, "D6_i18n": 0.10, "D7_a11y": 0.08,
    "D8_errors": 0.10, "D9_structure": 0.07
}
```

**PHPCS standards used (3 standards, verified):** `WordPress`, `WordPressVIPMinimum`, `Security` — all three run automatically inside `score_code()`. No need to call `run_phpcs()` individually from `reward_pipeline.py`.

**LLM checks:** Controlled by `RUBRIC_USE_LLM_CHECKS=1` env var inside `score_code()`. Default OFF (deterministic only). Reward pipeline should not set this env var — deterministic signals only in reward compute.

**No refactor needed for rubric_scorer.py.** It is already per-sample callable.

### 2. Judge Signal (30%): `eval/eval_judge.py` — Refactor Required

**Current state (verified from source):** `eval_judge.run_eval()` (line ~325) is batch-only — it loads a full JSONL dataset, iterates over all wp_judge examples, queries the judge endpoint in a loop, and returns aggregate Spearman results. There is NO per-sample entry point.

**Internal functions available for extraction (verified):**

```python
# From eval/eval_judge.py (lines 46-75)
def _judge_create(client, *, model, messages, max_tokens=1024, temperature=0.0):
    """Queries vLLM with enable_thinking=False guard (RC-A fix). MUST reuse this."""

# From eval/eval_judge.py (lines 154-207)
def parse_judge_response(response: str) -> Optional[dict]:
    """Strips <think> blocks, tries 4 JSON extraction strategies. Returns dict or None."""
    # Key field: parsed.get("overall_score") -> int/float
```

**Required new function** (`eval/eval_judge.py` or a new `eval/judge_utils.py`):

```python
def judge_score_single(
    php_code: str,
    client: openai.OpenAI,
    model: str,
) -> Optional[float]:
    """Invoke wp_judge on a single PHP code string. Returns raw overall_score (0-100) or None."""
    messages = [{"role": "user", "content": f"<wp_judge> Evaluate this WordPress code:\n\n{php_code}"}]
    resp = _judge_create(client, model=model, messages=messages, max_tokens=512, temperature=0.0)
    raw_text = resp.choices[0].message.content
    parsed = parse_judge_response(raw_text)
    if parsed is None:
        return None
    overall = parsed.get("overall_score")
    return float(overall) if isinstance(overall, (int, float)) else None
```

**Critical:** Must reuse `_judge_create` (not raw `client.chat.completions.create`) to preserve the `enable_thinking=False` RC-A guard. If `judge_score_single` bypasses `_judge_create`, it will reproduce the thinking-block parse failure on the merged model.

**Client setup pattern (from eval_judge.py lines 363-368):**

```python
import os
resolved_base_url = base_url or os.environ.get("EVAL_JUDGE_BASE_URL")
if not resolved_base_url:
    from scripts.dgx_toolbox import get_toolbox
    resolved_base_url = get_toolbox().vllm_endpoint()
client = openai.OpenAI(base_url=resolved_base_url, api_key="none")
```

### 3. Recalibration Artifact Shape (verified)

```json
{
  "score_offset": 3.58,
  "applies_to": "wp_judge overall score",
  "ci_95": [1.24, 6.09],
  "rank_invariant": true,
  "note": "Under MO-GRPO within-group normalization a uniform offset largely cancels..."
}
```

**Load at pipeline init:**

```python
import json
from pathlib import Path

_RECALIB_PATH = Path("output/eval_reasoning_v4_winner/judge_recalibration.json")

def _load_score_offset() -> float:
    data = json.loads(_RECALIB_PATH.read_text())
    return float(data["score_offset"])
```

Never hardcode `3.58` in `reward_pipeline.py`. The planner must add a task that reads this path at module load time and stores it in a module-level constant.

---

## MO-GRPO Normalization

### Within-Group Normalization Formula [ASSUMED]

For a rollout group of G samples for the same prompt, signal `s` with raw scores `[x_1, ..., x_G]`:

```
mu_s = mean([x_1, ..., x_G])
sigma_s = std([x_1, ..., x_G])           # population std over the group
epsilon = 1e-8                            # epsilon floor (D-08-04)
x_i_norm = (x_i - mu_s) / (sigma_s + epsilon)
```

This produces centered, unit-variance normalized signals within the group. The `epsilon` floor prevents division by zero when all rollouts produce identical scores on a signal (common for PHPCS on easy prompts or when the judge is very confident).

**Key consequence — the +3.58 offset largely cancels under normalization.** A uniform offset `c` added to all group members shifts `mu_s` by `c` but cancels in `(x_i - mu_s)`. The artifact note explicitly states this. The offset's real bite is on **absolute-threshold gates** (e.g., security cutoff, gate-time pass/fail verdicts) and CI-aware gate comparisons, NOT the RL advantage. The locked order (offset → clip → normalize) is correct and must be followed.

### Combining into 70/30 Composite

```python
composite = 0.70 * verifiable_norm + 0.30 * judge_norm
```

Where `verifiable_norm` is itself a weighted combination of the verifiable signal components. Suggested split within the 70% block (planner discretion):

- PHPCS overall (high-variance anchor): weight W1
- VeRPO partial credit: weight W2
- W1 + W2 = 1.0, both signals already normalized by MO-GRPO per-signal

**Anti-dominance:** Because each signal is independently normalized before combining, no single signal can dominate via scale differences. The epsilon floor prevents zero-variance signals from producing NaN and blowing up.

---

## VeRPO Partial Credit

### Per-Check Difficulty [ASSUMED]

For each WP-standards check `c` evaluated across the rollout group of G samples:

```
pass_rate_c = sum(check_pass[i][c] for i in 1..G) / G
difficulty_c = 1.0 - pass_rate_c    # rare pass -> higher difficulty -> more signal
```

### Per-Sample VeRPO Score [ASSUMED]

For sample i:

```
verpo_i = sum(difficulty_c * int(check_pass[i][c]) for c in CHECKS) / (sum(difficulty_c for c in CHECKS) + epsilon)
```

This is a difficulty-weighted pass fraction. A sample that passes a rare check (difficulty close to 1.0) gets more reward than a sample that passes a common check (difficulty close to 0.0).

### Integration with RubricScore

`result.triggered_checks` contains `{dim_key: [check_ids_that_fired]}`. For VeRPO, "check fired" means the check triggered (for NEGATIVE checks = violation, for POSITIVE checks = compliance). The planner must clarify direction: for POSITIVE checks, `check_hits[cid] == True` means pass (reward). For NEGATIVE checks, `check_hits[cid] == True` means violation (no reward). The mapping table `POSITIVE_CHECK_IDS` and `NEGATIVE_CHECK_IDS` in `rubric_scorer.py` is the canonical source.

**Practical scope:** VeRPO partial credit applies to WP-standards checks specifically (D2 security, D3 SQL, D5 WP API, D1 WPCS, etc.). Apply across all checks or only a WP-standards subset — this is a planner discretion item. Suggestion: apply to ALL 9 dimensions' checks (full rubric scope) for maximum signal breadth.

---

## Security Hard Gate

### Placement: Terminal Override (AFTER normalization + combine) [VERIFIED from D-08-02]

**Critical constraint:** The hard gate must be applied AFTER the composite is computed, not before or during normalization:

```python
composite = 0.70 * verifiable_norm + 0.30 * judge_norm
final = 0.0 if security_fail else composite   # TERMINAL override
```

**Why order matters:** If `reward=0` is injected BEFORE MO-GRPO normalization, the normalization step re-centers the group — the 0 becomes a negative advantage relative to other group members. It does not stay 0 in the RL loss. The terminal override pattern preserves the hard zero in the final scalar.

### Open Question: Defining "Security Failure"

Two candidate definitions exist and they are NOT equivalent:

| Option | Definition | Source |
|--------|-----------|--------|
| A | Any critical SEC-N* check fires (`"D2_security"` in `triggered_checks`) | Hardest; any single violation = 0 |
| B | `D2_security` dimension score < 8.0 (80% bar, confirmed in obs 2050) | Softer; minor violations pass |
| C | Any `CRITICAL_FLOOR_RULE` for `D2_security` fires (capping D2 below `cap`) | Middle ground; only critical patterns trigger gate |

D-08-02 specifies "fails security scan" but does not resolve which of A/B/C. The planner must surface this for user confirmation before implementation. The reward=0 gate must be tested explicitly with a "secure-failing but otherwise high-quality" case (SC2 from CONTEXT.md specifics).

**Recommendation:** Use Option C (CRITICAL_FLOOR_RULE triggers) as the default — it matches the rubric's own severity classification for what constitutes a critical security issue. Option A (any SEC-N*) is too aggressive for PHPCS minor-severity sniffs.

---

## Anti-Hack Eval Set (D-11)

### Three Perturbation Axes (locked D-08-03)

| Axis | Hack Type | Perturbation Recipe | Expected Effect |
|------|-----------|---------------------|-----------------|
| 1 | Verbose padding | Insert inert PHP comments, redundant docblocks, whitespace expansions | Inflates length without improving quality |
| 2 | Template-critique collapse | Replace reasoning with stock boilerplate critique phrases ("This code could be improved by...") | Passes length check but signals nothing |
| 3 | Self-preference swap | Modify judge input so judge is evaluating its own training-target output | Biases toward self-scoring high regardless of quality |

### Construction Protocol

**Source data:** Real gen+judge JSONL outputs from `data/` and `output/`. Look for `eval_gen_results.jsonl`, `eval_judge_results.pairs.jsonl` or equivalent under `output/eval_reasoning_v4_winner/`. These contain paired (generation, judge-score) records.

**Per-axis process:**
1. Select ~15-20 real records where the clean reward would be MEDIUM-HIGH (otherwise trivial to distinguish)
2. Apply axis-specific perturbation function (pure Python string manipulation — no model call)
3. Use Claude Code background agents to score perturbed candidates via the reward pipeline (confirms the pipeline produces a lower reward)
4. Accept cases where `bootstrap_ci(perturbed_rewards)[0] < clean_baseline_reward` (lower bound of perturbed CI is below clean baseline)

**Scoring agents:** Use `Agent(run_in_background=True)` per `wp-finetune:run-data-pipeline` SKILL.md. Each agent receives a batch of perturbed candidates and calls `reward_pipeline.compute_reward()`. Results are collected into a JSONL file.

**No external Anthropic API** in reward compute itself. The Claude Code agents only call `reward_pipeline` which uses local vLLM.

### Bootstrap CI Gate

**Reuse:** `scripts.compute_concentration.bootstrap_ci` (verified at line 42):

```python
from scripts.compute_concentration import bootstrap_ci

perturbed_rewards = [compute_reward(case) for case in perturbed_cases_axis]
clean_rewards = [compute_reward(case) for case in matched_clean_cases]

lo_perturbed, hi_perturbed = bootstrap_ci(np.array(perturbed_rewards), n_boot=1000)
lo_clean, hi_clean = bootstrap_ci(np.array(clean_rewards), n_boot=1000)

# Anti-hack gate passes if perturbed CI upper is below clean CI lower
gate_pass = hi_perturbed < lo_clean
```

**Reporting:** Always report `(lo_perturbed, hi_perturbed)` and `(lo_clean, hi_clean)` — not just pass/fail. The CI-aware disposition requires publishing both CIs in the acceptance report.

---

## Reward Output Contract

### Dataclass Definition (GRPO-01 / D-08-04)

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RewardBreakdown:
    # Pre-normalization (raw signal values)
    phpcs_raw: float              # rubric_scorer overall (0-100)
    verpo_raw: float              # VeRPO weighted pass fraction (0-1)
    judge_raw: Optional[float]    # raw wp_judge overall_score (0-100), None if parse failure
    judge_offset_applied: float   # judge_raw + score_offset, clipped (0-100)
    security_score_raw: float     # D2_security dimension score (0-10)
    security_fail: bool           # whether hard gate triggered

    # Post-normalization (within-group centered)
    phpcs_norm: float
    verpo_norm: float
    judge_norm: float

    # Composite (pre-gate)
    composite_pre_gate: float

    # Per-check VeRPO difficulty (for RLEV-02 logging)
    check_pass_rates: dict         # {check_id: pass_rate_across_group}
    check_difficulties: dict       # {check_id: difficulty_weight}

    # Group metadata
    group_size: int
    group_phpcs_mean: float
    group_phpcs_std: float
    group_judge_mean: float
    group_judge_std: float


@dataclass
class RewardResult:
    scalar: float                 # Final reward (0.0 if security gate, else composite)
    breakdown: RewardBreakdown    # Full breakdown for RLEV-02 logging
```

### Per-Sample API Signature (Phase 9 GSPO trainer interface)

```python
def compute_reward(
    php_code: str,
    group_signals: "GroupSignals",   # pre-computed group stats (normalization params)
    judge_client: openai.OpenAI,
    judge_model: str,
) -> RewardResult:
    """Score a single generation given pre-computed group normalization params."""
    ...

def compute_group_rewards(
    php_codes: list[str],            # G rollouts for one prompt
    judge_client: openai.OpenAI,
    judge_model: str,
) -> list[RewardResult]:
    """Score all G generations, computing group normalization in one pass."""
    ...
```

**Two-pass design for group rewards:**
1. Collect all raw signals for all G samples (rubric + judge)
2. Compute group mean/std per signal
3. Normalize each sample against group stats
4. Apply hard gate, return list of RewardResult

---

## Common Pitfalls

### Pitfall 1: Security Gate Before Normalization
**What goes wrong:** Injecting `reward=0` before MO-GRPO normalization. Normalization re-centers; the 0 becomes a negative relative advantage, not a hard floor.
**Why it happens:** Natural intuition is "set to 0 early." But normalization is a group operation.
**How to avoid:** Apply security override AFTER `composite = 0.70 * verifiable_norm + 0.30 * judge_norm`. The terminal override pattern: `final = 0.0 if security_fail else composite`.
**Warning signs:** Integration test with security-failing + high-quality samples shows non-zero final reward.

### Pitfall 2: Bypassing `_judge_create` in Judge Wrapper
**What goes wrong:** Using `client.chat.completions.create` directly for per-sample judge invocation. The merged model emits unclosed `<think>` blocks without `enable_thinking=False` kwarg, causing 19-25% parse failures.
**Why it happens:** `_judge_create` is a private function; it looks like an implementation detail.
**How to avoid:** The new `judge_score_single` wrapper MUST call `_judge_create` (or copy its logic exactly). This is the RC-A fix established in Phase 04.4.

### Pitfall 3: Hardcoding the 3.58 Offset
**What goes wrong:** `judge_offset = raw_score + 3.58` hardcoded in `reward_pipeline.py`. Future recalibration requires a code change, not a JSON update.
**Why it happens:** Convenient during development.
**How to avoid:** `score_offset = _load_score_offset()` at module init; use the variable everywhere.

### Pitfall 4: Zero-Variance Group Produces NaN
**What goes wrong:** All G rollouts score identically on one signal (e.g., PHPCS all pass) → `sigma = 0` → `x_i_norm = NaN`.
**Why it happens:** Common for easy prompts or when model collapses.
**How to avoid:** `sigma_eff = sigma + epsilon` where `epsilon = 1e-8`. Every division uses `sigma_eff`.

### Pitfall 5: VeRPO Check Direction Confusion
**What goes wrong:** Treating NEGATIVE check fires as "pass" (they indicate violations). This inverts the reward signal for negative-polarity checks.
**Why it happens:** `triggered_checks` contains IDs for all checks that "fired" — but for negative checks, firing = violation.
**How to avoid:** Consult `POSITIVE_CHECK_IDS` and `NEGATIVE_CHECK_IDS` from `rubric_scorer.py`. For VeRPO pass/fail: positive check fired = pass; negative check fired = fail.

### Pitfall 6: LLM Checks Running in Reward Compute
**What goes wrong:** `RUBRIC_USE_LLM_CHECKS=1` env var set in the RL training environment triggers 41-check LLM batches per generation, making reward compute take 30+ seconds per sample.
**Why it happens:** Env var leaks from evaluation environment.
**How to avoid:** `reward_pipeline.py` should explicitly ensure `os.environ.pop("RUBRIC_USE_LLM_CHECKS", None)` or document that reward compute must not have this env var set. Reward compute is deterministic signals only.

### Pitfall 7: Anti-Hack Cases on Trivially-Bad Originals
**What goes wrong:** Perturbing code that already scores low makes the perturbation indistinguishable from noise.
**Why it happens:** Random selection from real JSONL includes low-quality examples.
**How to avoid:** Filter source records to `rubric_score.overall >= 65.0` before perturbation. Perturbation should make MEDIUM-HIGH quality code score worse, not make BAD code score slightly worse.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PHPCS / PHPStan / WP-standards scoring | Custom PHP linter wrapper | `eval.rubric_scorer.score_code()` | Already handles 3 PHPCS standards + PHPStan + regex + edge cases (missing `<?php` tag, timeouts, N/A detection) |
| Judge invocation + think-block stripping | Raw `openai.OpenAI.chat.completions.create` | `eval.eval_judge._judge_create` + `parse_judge_response` | RC-A fix for unclosed `<think>` blocks; 4-strategy JSON extraction |
| Bootstrap confidence interval | Custom resampling loop | `scripts.compute_concentration.bootstrap_ci(values, n_boot=1000, alpha=0.05)` | Tested, matches Phase 7 convention |
| Dimension weights / check registry | Config dict in reward_pipeline.py | `eval/dim_map.json` + `eval.rubric_definitions.CHECK_REGISTRY` | Canonical sources; duplicate = drift |

**Key insight:** The eval harness was built to score exactly what the reward pipeline needs to score. Wrapping it rather than reimplementing avoids silent drift between eval time and training time.

---

## Testing

### Repo Pytest Conventions (verified from `tests/`)

- Test files in `tests/` directory, named `test_*.py`
- Class-based test grouping (`class TestXxx:`)
- GPU-free where possible; external service calls mocked
- Existing pattern for CI-aware gates: `tests/test_bootstrap_ci.py` (class `TestBootstrapCI`)
- Import style: `from scripts.module import function`
- Run with: `pytest tests/ -x` (stop on first failure)

### Unit Test Plan: `tests/test_reward_pipeline.py`

| Test Class | Test Cases | What It Verifies |
|-----------|-----------|------------------|
| `TestJudgeWrapper` | `test_judge_score_single_returns_float`, `test_judge_uses_judge_create`, `test_parse_failure_returns_none` | Refactored per-sample wrapper; RC-A guard present |
| `TestOffsetApply` | `test_offset_read_from_json`, `test_offset_clip_upper`, `test_offset_clip_lower`, `test_no_hardcoded_literal` | Recalibration load; clipping to [0,100] |
| `TestMOGRPONorm` | `test_zero_variance_epsilon`, `test_mean_centered`, `test_unit_variance`, `test_group_of_one` | Formula correctness; NaN prevention |
| `TestVeRPO` | `test_rare_check_weights_more`, `test_all_pass_score`, `test_all_fail_score`, `test_positive_negative_polarity` | Difficulty weighting; check direction |
| `TestSecurityGate` | `test_security_fail_overrides_to_zero`, `test_security_pass_preserves_composite`, `test_gate_applied_after_normalization` | Hard gate placement; SC2 case |
| `TestCompositeWeights` | `test_70_30_split`, `test_no_single_signal_dominance` | Weight fractions |
| `TestBreakdownContract` | `test_breakdown_has_pre_post_norm`, `test_breakdown_serializable` | RLEV-02 logging shape |

### Integration Test: 50-Case Known-Good/Bad Suite

```
tests/fixtures/reward_integration_cases/
├── known_good_php/          # 25 PHP files expected to score high
├── known_bad_php/           # 24 PHP files expected to score low
└── secure_fail_high_quality.php  # 1 file: high quality but security-failing -> reward=0 (SC2)
```

Integration test assertions:
- `known_good` mean reward significantly above 0
- `known_bad` mean reward significantly below `known_good`
- `secure_fail_high_quality.php` → `reward_result.scalar == 0.0`
- All results have complete `breakdown` dict with pre/post-norm fields

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (existing; no new install) |
| Config file | None detected; use `pytest tests/ -x` directly |
| Quick run command | `pytest tests/test_reward_pipeline.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase 8 Success Criteria → Test Map

From CONTEXT.md / GRPO-01..04 and RLEV-02:

| Req ID | Success Criterion | Test Type | Automated Command | File Exists? |
|--------|------------------|-----------|-------------------|-------------|
| GRPO-01 | 70/30 composite reward pipeline computes correct scalar | unit | `pytest tests/test_reward_pipeline.py::TestCompositeWeights -x` | No — Wave 0 |
| GRPO-02 | Security hard gate: reward=0 when security check fires | unit | `pytest tests/test_reward_pipeline.py::TestSecurityGate -x` | No — Wave 0 |
| GRPO-03 | MO-GRPO normalization: within-group centering, epsilon floor | unit | `pytest tests/test_reward_pipeline.py::TestMOGRPONorm -x` | No — Wave 0 |
| GRPO-04 | VeRPO partial credit: rare checks weighted higher | unit | `pytest tests/test_reward_pipeline.py::TestVeRPO -x` | No — Wave 0 |
| SC2 (specific) | Secure-failing but high-quality → reward=0 | integration | `pytest tests/test_reward_pipeline.py -k "secure_fail" -x` | No — Wave 0 |
| D-11 | Anti-hack set: all 3 axes perturbed CI below clean CI lower bound | integration | `pytest tests/test_antihack.py -x` | No — Wave 0 |
| RLEV-02 | Breakdown dict has all required fields for logging | unit | `pytest tests/test_reward_pipeline.py::TestBreakdownContract -x` | No — Wave 0 |

### CI-Aware Gate Measurements

For each acceptance gate:

1. Compute metric on baseline (clean-reward distribution): `lo_clean, hi_clean = bootstrap_ci(clean_rewards)`
2. Compute metric on candidate (perturbed/adversarial): `lo_perturbed, hi_perturbed = bootstrap_ci(perturbed_rewards)`
3. Gate disposition: `PASS` if `hi_perturbed < lo_clean` (anti-hack axis below clean)
4. Report: publish all four CI bounds in acceptance report, not just pass/fail

**Sampling rate:**
- Per task commit: `pytest tests/test_reward_pipeline.py -x -q`
- Per wave merge: `pytest tests/ -x -q`
- Phase gate: Full suite green before `/gsd:verify-work`

### Wave 0 Gaps (must be created before implementation)

- [ ] `tests/test_reward_pipeline.py` — covers GRPO-01..04 and SC2
- [ ] `tests/test_antihack.py` — covers D-11 CI-aware gate
- [ ] `tests/fixtures/reward_integration_cases/` — 50 PHP fixture files
- [ ] `tests/fixtures/reward_integration_cases/secure_fail_high_quality.php` — SC2 fixture

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single-signal reward (PHPCS only) | Multi-objective composite (70/30 verifiable+judge) | Phase 8 design | Prevents reward hacking via single metric; combines static analysis + semantic quality |
| Fixed reward thresholds | MO-GRPO within-group normalization | MO-GRPO literature | Adapts to group difficulty; self-normalizing across prompts |
| Equal check weights | VeRPO difficulty-weighted partial credit | Phase 8 design | Signal proportional to informativeness; rare-pass checks carry more weight |
| Bare pass/fail gates | CI-aware bootstrap gates (D-09 established Phase 7) | Phase 7 completion | Accounts for measurement noise; distinguishes signal from noise-band |

---

## Open Questions

1. **Security failure definition (CRITICAL for implementation)**
   - What we know: `D2_security` score range is 0-10; 8.0 is an established 80% threshold (obs 2050); CRITICAL_FLOOR_RULES exist for the security dimension; SEC-N* checks are granular
   - What's unclear: Is "security failure" = any SEC-N* check firing (Option A), or D2_security < 8.0 (Option B), or CRITICAL_FLOOR_RULE firing for D2_security (Option C)?
   - Recommendation: Default to Option C (CRITICAL_FLOOR_RULE triggers) as most aligned with rubric severity classification; confirm with user before implementation

2. **VeRPO scope: all 9 dimensions vs WP-standards subset**
   - What we know: GRPO-04 says "WordPress standards checks"; all 9 dimensions are available from `score_code()`
   - What's unclear: Does "WP-standards checks" mean only D1_wpcs + specific sniffs, or all 9 rubric dimensions?
   - Recommendation: Apply VeRPO to all checks across all 9 dimensions for maximum signal; narrow to WP-specific if this creates noise

3. **Internal split within the 70% verifiable block**
   - What we know: GRPO-01 says 70% verifiable = PHPCS pass rate + security (hard gate) + VeRPO partial credit
   - What's unclear: What fraction of the 70% does PHPCS overall vs VeRPO get?
   - Recommendation: Equal split (35/35) with both independently normalized; adjust if training shows imbalance

4. **Judge parse failure handling in reward**
   - What we know: ~5-25% parse failure rate is possible (RC-A fix reduces this but does not eliminate it)
   - What's unclear: If `judge_score_single` returns `None`, what is the judge component reward? (0, imputed from group mean, skip this sample)
   - Recommendation: Use group mean as fallback; log parse failures in breakdown_dict; flag if parse failure rate > 10%

5. **Anti-hack set size per axis**
   - What we know: D-11 specifies 3 axes; suggested 15-20 per axis = 45-60 total
   - What's unclear: CONTEXT.md does not fix a number
   - Recommendation: 15 per axis = 45 total; this fits in a 50-case integration test with the SC2 case

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `phpcs` CLI | `rubric_scorer.run_phpcs()` | Assumed present (used by Phase 4 eval) | — | Returns `{"_unavailable": True}` — graceful fallback exists |
| `phpstan` CLI | `rubric_scorer.run_phpstan()` | Assumed present | — | Graceful fallback exists |
| Local vLLM endpoint | `judge_score_single()` | Assumed running during RL training | — | No fallback — training requires it |
| `output/eval_reasoning_v4_winner/judge_recalibration.json` | `_load_score_offset()` | Verified present | 2026-06-14 | None — must exist at startup |
| `numpy` | Bootstrap CI | Present (existing dep) | — | None needed |
| `openai` Python SDK | Judge client | Present (existing dep) | — | None needed |

**Missing dependencies with no fallback:**
- Local vLLM endpoint serving `qwen3-wp` — required for judge component; reward pipeline must validate endpoint is reachable at startup and fail fast with clear error if not

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | MO-GRPO within-group normalization formula: `(x - mu) / (sigma + epsilon)` | §mo-grpo-normalization | If GSPO uses a different normalization (e.g., advantage clipping), mismatch with Phase 9 trainer expectation; Phase 9 planner must confirm |
| A2 | VeRPO difficulty weighting formula: `difficulty_c = 1 - pass_rate_c`; weighted sum | §verpo-partial-credit | If VeRPO uses a different weighting (e.g., log-odds), partial credit signal will be incorrect; confirm against VeRPO paper or Phase 9 requirements |
| A3 | Judge `overall_score` field range is 0-100 (matches `judge_recalibration.json` notes) | §eval-reuse-surface | If judge emits 0-10, offset and clipping logic is wrong; verify against one real judge output sample |
| A4 | `phpcs` and `phpstan` are available on the training machine (not just eval machine) | §environment-availability | Reward compute on DGX may fail if linters not installed; verify DGX environment before Phase 9 |
| A5 | VeRPO scope should cover all 9 rubric dimensions | §verpo-partial-credit | Narrower scope may be intended; see Open Question 2 |

---

## Sources

### Primary (HIGH confidence — verified from source in this session)

- `eval/rubric_scorer.py` lines 723-817 — `score_code()` signature, `RubricScore` dataclass fields, LLM checks opt-in mechanism
- `eval/eval_judge.py` lines 46-75, 154-207, 325-395 — `_judge_create()` RC-A fix, `parse_judge_response()`, `run_eval()` batch-only structure
- `output/eval_reasoning_v4_winner/judge_recalibration.json` — `score_offset=3.58`, shape, note on normalization cancellation
- `eval/dim_map.json` — 9 dimension keys, `dimension_weights` dict, canonical source designation
- `scripts/compute_concentration.py` lines 42-72 — `bootstrap_ci(values, n_boot=1000, alpha=0.05) -> (lo, hi)` signature and implementation
- `.planning/phases/08-reward-infrastructure/08-CONTEXT.md` — all locked decisions D-08-01..04
- `.planning/REQUIREMENTS.md` lines 158-161, 173 — GRPO-01..04, RLEV-02 verbatim

### Secondary (MEDIUM confidence — consistent with source but not exhaustively verified)

- `tests/test_bootstrap_ci.py` — pytest class-based convention; GPU-free test pattern
- `eval/rubric_definitions.py` (imported by rubric_scorer) — `POSITIVE_CHECK_IDS`, `NEGATIVE_CHECK_IDS`, `CRITICAL_FLOOR_RULES` existence confirmed via rubric_scorer imports

### Tertiary (LOW confidence / ASSUMED — see Assumptions Log)

- MO-GRPO normalization formula (A1): standard within-group standardization; not verified against Phase 9 GSPO implementation
- VeRPO partial credit formula (A2): standard difficulty-weighted pass fraction; not verified against original VeRPO paper

---

## Metadata

**Confidence breakdown:**
- Eval harness reuse surface: HIGH — verified directly from source files
- Security hard gate placement: HIGH — reasoning from normalization mechanics is definitive; formula for trigger definition is MEDIUM (Open Question 1)
- MO-GRPO formula: LOW-MEDIUM — standard formula, ASSUMED, planner must confirm with Phase 9 trainer
- VeRPO formula: LOW-MEDIUM — standard difficulty weighting, ASSUMED
- Anti-hack protocol: HIGH — locked by D-08-03/D-11; bootstrap helper verified
- Output contract: HIGH — derived from D-08-04 + RLEV-02 verbatim

**Research date:** 2026-06-19
**Valid until:** 2026-07-19 (stable; external APIs not involved; decay risk is LOW)

---

## Suggested Plan Breakdown

Three PLAN.md files covering five implementation waves:

**PLAN-08-01: Foundation (Wave 0 + Wave 1)**
- Objective: Create test fixtures + test stubs; refactor `eval_judge.py` to expose `judge_score_single()`; implement and test recalibration loader
- Key tasks: Wave 0 gap creation (test files + fixture PHP files); `judge_score_single` extraction with RC-A guard; `_load_score_offset()` module init; unit tests for offset/clip and judge wrapper
- Gate: All unit tests in `TestJudgeWrapper` + `TestOffsetApply` green

**PLAN-08-02: Core Reward Pipeline (Wave 2)**
- Objective: Implement `scripts/reward_pipeline.py` with MO-GRPO normalization, VeRPO partial credit, security hard gate, and `(scalar, breakdown)` return contract
- Key tasks: `RewardBreakdown` + `RewardResult` dataclasses; `compute_group_rewards()` two-pass implementation; MO-GRPO normalization with epsilon floor; VeRPO difficulty weighting; security terminal override; 50-case integration test (includes SC2)
- Gate: Full `tests/test_reward_pipeline.py` green including `TestSecurityGate::test_gate_applied_after_normalization` and SC2 fixture

**PLAN-08-03: Anti-Hack Set Construction + CI-Aware Validation (Wave 3)**
- Objective: Build and validate anti-hack eval set (D-11 D-08-03); run CI-aware bootstrap gate; produce acceptance report
- Key tasks: Perturbation script (`scripts/build_antihack_set.py`) covering 3 axes; Claude Code agent scoring batch; bootstrap CI comparison; acceptance report with all 4 CI bounds per axis
- Gate: `tests/test_antihack.py` green; acceptance report published to `output/antihack_validation/`; all 3 axes pass CI-aware gate (hi_perturbed < lo_clean)
