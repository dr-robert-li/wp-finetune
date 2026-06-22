---
phase: 09-gspo-training
reviewed: 2026-06-20T09:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - scripts/rl_train.py
  - scripts/rl_rollouts.py
  - scripts/rl_judge_dispatch.py
  - scripts/build_rl_prompts.py
  - scripts/tinker_rl_data.py
  - .claude/skills/wp-finetune:run-rl-training/SKILL.md
findings:
  critical: 7
  warning: 5
  info: 3
  total: 15
status: issues_found
---

# Phase 09: GSPO Training — Code Review Report

**Reviewed:** 2026-06-20T09:00:00Z
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Phase 9 implements a GSPO/GRPO training loop over Tinker SDK primitives. The core algorithmic primitives (RSPO floor, GSPO loss closure, `forward_backward_custom` dispatch) are correct. However, the wiring that connects those primitives together in `main()` is broken in three mutually fatal ways: the training loop cannot complete a single step in production. Additionally, the reward computation feeds z-scored values into a function that expects raw [0,1] inputs, corrupting every reward signal. Seven critical bugs, five warnings.

**Verified clean:** No hardcoded credentials. No Anthropic API import anywhere in `scripts/`. No `run_in_background` in judge dispatch. `judge_consistency_weight = 0.3` asserted ≤ 0.5 at module import. `ttl_seconds=None` (persistent) on all `save_weights_for_sampler` calls. `reward_pipeline.py` unmodified since Phase 8 (MO-GRPO normalization inside pipeline is correct and untouched).

---

## Critical Issues

### CR-01: Training loop cannot run — `sampling_client` assigned the wrong object

**File:** `scripts/rl_train.py:657-659`

**Issue:** `save_weights_for_sampler()` returns a checkpoint reference object (with `.path`), not a sampling client with a `.generate()` method. The result is stored as `sampling_client` and passed to `collect_rollouts`, which calls `sampling_client.generate(...)`. This raises `AttributeError` on step 0 of every real training run. The dry-run path (L552-615) mocks `tc` directly and never exercises this code path, so "dry-run passed" provides no confidence here.

```python
# L657-659 — WRONG
sampling_client = _res(tc.save_weights_for_sampler(name="init", ttl_seconds=None))

# FIX: obtain sampling client from tc directly, then save weights separately
sampling_client = tc.sampling_client()   # or however Tinker SDK exposes this
_res(tc.save_weights_for_sampler(name="init", ttl_seconds=None))
```

---

### CR-02: Training loop cannot run — `gen_pool` and `judge_pool` hardcoded empty

**File:** `scripts/rl_train.py:664-666`

**Issue:** `collect_rollouts` is called with `gen_pool=[], judge_pool=[]` on every step. `sample_interleaved_prompts` raises `ValueError("No prompts available in either pool")` immediately (or returns an empty batch, yielding a zero-length training step). `tinker_rl_data.load_rl_prompts()` is never called anywhere in `main()`. The module `tinker_rl_data` is imported at L5 but never invoked.

```python
# L664-666 — WRONG
rollouts = collect_rollouts(
    sampling_client=sampling_client,
    gen_pool=[],     # never populated
    judge_pool=[],   # never populated
    args=args,
)

# FIX
from scripts.tinker_rl_data import load_rl_prompts, BASE_MODEL
gen_pool = load_rl_prompts("gen")
judge_pool = load_rl_prompts("judge")
rollouts = collect_rollouts(
    sampling_client=sampling_client,
    gen_pool=gen_pool,
    judge_pool=judge_pool,
    args=args,
)
```

---

### CR-03: Tag filtering drops all rollouts — `"tag"` field never set on pool items

**File:** `scripts/rl_rollouts.py:439-440`

**Issue:** `collect_rollouts` splits the raw batch into `gen_rollouts` and `judge_rollouts` by filtering on `item.get("tag")`. But `load_rl_prompts()` in `tinker_rl_data.py` returns dicts of shape `{"messages": [...]}` with no `"tag"` key. Both filtered lists are always empty. All batch items are silently discarded; no completions are ever generated.

```python
# L439-440 — WRONG
gen_rollouts = [item for item in batch if item.get("tag") == "gen"]    # always []
judge_rollouts = [item for item in batch if item.get("tag") == "judge"] # always []

# FIX: tag must be set in load_rl_prompts or in sample_interleaved_prompts
# Option A — in tinker_rl_data.py, add tag to each returned dict:
#   return {"messages": turns, "tag": pool}   # pool = "gen" | "judge"
# Option B — in collect_rollouts, track which items came from which pool
#   via a separate index list returned by sample_interleaved_prompts.
```

---

### CR-04: `optim_step()` commits weight update before halt check

**File:** `scripts/rl_train.py:686,712`

**Issue:** `tc.optim_step()` runs at L686. KL is computed at L688-707. `check_halt` runs at L712. When halt triggers (KL > 0.3 or e_frac < 0.5), the divergent weight update has already been committed. The "emergency checkpoint" at L743 then saves these divergent weights under `emergency-halt-step-N`, which will be the most recent checkpoint and may be loaded for recovery. Additionally, KL computation failure at L701-707 silently substitutes `0.0` for both KL values — a compute error reads as "perfect KL, no halt needed," disabling the guard entirely.

```python
# Current (WRONG) order:
tc.optim_step()     # L686 — commits update
kl = compute_kl()  # L688-707
halt = check_halt() # L712 — too late

# FIX: compute KL from fb_out metrics, check halt, THEN commit
kl_metrics = _extract_kl_from_fb_out(fb_out)   # no exception path
halt_reason = check_halt(
    kl_v1=kl_metrics["optim/kl_sample_train_v1"],
    ...
)
if halt_reason is not None:
    logger.error("HALT at step %d: %s", step, halt_reason)
    _save_checkpoint(tc, name=f"pre-halt-step-{step}", manifest=manifest)
    raise RuntimeError(f"Training halted at step {step}: {halt_reason}")
tc.optim_step()  # only reached when safe
```

---

### CR-05: MO-GRPO pre-normalization feeds z-scores into `combine_judge_reward`

**File:** `scripts/rl_rollouts.py:468-495`

**Issue:** `_mo_grpo_norm(fix_arr)` standardizes `fix_correctness` values to z-scores (mean=0, std=1, range roughly [-3, +3]). The normalized values are then passed as `fix_correctness` to `combine_judge_reward(fix_correctness=fix_norm[i], consistency=consistency_scores[i])`. `consistency_scores[i]` is raw [0,1]. The combined reward `(1-0.3)*z_score + 0.3*consistency` can be negative (e.g., z=-1.414, consistency=0.5 → combined=-0.840) even for groups where all rewards are positive. This corrupts every advantage computation and violates the D-09-05 reward cap logic (which assumes `fix_correctness ∈ [0,1]`).

```python
# L468-473 — WRONG
fix_arr = np.array(fix_scores)
fix_norm = _mo_grpo_norm(fix_arr).tolist()  # z-scores, NOT [0,1]
...
combined_scalar = combine_judge_reward(
    fix_correctness=fix_norm[i],   # can be negative or >1
    consistency=consistency_scores[i],
)

# FIX: normalize AFTER combining, or pass raw fix_correctness to combine_judge_reward
# and apply _mo_grpo_norm to the final combined scalar across the group:
combined_scores = [
    combine_judge_reward(fix_correctness=fc, consistency=cs)
    for fc, cs in zip(fix_scores, consistency_scores)
]
combined_norm = _mo_grpo_norm(np.array(combined_scores)).tolist()
```

---

### CR-06: `_inline_remove_constant_reward_groups` filters on entire batch, not per-prompt

**File:** `scripts/rl_rollouts.py:264-273`

**Issue:** Per-prompt constant-reward filtering must check whether all completions for a **single prompt** have identical rewards (zero advantage — no learning signal). The inline implementation receives `flat_groups` (a flat list of all rollouts across all prompts) and checks `len(set(all_rewards)) <= 1` on the whole batch. If any two rollouts happen to share the same reward value, or if the batch contains only one unique reward across all prompts, the entire batch is dropped. Conversely, a prompt where all N completions have the same reward passes through unflagged because other prompts introduce reward diversity. This is structurally wrong regardless of the JSONL cookbook availability.

```python
# L264-273 — WRONG: treats entire batch as one group
def _inline_remove_constant_reward_groups(groups):
    all_rewards = [g["reward"] for g in groups]
    if len(set(all_rewards)) <= 1:
        return []
    return groups

# FIX: filter per prompt_id group
def _inline_remove_constant_reward_groups(groups):
    by_prompt = {}
    for g in groups:
        by_prompt.setdefault(g["prompt_id"], []).append(g)
    result = []
    for prompt_groups in by_prompt.values():
        rewards = {g["reward"] for g in prompt_groups}
        if len(rewards) > 1:
            result.extend(prompt_groups)
    return result
```

---

### CR-07: Always-inline fallback — unconditional `raise ImportError` bypasses cookbook permanently

**File:** `scripts/rl_rollouts.py:368`

**Issue:** Inside the `try` block that imports `tinker_cookbook.rl.data_processing`, line 368 unconditionally raises `ImportError("tinker absent or cookbook types unavailable — use inline path")`. This was scaffolding that was never removed. The cookbook import always falls through to the inline path, meaning the `_inline_remove_constant_reward_groups` bug (CR-06) and other inline limitations are always active even when Tinker is installed and the cookbook is available.

```python
# L364-370 — WRONG
try:
    from tinker_cookbook.rl.data_processing import (
        compute_advantages, remove_constant_reward_groups, assemble_training_data,
    )
    raise ImportError("tinker absent or cookbook types unavailable — use inline path")  # ALWAYS fires
except (ImportError, Exception):
    ...  # inline path always used

# FIX: remove the unconditional raise
try:
    from tinker_cookbook.rl.data_processing import (
        compute_advantages, remove_constant_reward_groups, assemble_training_data,
    )
    # use cookbook path here directly
    ...
except (ImportError, AttributeError):
    # genuine fallback only when cookbook unavailable
    ...
```

---

## Warnings

### WR-01: `_res()` bare `except Exception: pass` swallows real Tinker errors

**File:** `scripts/rl_train.py:71-72`

**Issue:** When `tinker` is installed and `f` is a genuine `APIFuture`, `f.result()` may raise `tinker.TinkerError` (network failure, quota exceeded, etc.). The bare `except Exception: pass` catches this, falls through, and returns the unresolved `APIFuture` object as if it were the result. Callers then receive a `Future` instead of the expected value. For example, `sampling_client = _res(tc.save_weights_for_sampler(...))` would silently get back a Future on error, and the subsequent `sampling_client.generate()` call would produce a confusing `AttributeError` rather than the original `TinkerError`.

```python
# L65-74 — WRONG
try:
    import tinker as _tinker
    if isinstance(f, _tinker.APIFuture):
        return f.result()
    return f
except Exception:   # swallows TinkerError from f.result()
    pass
return f

# FIX: only catch ImportError for the tinker import itself
try:
    import tinker as _tinker
except ImportError:
    return f   # offline path: return as-is
if isinstance(f, _tinker.APIFuture):
    return f.result()  # let TinkerError propagate
return f
```

---

### WR-02: Cache key collision in `rl_judge_dispatch.py`

**File:** `scripts/rl_judge_dispatch.py:75`

**Issue:** The cache key is `sha256(php_code[:512] + consistency_text[:512])`. String concatenation before hashing causes collisions: `key("ab", "cd") == key("a", "bcd")` — confirmed identical SHA-256. Cache hits return stale results for different inputs. Likelihood increases as `php_code` or `critique_text` approaches 512 characters (truncation point).

```python
# WRONG
digest = hashlib.sha256((php_code[:512] + critique_text[:512]).encode()).hexdigest()

# FIX: use a separator that cannot appear in either input, or hash each field separately
import json
digest = hashlib.sha256(
    json.dumps([php_code[:512], critique_text[:512]]).encode()
).hexdigest()
```

---

### WR-03: Thread leak in `score_judge_consistency_batch` on timeout

**File:** `scripts/rl_judge_dispatch.py:211-221`

**Issue:** `asyncio.wait_for(loop.run_in_executor(None, lambda: score_judge_consistency(...)), timeout=120)` — when the 120s asyncio timeout fires, the underlying thread continues running (executor threads are not cancellable). `score_judge_consistency` calls `subprocess.run(..., timeout=300)` internally — meaning the thread can persist for up to 300 additional seconds per N-vote call after asyncio has already moved on. Under concurrent batch scoring with many timeouts, this leaks thread-pool slots and delays Python process exit.

```python
# FIX: use a threading.Event to signal the worker, or reduce subprocess timeout
# to match the asyncio deadline; at minimum, cap subprocess timeout:
score = await asyncio.wait_for(
    loop.run_in_executor(None, lambda: score_judge_consistency(
        ..., timeout=110  # leave 10s margin below asyncio timeout
    )),
    timeout=120,
)
```

---

### WR-04: Panickssery D-09-05 R1 self-preference monitor is completely inert

**File:** `scripts/rl_train.py:447-453`

**Issue:** `_panickssery_spot_check` filters `breakdown` with `if not isinstance(bd, dict): continue`. But `build_trajectory_groups` (in `rl_rollouts.py:242-247`) stores `breakdown` as a `_JudgeBreakdown` object or `types.SimpleNamespace` — never a plain `dict`. Every item in `data` is skipped. The divergence log (`|fix_correctness - consistency| > 0.3`) never fires regardless of actual training dynamics. D-09-05 R1 is a mandatory monitoring requirement; a monitor that always produces empty output is functionally absent.

```python
# L447-449 — WRONG
bd = item.get("breakdown") or {}
if not isinstance(bd, dict):    # always True for _JudgeBreakdown / SimpleNamespace
    continue

# FIX: access attributes directly
bd = item.get("breakdown")
if bd is None:
    continue
fix_corr = getattr(bd, "fix_correctness", getattr(bd, "fix_score", None))
consistency = getattr(bd, "judge_consistency", getattr(bd, "consistency", None))
if fix_corr is not None and consistency is not None:
    if abs(float(fix_corr) - float(consistency)) > 0.3:
        divergent.append(...)
```

---

### WR-05: SKILL.md manifest reader has `AttributeError` on `m["started_at"].get(...)`

**File:** `.claude/skills/wp-finetune:run-rl-training/SKILL.md` (Step 9 code block)

**Issue:** The step-9 manifest inspection snippet does:
```python
keys = list(m.keys())        # e.g. ['checkpoints', 'run_args', 'started_at']
m[keys[-1]].get('sampler_path', '...')
```
`keys[-1]` is `'started_at'`, which maps to an ISO timestamp string. Calling `.get(...)` on a string raises `AttributeError: 'str' object has no attribute 'get'`. The step-6 snippet has the same issue: `list(m.keys())[-3:]` returns top-level manifest keys, not the last 3 checkpoint entries.

**Fix:** The Step 9 snippet should iterate `m["checkpoints"]`, not use `m[keys[-1]]`:
```python
checkpoints = m.get("checkpoints", [])
for ck in checkpoints[-3:]:
    print(ck.get("sampler_path", "n/a"), ck.get("name"))
```

---

## Info

### IN-01: `test_security_zero_group_dropped` is vacuous — no assertion

**File:** `tests/test_rl_rollouts.py:197-229`

**Issue:** The test body ends without any `assert` statement. It exercises `build_trajectory_groups` with a single security-failed rollout, catches no assertion failure, and reports as passed regardless of whether the rollout was actually dropped. The correct behavior (T-09-SECDROP: drop entirely, not zero) is not verified by this test.

**Fix:** Add: `assert len(groups) == 0, f"Expected 0 groups after security drop, got {len(groups)}"`

---

### IN-02: Unguarded `row["messages"][0]["content"]` in `build_rl_prompts.py`

**File:** `scripts/build_rl_prompts.py:100,134`

**Issue:** Both access sites assume `messages` is non-empty and has a `"content"` key. Malformed rows from upstream JSONL emit `IndexError` (empty list) or `KeyError` (missing key) with no context on which row failed. `get()` with a default would be safer and more debuggable.

**Fix:**
```python
content = (row.get("messages") or [{}])[0].get("content", "")
if not content:
    logger.warning("Skipping row with missing content: %s", row.get("id", "?"))
    continue
```

---

### IN-03: Val-leakage PROVENANCE claim says "Assertion" but no code assertion exists

**File:** `scripts/build_rl_prompts.py` (PROVENANCE block, line ~290)

**Issue:** The generated PROVENANCE.md includes the line `"Assertion: NO val-set sha256 in output"`. The code does drop matching rows via `continue` (the actual guard is correct), but no `assert val_leak_dropped == 0` exists. If the upstream dataset accidentally collides, the script silently continues and the PROVENANCE.md claim of "Assertion" is misleading. A warning log when `val_leak_dropped > 0` would close this gap.

**Fix:**
```python
if val_leak_dropped > 0:
    logger.warning(
        "Val-leakage guard dropped %d rows. PROVENANCE 'Assertion' may mislead — "
        "consider assert val_leak_dropped == 0 if zero leakage is required.",
        val_leak_dropped,
    )
```

---

## Verified Clean

The following were explicitly checked and found correct:

- **No hardcoded credentials** — `ServiceClient()` reads `~/.tinker` or env. No API keys in any reviewed file.
- **No Anthropic API import** — `rl_judge_dispatch.py` uses `scripts.claude_agent.generate_json` (subprocess path only). Confirmed by AST in `test_no_anthropic_api_import`.
- **No `run_in_background` in judge dispatch** — dispatches are blocking calls, not background agents. Confirmed by source grep.
- **`judge_consistency_weight` cap asserted** — module-level `assert judge_consistency_weight <= 0.5` at `rl_rollouts.py` import time. Current value is `0.3`.
- **Persistent checkpoints** — all `save_weights_for_sampler` calls use `ttl_seconds=None`. Confirmed at L375 and L658.
- **`reward_pipeline.py` unmodified** — MO-GRPO normalization is inside the pipeline (Phase 8). Not re-normalized upstream. Git last-touch is Phase 8 commit `b202d38`.
- **RSPO floor matches spec** — `rspo_floored_ratio` implements `ratio.clamp(min=1.0)` per locked D-09-03 decision. No deviation from spec.
- **Router gates frozen** — LoRA config sets `train_mlp=True`, `train_attn=True`, `train_unembed=True`, no `train_router` argument. Confirmed by `test_lora_config`.

---

_Reviewed: 2026-06-20T09:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
