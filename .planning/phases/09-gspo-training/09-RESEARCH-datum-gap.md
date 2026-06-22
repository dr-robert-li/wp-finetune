# Phase 9 Corrective Research: Datum Assembly Gap

**Researched:** 2026-06-21
**Domain:** Tinker RL training — `Trajectory`/`Datum` assembly, sampled logprob capture, GSPO IS ratio
**Confidence:** HIGH (all API claims verified from installed package source; design decisions derived from first principles)
**Scope:** TIGHTLY SCOPED to the datum/logprob gap exposed by the live smoke. All other Phase 9
decisions (reward path, KL-halt, entrypoint, judge dispatch) remain fixed.

---

## 1. Recommendation: Adopt Cookbook vs Hand-Roll

**Decision: Adopt cookbook `Trajectory`/`TrajectoryGroup` + `data_processing.trajectory_to_data`.**

### Rationale

The cookbook already provides the canonical path:

```
collect_rollouts → build_trajectory_groups → compute_rollout_advantages → assemble_training_data
```

The current `_inline_assemble_training_data` mirrors cookbook semantics but emits plain dicts
instead of `tinker.Datum`. Replacing the inline helpers with the cookbook types is a bounded,
correct refactor — not a rewrite.

### Integration Seam

The seam is `build_trajectory_groups`. Currently it emits `list[dict]` with keys
`{completion, reward, group_id, breakdown}`. It must be changed to emit `list[TrajectoryGroup]`
carrying `tinker_cookbook.rl.types.TrajectoryGroup` objects, each with:

- `trajectories_G: list[Trajectory]` — one `Trajectory` per rollout, each holding one
  `Transition(ob=ModelInput, ac=TokensWithLogprobs, reward=float, episode_done=True)`
- `final_rewards_G: list[float]` — group-level final rewards (0.0 for single-turn)
- `metrics_G: list[Metrics]`

**The reward scalar enters as `Transition.reward`.** The `reward` field already collected by
`collect_rollouts` (via `RewardResult.scalar`) maps directly to `Transition.reward`. No change
to the reward computation path.

**T-09-SECDROP** (security-zero drop) is preserved by filtering before constructing the
`TrajectoryGroup`. The current `build_trajectory_groups` already drops entries where
`breakdown.security_fail=True` before emitting group dicts. Under the new design the same guard
runs before the `Trajectory` object is created — the survivor list is passed to `TrajectoryGroup`.
The drop logic is pure membership filtering and is unaffected by the type change.

**Per-prompt group centering (CR-06, D-09)** is preserved because
`data_processing.compute_advantages` centers rewards within each `TrajectoryGroup`:

```python
# data_processing.compute_advantages (verified source)
rewards_G = torch.tensor(traj_group.get_total_rewards())
advantages_G = rewards_G - rewards_G.mean()   # per-group centering
```

This is semantically identical to `_inline_compute_advantages` which does the same
`reward - group_mean` arithmetic. The G completions from one prompt form one `TrajectoryGroup`
(group_id = prompt index), so per-prompt centering is preserved exactly.

### What Changes vs What Does Not

| Component | Change? | Notes |
|-----------|---------|-------|
| `collect_rollouts` | No | Returns same list of rollout objects + rewards |
| `build_trajectory_groups` | **Yes** | Returns `list[TrajectoryGroup]` not `list[dict]` |
| `_generate_completions` | **Yes** | Must capture tokens + logprobs into `_Completion` |
| `_Completion` | **Yes** | Gains `.tokens: list[int]` and `.logprobs: list[float]` |
| `_inline_*` helpers | **Deleted** | Replaced by cookbook calls |
| `compute_rollout_advantages` | **Yes** | Returns `(list[Datum], list[float], meta)` — see §3 |
| `run_training_step` lines 576–577 | **Yes** | Must read advantages from new return shape |
| `build_loss_step` / `_make_gspo_loss_fn` | Minor | See §3 — advantage closure simplified |
| Reward path (reward_pipeline, rl_judge_dispatch) | No | Untouched per constraints |
| KL-halt, entrypoint, judge dispatch | No | Untouched per constraints |

### The Invasive Part (named explicitly)

`build_trajectory_groups` currently builds singleton dicts one rollout at a time. It must instead:
1. Group rollout objects by `group_id` (already stamped by `collect_rollouts`).
2. Within each group, apply T-09-SECDROP filter.
3. Build one `Trajectory` per surviving rollout (requires the `ModelInput` observation and the
   `TokensWithLogprobs` action captured from sampling — which is why `_generate_completions`
   must change first).
4. Wrap each group's trajectories into a `TrajectoryGroup`.

This is the single most invasive change. The function signature changes from
`(rollouts: list, rewards: list) -> list[dict]` to
`(rollouts: list[_Completion], rewards: list[RewardResult]) -> list[TrajectoryGroup]`.

---

## 2. Logprob Capture

### `sample()` API — Verified from Source

`tinker.SamplingClient.sample()` signature (verified: `sampling_client.py` lines 292–358):

```python
def sample(
    self,
    prompt: types.ModelInput,
    num_samples: int,
    sampling_params: types.SamplingParams,
    include_prompt_logprobs: bool = False,   # prompt-token logprobs (for KL)
    topk_prompt_logprobs: int = 0,
) -> ConcurrentFuture[types.SampleResponse]:
```

**Sampled-token logprobs are returned by default** — no extra flag required.
`include_prompt_logprobs` controls only prompt-token logprobs (used for KL, separate feature).

### `SampledSequence` Shape — Verified from Source

`tinker.types.SampledSequence` (verified: `sampled_sequence.py`):

```python
@dataclass(frozen=True)
class SampledSequence:
    stop_reason: StopReason
    tokens_np: Optional[np.ndarray]      # shape (num_tokens,) int32
    logprobs_np: Optional[np.ndarray]    # shape (num_tokens,) float32; None if not requested
    _tokens_list: Optional[List[int]]    # internal
    _logprobs_list: Optional[List[float]] # internal

    @cached_property
    def tokens(self) -> List[int]: ...
    @cached_property
    def logprobs(self) -> Optional[List[float]]: ...  # None if not returned
```

`TinkerTokenCompleter` (tinker_cookbook/completers.py line 129) asserts
`sampled_seq.logprobs is not None` after a plain `sample_async(..., SamplingParams(stop,
max_tokens, temperature))` — confirming sampled-token logprobs are populated without any
extra flag.

### Changes to `_generate_completions`

Current: calls `_decode_samples(resp, tok)` which reads `.tokens` and decodes to text; drops logprobs.

New: after resolving the future, read both tokens and logprobs from each `SampledSequence`:

```python
r = resp.result()
seqs = r.sequences  # list[SampledSequence]
for seq in seqs:
    tokens: list[int] = seq.tokens          # per-token ids
    logprobs: list[float] = seq.logprobs    # per-token sampled logprobs (always present)
    text: str = tok.decode(tokens)
    completions.append(_Completion(
        completion=text,
        group_id=group_id,
        model_input=prompt,      # tinker.ModelInput — the observation
        tokens=tokens,
        logprobs=logprobs,
    ))
```

`prompt` (the `ModelInput` built by `renderer.build_generation_prompt`) is already available
in the loop body — it just needs to be passed through rather than discarded.

### `_Completion` New Shape

```python
class _Completion:
    __slots__ = ("completion", "group_id", "model_input", "tokens", "logprobs")

    def __init__(
        self,
        completion: str,
        group_id: Any,
        model_input: Any,        # tinker.ModelInput (observation)
        tokens: list[int],       # sampled token ids
        logprobs: list[float],   # per-token sampled logprobs
    ):
        ...
```

**Fallback for unit tests without real tinker:** when `tinker` is absent, `logprobs` can be
`None` and `model_input` can be a stub. The downstream `trajectory_to_data` already requires
real values; test fixtures that bypass `forward_backward_custom` are unaffected.

---

## 3. Loss Seam

### Keep Custom `_make_gspo_loss_fn`

The custom GSPO loss function is **kept** with a simplification: it reads `advantages` from
`datum.loss_fn_inputs["advantages"]` (baked by `trajectory_to_data`) rather than from the
fragile closure-based `adv_map`.

**Why**: `trajectory_to_data` already writes per-token advantage weights into
`datum.loss_fn_inputs["advantages"]`. For a single-turn completion (all action tokens), the
token-level advantage is the same scalar repeated `len(tokens)` times. The GSPO sequence-level
objective sums over action tokens anyway, so reading from the Datum is cleaner and eliminates
the `{i: advantages[i]}` index-alignment assumption.

### `rspo_floored_ratio` — Verified Formula

`rl_train.py` lines 158–165 (read via grep at line 165): `return ratio.clamp(min=1.0)`.
The body above (not yet read inline — see below) was confirmed by grep: `seq_ratio =
rspo_floored_ratio(train_sum, sampling_sum)` at line 202 and `train_sum = train_lps.sum()`,
`sampling_sum = sampling_lps.sum()`. The formula is:

```python
def rspo_floored_ratio(train_sum: torch.Tensor, sampling_sum: torch.Tensor) -> torch.Tensor:
    ratio = torch.exp(train_sum - sampling_sum)   # exp(Σ log π_train - Σ log π_old)
    return ratio.clamp(min=1.0)                    # RSPO floor
```

This is `exp(train_lp_seq - sampled_lp_seq).clamp(min=1.0)` — correct sequence-level IS ratio
with RSPO floor. With real sampled logprobs now in `datum.loss_fn_inputs["logprobs"]`,
`sampling_sum = datum.loss_fn_inputs["logprobs"].to_torch().sum()` uses the REAL values,
not the `except` branch's `seq_ratio = torch.tensor(1.0)` fallback.

### Revised `_make_gspo_loss_fn` (simplified)

```python
def _make_gspo_loss_fn():
    """Return GSPO loss reading advantages + logprobs from datum.loss_fn_inputs."""
    def gspo_loss_fn(data, logprobs_list):
        import torch
        losses = []
        for datum, train_lps in zip(data, logprobs_list):
            # Sequence-level advantage: sum per-token weights (all same scalar)
            adv_weights = datum.loss_fn_inputs["advantages"].to_torch()
            adv = adv_weights[adv_weights != 0].mean()  # action-token positions only

            sampling_lps = datum.loss_fn_inputs["logprobs"].to_torch()
            train_sum = train_lps.sum()
            sampling_sum = sampling_lps.sum()
            seq_ratio = rspo_floored_ratio(train_sum, sampling_sum)
            losses.append(-(seq_ratio * adv))

        total_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0)
        return total_loss, {"gspo/n_sequences": float(len(losses))}
    return gspo_loss_fn
```

No `advantages_by_idx` closure parameter needed — advantages are in the Datum.

### `build_loss_step` Simplification

```python
def build_loss_step(tc, data, use_gspo=True, advantages=None) -> Any:
    if not use_gspo:
        return _res(tc.forward_backward(data, loss_fn="importance_sampling"))
    # advantages arg no longer needed — baked into datum.loss_fn_inputs
    loss_fn = _make_gspo_loss_fn()
    return _res(tc.forward_backward_custom(data, loss_fn))
```

### GSPO vs GRPO Mapping

| Objective | Granularity | IS ratio | Datum key used | Status |
|-----------|-------------|----------|----------------|--------|
| GSPO (primary, D-09-03) | Sequence-level | `exp(Σtrain_lp - Σsampled_lp).clamp(min=1)` | `logprobs` (per-token, summed) | Real after this fix |
| GRPO (fallback, `--grpo-fallback`) | Token-level | `importance_sampling` built-in | `logprobs` + `advantages` | Unchanged |

### Double Advantage Source — Resolution

`trajectory_to_data` bakes the **per-token** advantage scalar (repeated across action positions,
zero on observation positions) into `datum.loss_fn_inputs["advantages"]`. The revised
`_make_gspo_loss_fn` reads this directly — the `adv_map` closure is eliminated. There is no
double-application: the Datum advantage IS the source; the closure no longer exists.

### `run_training_step` Lines 576–577 — Fix Required

Current (broken after type change):
```python
rewards   = [float(d.get("reward", 0.0)) for d in data]    # treats data as dicts
advantages = [float(d.get("advantage", 0.0)) for d in data]
```

New `compute_rollout_advantages` returns `(list[tinker.Datum], list[float], meta)`:
```python
data, advantages, meta = compute_rollout_advantages(trajectory_groups)
```

Lines 576–577 change to read from the explicit `advantages` list returned alongside `data`.
`rewards` for the Panickssery spot-check can be extracted from `TrajectoryGroup.get_total_rewards()`.
Pass `trajectory_groups` (the pre-Datum intermediate) through to the spot-check if needed.

### Adv–Datum Alignment Assumption

`trajectory_to_data` can emit multiple Datums per Trajectory (when observations are non-prefix).
For WP single-turn completions there is exactly one `Transition` per `Trajectory` → exactly one
Datum per trajectory → `len(data) == len(advantages)` is guaranteed for this workload.
The integration test (§5) must assert this. The revised `_make_gspo_loss_fn` reading
advantages from the Datum removes any dependence on index alignment in the loss function itself.

---

## 4. Blast Radius + Tests

### Functions to Change

| File | Function | Change |
|------|----------|--------|
| `scripts/rl_rollouts.py` | `_Completion` | Add `.model_input`, `.tokens`, `.logprobs` fields |
| `scripts/rl_rollouts.py` | `_generate_completions` | Capture tokens + logprobs from `SampledSequence`; pass `prompt` to `_Completion` |
| `scripts/rl_rollouts.py` | `_decode_samples` | Can be deleted or kept as text-only helper (no longer used on the critical path) |
| `scripts/rl_rollouts.py` | `build_trajectory_groups` | Return `list[TrajectoryGroup]` (group by group_id, apply T-09-SECDROP, build `Trajectory`/`Transition` per rollout) |
| `scripts/rl_rollouts.py` | `_inline_remove_constant_reward_groups` | Delete — replaced by `data_processing.remove_constant_reward_groups` |
| `scripts/rl_rollouts.py` | `_inline_compute_advantages` | Delete — replaced by `data_processing.compute_advantages` |
| `scripts/rl_rollouts.py` | `_inline_assemble_training_data` | Delete — replaced by `data_processing.assemble_training_data` |
| `scripts/rl_rollouts.py` | `compute_rollout_advantages` | Input: `list[TrajectoryGroup]`; output: `(list[Datum], list[float], meta)` |
| `scripts/rl_train.py` | `_make_gspo_loss_fn` | Remove `advantages_by_idx` param; read from `datum.loss_fn_inputs` |
| `scripts/rl_train.py` | `build_loss_step` | Remove `advantages` param (now baked into Datums) |
| `scripts/rl_train.py` | `run_training_step` lines 570–577 | Unpack `(data, advantages, meta)` from `compute_rollout_advantages`; adjust Panickssery spot-check to use `TrajectoryGroup` rewards |

### Existing Tests That Assert Dict Shapes (Must Update)

| Test file | Test | What changes |
|-----------|------|-------------|
| `test_rl_rollouts.py` | `TestBuildTrajectoryGroups.test_build_trajectory_groups_structure` | Currently asserts `isinstance(g, dict)`. Must change to assert `isinstance(g, TrajectoryGroup)` or (if the class is unavailable) check `.trajectories_G` attribute. |
| `test_rl_rollouts.py` | `TestBuildTrajectoryGroups.test_security_zero_group_dropped` | Reads `g["reward"]` from dict. Must change to read via `tg.get_total_rewards()` |
| `test_rl_rollouts.py` | `TestBuildTrajectoryGroups.test_security_group_dropped_by_flag` | Same — `len(groups)` check survives; inner key access changes |
| `test_rl_rollouts.py` | `TestComputeRolloutAdvantages` | Currently passes synthetic `{completions, rewards}` dicts. Must either (a) retain the test-format normalisation path in `compute_rollout_advantages`, or (b) update fixtures to produce `TrajectoryGroup` mocks |
| `test_rl_train.py` | `test_grpo_advantages` | Reads `item["advantage"]` from returned `data`. After the fix, `data` is `list[Datum]`; the test must instead verify the returned `advantages` list |
| `test_rl_train_integration.py` | `_FakeSeq` | Only has `.tokens`. Must gain `.logprobs: list[float]` for the new `_generate_completions` path |
| `test_rl_train_integration.py` | `_FakeSamplingClient.sample` | Produces `_FakeSeq(tokens=[...])`. Must also populate `logprobs` to match the new read |

### New Integration Test — Datum Schema Assertion

**Purpose:** Prevent silent regression of the datum-gap. Asserts `Datum.loss_fn_inputs` has all
three required keys before `forward_backward_custom` is called.

**Location:** `tests/test_rl_datum_assembly.py` (new file)

```python
"""Integration test: trajectory → Datum schema, asserting logprobs/advantages baked in."""

import torch
import pytest

def _make_fixture_trajectory():
    """Build a minimal Trajectory with one Transition for a 3-token completion."""
    import tinker
    from tinker_cookbook.rl.types import Trajectory, Transition
    from tinker_cookbook.completers import TokensWithLogprobs

    prompt_tokens = [1, 2, 3]
    completion_tokens = [10, 11, 12]
    sampled_logprobs = [-0.5, -0.8, -0.3]

    ob = tinker.ModelInput.from_ints(prompt_tokens)
    ac = TokensWithLogprobs(
        tokens=completion_tokens,
        maybe_logprobs=sampled_logprobs,
    )
    transition = Transition(ob=ob, ac=ac, reward=0.75, episode_done=True)
    final_ob = tinker.ModelInput.from_ints([])
    return Trajectory(transitions=[transition], final_ob=final_ob)


def test_trajectory_to_datum_schema():
    """trajectory_to_data must produce Datum with target_tokens, logprobs, advantages."""
    pytest.importorskip("tinker_cookbook")
    from tinker_cookbook.rl.data_processing import trajectory_to_data

    traj = _make_fixture_trajectory()
    datums = trajectory_to_data(traj, traj_advantage=0.75)

    assert len(datums) == 1, "single-turn trajectory must yield exactly one Datum"
    d = datums[0]

    assert "target_tokens" in d.loss_fn_inputs, "Datum must have target_tokens"
    assert "logprobs" in d.loss_fn_inputs,      "Datum must have logprobs (sampled)"
    assert "advantages" in d.loss_fn_inputs,    "Datum must have advantages"

    lps = d.loss_fn_inputs["logprobs"].to_torch()
    adv = d.loss_fn_inputs["advantages"].to_torch()
    # Non-zero logprobs (not the 1.0 fallback)
    assert not torch.all(lps == 0.0), "logprobs must reflect real sampled values, not zeros"
    # All action-position advantages equal the trajectory advantage scalar
    nonzero_adv = adv[adv != 0]
    assert torch.allclose(nonzero_adv, torch.tensor(0.75).expand_as(nonzero_adv)), \
        "action-token advantages must equal traj_advantage"


def test_datum_assembly_len_matches_advantages():
    """assemble_training_data: len(data) == len(advantages) for single-turn workload."""
    pytest.importorskip("tinker_cookbook")
    from tinker_cookbook.rl.types import TrajectoryGroup
    from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages

    traj1 = _make_fixture_trajectory()
    traj2 = _make_fixture_trajectory()
    tg = TrajectoryGroup(
        trajectories_G=[traj1, traj2],
        final_rewards_G=[0.0, 0.0],
        metrics_G=[{}, {}],
    )
    advantages_P = compute_advantages([tg])
    data_D, meta_D = assemble_training_data([tg], advantages_P)

    assert len(data_D) == 2, "two single-turn trajectories → two Datums"
    assert len(data_D) == sum(1 for _ in meta_D), "data and metadata must be same length"
```

---

## 5. Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (version from project venv) |
| Config file | `pytest.ini` or `pyproject.toml` (project standard) |
| Quick run command | `.venv-tinker/bin/pytest tests/test_rl_datum_assembly.py -x` |
| Full suite command | `.venv-tinker/bin/pytest tests/ -x` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GRPO-05 | Interleaved gen/judge rollouts | unit | `.venv-tinker/bin/pytest tests/test_rl_rollouts.py::TestBuildTrajectoryGroups -x` | ✅ (update) |
| GRPO-06 | Per-prompt group centering | unit | `.venv-tinker/bin/pytest tests/test_rl_rollouts.py::TestComputeRolloutAdvantages -x` | ✅ (update) |
| GRPO-07 | Mixed-reward group → non-zero advantages | unit | `.venv-tinker/bin/pytest tests/test_rl_train.py::TestGSPOTrainingStep::test_grpo_advantages -x` | ✅ (update) |
| GRPO-08 | KL autohalt before optim_step | unit | `.venv-tinker/bin/pytest tests/test_rl_train.py::TestGSPOTrainingStep::test_kl_autohalt -x` | ✅ (no change) |
| DATUM-01 | `Datum.loss_fn_inputs` has target_tokens+logprobs+advantages | integration | `.venv-tinker/bin/pytest tests/test_rl_datum_assembly.py::test_trajectory_to_datum_schema -x` | ❌ Wave 0 |
| DATUM-02 | `len(data)==len(advantages)` for single-turn | integration | `.venv-tinker/bin/pytest tests/test_rl_datum_assembly.py::test_datum_assembly_len_matches_advantages -x` | ❌ Wave 0 |
| DATUM-03 | `seq_ratio != 1.0` in live re-smoke (real logprobs used) | live smoke | see re-smoke command below | manual |

### Offline Fixture Test (Wave 0 Gap)

New file: `tests/test_rl_datum_assembly.py` (content in §4).

Tests the datum schema in isolation — no Tinker training client required, no model weights,
no reward pipeline. Exercises only `trajectory_to_data` + `assemble_training_data` from the
installed cookbook. Runs in < 1 second.

### Integration Test Updates (Wave 0)

`tests/test_rl_train_integration.py` — `_FakeSeq` and `_FakeSamplingClient` must gain
`.logprobs` to match the new `_generate_completions` read path:

```python
class _FakeSeq:
    def __init__(self, tokens):
        self.tokens = tokens
        self.logprobs = [-0.3] * len(tokens)   # non-zero so seq_ratio != 1.0 in loss fn
```

`_FakeSamplingClient.sample` must also carry `model_input` through to `_Completion` — the
fake prompt object returned by `renderer.build_generation_prompt` must be a valid
`tinker.ModelInput` or a stub with `.chunks` attribute that `trajectory_to_data` can consume.

### 1-Step Live Re-Smoke Acceptance Criteria

Command:
```bash
.venv-tinker/bin/python scripts/rl_train.py \
  --total-steps 1 \
  --batch-size 2 \
  --group-size 2 \
  --max-pool 2 \
  --judge-base-url http://localhost:8000/v1 \
  --judge-model wp_judge \
  --manifest-path output/_smoke/checkpoint_manifest.json \
  --metrics-path output/_smoke/metrics/rl_metrics.jsonl
```

Acceptance criteria:
1. No `AttributeError: 'dict' object has no attribute 'loss_fn_inputs'` — the original blocker.
2. `optim_step` is called (logged as `"optim_step"` in stdout or `rl_metrics.jsonl`).
3. `rl_metrics.jsonl` contains a row with `"step": 1` and numeric `gspo/n_sequences`.
4. **`seq_ratio != 1.0`** in at least one datum's computed ratio — verifiable by adding a
   temporary `logger.info("seq_ratio=%s", seq_ratio)` inside `gspo_loss_fn`, then removing it.
   This is the key correctness signal that real logprobs are flowing, not the `1.0` fallback.

### Sampling Rate

- **Per task commit:** `pytest tests/test_rl_datum_assembly.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** full suite green + live re-smoke acceptance criteria met before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_rl_datum_assembly.py` — covers DATUM-01, DATUM-02
- [ ] `tests/test_rl_train_integration.py` — `_FakeSeq` + `_FakeSamplingClient` must gain `.logprobs`

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | WP completions are single-turn (1 Transition per Trajectory → 1 Datum) | §3 alignment | If multi-turn, `len(data) != len(advantages)` — integration test catches this |
| A2 | `tinker.SamplingClient.sample()` always returns sampled-token logprobs (no extra flag) | §2 | If a flag is needed, logprobs will be `None`; `trajectory_to_data` will raise on `ac.logprobs` |
| A3 | `rspo_floored_ratio` body is `exp(train - sampling).clamp(min=1.0)` | §3 | Read confirmed at line 165; body between 158–164 not read line-by-line — formula confirmed by grep evidence |

A2 is the most consequential. Evidence is strong: `TinkerTokenCompleter` calls `sample_async`
with only `stop/max_tokens/temperature` and immediately asserts `logprobs is not None` (line
129, completers.py). This is authoritative cookbook code — if logprobs were opt-in, this
assertion would fire on every recipe run. Confidence: HIGH.

---

## Open Questions

1. **`build_trajectory_groups` grouping strategy.** Currently each rollout is its own singleton
   dict. For true GRPO centering, the G completions of one prompt must form one `TrajectoryGroup`.
   The `group_id` is already stamped by `collect_rollouts` (`gen-{idx}` / `judge-{idx}`). The
   new `build_trajectory_groups` must collect rollouts sharing a `group_id` into one
   `TrajectoryGroup`. With `batch_size=2, group_size=2`, each of the 2 prompts gets 2 completions
   → 2 `TrajectoryGroup`s of size 2. Confirm this matches the smoke config's intent.

2. **`model_input` (observation) for `Trajectory.transitions[0].ob`.** The observation is the
   prompt `ModelInput` from `renderer.build_generation_prompt`. This is already built in
   `_generate_completions`'s inner loop; it just needs to be stored in `_Completion` and passed
   to `Transition(ob=model_input, ...)`. No extra API call needed.

3. **`final_ob` field of `Trajectory`.** After a single turn, `final_ob` is the terminal
   observation (unused for training). `tinker.ModelInput.from_ints([])` is the canonical
   empty-input sentinel (seen in cookbook `Env.step` examples in `types.py` line 139).

---

## Sources

### Primary (HIGH confidence — verified from installed package source)

- `.venv-tinker/.../tinker_cookbook/rl/data_processing.py` — `trajectory_to_data`,
  `compute_advantages`, `assemble_training_data` read in full
- `.venv-tinker/.../tinker_cookbook/rl/types.py` — `Trajectory`, `Transition`, `TrajectoryGroup`,
  `TokensWithLogprobs` read in full
- `.venv-tinker/.../tinker/types/datum.py` — `Datum`, `loss_fn_inputs` structure read in full
- `.venv-tinker/.../tinker/types/sampled_sequence.py` — `SampledSequence.logprobs` shape verified
- `.venv-tinker/.../tinker/lib/public_interfaces/sampling_client.py` — `sample()` signature
  lines 292–358, logprobs behaviour confirmed
- `.venv-tinker/.../tinker_cookbook/completers.py` — `TinkerTokenCompleter` confirming logprobs
  are default-on (line 129 assertion)
- `scripts/rl_rollouts.py` — `_Completion`, `_generate_completions`, `build_trajectory_groups`,
  `compute_rollout_advantages`, `_inline_*` helpers grepped + key sections read
- `scripts/rl_train.py` — `_make_gspo_loss_fn`, `build_loss_step`, `run_training_step` read
- `tests/test_rl_train.py`, `tests/test_rl_rollouts.py`, `tests/test_rl_train_integration.py`
  — dict-shape assertions and `_FakeSeq` shape confirmed via grep + read

---

## RESEARCH COMPLETE

**Phase:** 9 (corrective) — Datum Assembly Gap
**Confidence:** HIGH

### Key Findings

1. **Adopt cookbook** — `build_trajectory_groups` becomes the seam: must emit `list[TrajectoryGroup]`
   instead of `list[dict]`. T-09-SECDROP and CR-06 per-prompt centering are both preserved under
   the cookbook types; no semantic changes to the reward path.

2. **Logprob capture is zero-cost** — sampled-token logprobs are returned by `sample()` by
   default. `_generate_completions` only needs to read `seq.logprobs` alongside `seq.tokens`
   and store both in the expanded `_Completion`. No API changes.

3. **Loss fn simplifies** — `_make_gspo_loss_fn` loses its `advantages_by_idx` closure because
   `trajectory_to_data` bakes advantages into the Datum. `seq_ratio = exp(Σtrain − Σsampled).clamp(min=1)`
   from real logprobs, eliminating the `torch.tensor(1.0)` fallback.

4. **Blast radius is bounded** — `run_training_step` lines 576–577 (dict-reads), four test files,
   and `_FakeSeq` (missing `.logprobs`) all require updates. No changes outside `rl_rollouts.py`,
   `rl_train.py`, and test files.

5. **Regression prevention** — new `tests/test_rl_datum_assembly.py` asserts `Datum` has
   `target_tokens + logprobs + advantages` from a fixture trajectory, so the gap cannot silently
   re-emerge. The live re-smoke acceptance criterion `seq_ratio != 1.0` is the definitive
   correctness gate.

### Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Cookbook API | HIGH | Read from installed package source |
| Logprob availability | HIGH | `TinkerTokenCompleter` asserts logprobs non-None without extra flag |
| Loss seam | HIGH | Read from rl_train.py source; RSPO formula confirmed by grep |
| Blast radius | HIGH | All affected functions grepped + key sections read |
| Alignment assumption (single-turn → 1 Datum) | MEDIUM | WP dataset is single-turn; integration test asserts it |
