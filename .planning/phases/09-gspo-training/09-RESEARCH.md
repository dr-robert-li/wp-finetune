# Phase 9: GSPO Training - Research

**Researched:** 2026-06-20
**Domain:** Tinker RL / GSPO / dual-mode group RL on Qwen3-30B-A3B MoE
**Confidence:** HIGH (all primary claims verified from installed SDK source)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-09-01 (LOCKED):** Execution venue = Tinker cloud, NOT DGX-local. `dgx.execute(...)` path is dead. New `wp-finetune:run-rl-training` skill is Tinker-native.
- **D-09-04 (LOCKED):** Interleaved dual-mode training — both `<wp_gen>` and `<wp_judge>` task pathways trained every batch. Budget ~60% judge / ~40% gen (judge-weighted start, tunable).
- **D-09-05 sub-decisions (from discussion):** Judge reward = capped Claude-consistency + anchored deterministic fix-correctness. Score-reasoning consistency scored by Claude Code agents (Agent(run_in_background=true)), NOT Anthropic API. Fix-correctness is the deterministic anchor.
- **Reward pipeline contract (LOCKED):** `scripts/reward_pipeline.py` is Phase 8's deliverable. Phase 9 CONSUMES it via `compute_group_rewards(php_codes, judge_client, judge_model)` → `list[RewardResult]`. Phase 9 does NOT rebuild or modify reward logic.
- **CI-aware gate disposition (LOCKED per D-09 in Phase 7 CONTEXT):** Bootstrap lower bound must clear bar; measured identically on baseline + candidate.

### Claude's Discretion
- D-09-02: Router-training scope — research-gated, resolved by this research (see below).
- D-09-03: GSPO **expressibility/feasibility** on Tinker — research-gated, resolved FEASIBLE by this research (see below). NOTE: only feasibility was discretionary; the "GSPO is primary" disposition itself is LOCKED in CONTEXT.md D-09-03 (GRPO is fallback only if GSPO proves infeasible/unstable — which it did not).
- D-09-05 R1-R3: Panickssery check procedure, reward noise budget, latency mitigation — resolved by this research.
- Precise group sizes (G_gen, G_judge), LR schedule, batch sizes, epoch count.
- KL penalty coefficient and halt thresholds for GRPO-07/08.
- Checkpoint cadence and naming scheme.

### Deferred Ideas (OUT OF SCOPE)
- DGX-local training path.
- Anthropic API usage for judge scoring.
- Router gate training (infeasible on Tinker — see D-09-02 resolution below).
- Pro-GRPO expand-then-prune expert strategy (requires structural MoE control Tinker does not expose).
- Phase 10 eval implementation (RLEV-01/02 fields are log-format requirements, not Phase 9 build items).
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GRPO-05 | Dual-mode RL: both wp_gen and wp_judge trained every batch; judge ≥ gen budget | Confirmed via cookbook `rl/interleaved.py` + `rl/rollouts.py`; `importance_sampling` loss handles both |
| GRPO-06 | Full-MoE RL (experts + attn + unembed) + protected-expert regularizer; GSPO primary / GRPO fallback | Router gates FROZEN (LoraConfig has no `train_router`); experts trainable via `train_mlp=True`; protected-expert monitor-only via Phase 7 mask |
| GRPO-07 | Router-shift stabilization (RSPO stop-gradient floor for GSPO) | `forward_backward_custom` provides escape hatch; `compute_kl_sample_train` provides per-step policy-shift proxy |
| GRPO-08 | Per-step router-shift monitoring + auto-halt | `ForwardBackwardOutput.metrics` exposes `e_frac_with_tokens:mean`, `e_max_violation:mean/max` per forward pass — native MoE monitoring; `kl_sample_train_v1/v2` for policy-shift halt |
</phase_requirements>

---

## Summary

Phase 9 trains Qwen3-30B-A3B on the RL objective using Tinker cloud as the exclusive execution venue. The training loop is dual-mode: every batch interleaves `<wp_gen>` samples (code generation) and `<wp_judge>` samples (code critique/fix), with the judge pathway weighted at approximately 60% of total samples. Reward signals come from Phase 8's `reward_pipeline.py` without modification.

The three research-gated decisions are now closed by primary-source SDK evidence. Router gates are architecturally frozen (no `train_router` in LoraConfig); this is a justified deviation from GRPO-06's literal text, with protected-expert monitoring running offline using native `ForwardBackwardOutput.metrics`. GSPO is expressible on Tinker via `forward_backward_custom` for the true sequence-level IS ratio + RSPO floor (the PRIMARY path, per locked D-09-03), with `importance_sampling` (GRPO-style token-level IS) available as the instability fallback. Policy-shift monitoring for GRPO-07/08 is achievable using `compute_kl_sample_train` (per step, no extra inference) and `e_frac_with_tokens:mean` / `e_max_violation` (native MoE metrics from every forward pass).

The critical operational risk is judge-agent latency on the critical path. Claude Code agents spawn between `sampling_client.sample()` and `forward_backward()` — agent wall-clock time directly stalls the training client. Batch-parallel dispatch (all samples in one async gather before any `forward_backward`) and content-hash caching are the standard mitigations. This is the primary architectural decision Phase 9 planning must address.

**Primary recommendation:** Use **GSPO** (`forward_backward_custom`, sequence-level IS + RSPO stop-gradient floor) as the PRIMARY RL loss per locked decision D-09-03 (research confirms it FEASIBLE on Tinker); `importance_sampling` (GRPO token-level IS) is the documented **instability fallback** reachable via `--grpo-fallback`/`--no-gspo`. Cook from `tinker_cookbook/rl/train.py` as the structural scaffold, and treat `ForwardBackwardOutput.metrics` MoE keys as the native per-step router monitor.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| RL rollout sampling | Tinker cloud (sampling client) | — | Model inference happens server-side; client calls `sampling_client.sample()` |
| Reward computation | Local orchestrator (reward_pipeline.py) | — | Phase 8 contract; runs PHP, VeRPO, judge locally |
| Judge scoring (Claude-consistency) | Claude Code agents (Agent) | — | D-09-05 lock; NOT Anthropic API |
| Forward/backward + optimizer step | Tinker cloud (training client) | — | `forward_backward` + `optim_step` are cloud-side |
| MoE routing monitoring | Tinker cloud (ForwardBackwardOutput.metrics) | Offline weight export | Native per-step; offline check between checkpoints |
| Policy-shift KL | Local orchestrator (rl/metrics.py) | — | `compute_kl_sample_train` runs locally on returned logprobs |
| Checkpoint persistence | Tinker cloud (save_weights_for_sampler) | Local metadata file | Persistent path vs ephemeral; see Pitfall 1 |
| Interleaved batching | Local orchestrator (rl/interleaved.py) | — | Gen/judge ratio managed client-side |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| tinker | 0.22.3 | RL training client (sample, forward_backward, optim_step, save) | Established in .venv-tinker; verified via wp-reasoning-v2 SFT run |
| tinker-cookbook | 0.4.1 | RL loop scaffold (rl/train.py, rl/data_processing.py, rl/rollouts.py, rl/metrics.py, rl/interleaved.py) | Project standard; rl/ module is the structural template |
| torch | (bundled with tinker) | Local tensor math for advantage computation, KL metrics | Imported by cookbook |
| scripts/reward_pipeline.py | Phase 8 | Reward computation: PHPCS + VeRPO + judge composite | Phase 8 deliverable, Phase 9 consumes without modification |

[VERIFIED: installed in .venv-tinker, paths confirmed from SDK source files]

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| asyncio | stdlib | Async gather for batch agent dispatch | Used in incorporate_kl_penalty and batch judge calls |
| json / pathlib | stdlib | Checkpoint manifest, reward log serialization | All training loop metadata |
| protected_expert_mask.npy | Phase 7 artifact | [48,128] bool mask of 1,480 protected experts | Monitor-only; loaded at loop init, checked offline |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `importance_sampling` (GRPO token-level) | `forward_backward_custom` (sequence-level GSPO) | Custom requires manual IS ratio aggregation; `importance_sampling` is battle-tested in cookbook |
| GRPO group-centering | `ppo` loss type | PPO clip is available but adds clip-ratio bookkeeping; GRPO with MO-GRPO norm is simpler and sufficient |
| `cispo` loss | GRPO | CISPO is clipped IS policy optimization — may be relevant if KL blowup persists; investigate if primary path unstable |

**Installation:**
```bash
# tinker + cookbook already installed in .venv-tinker
# New training script uses existing venv — no new packages required
source .venv-tinker/bin/activate
python scripts/rl_train.py  # new Phase 9 script
```

---

## Package Legitimacy Audit

No new external packages required for Phase 9. All dependencies (`tinker`, `tinker-cookbook`) were installed and validated in Phase 5 (TINKER-PIVOT) and Phase 8. `reward_pipeline.py` is a local file, not an external package.

**Packages removed due to slopcheck [SLOP] verdict:** none — no new packages
**Packages flagged as suspicious [SUS]:** none

---

## Research-Gated Decision Resolutions

### D-09-02: Router-Training Scope — RESOLVED: FROZEN

**Evidence (primary source, SDK):**
```python
# .venv-tinker/.../tinker/types/lora_config.py
class LoraConfig(StrictBase):
    rank: int
    seed: Optional[int] = None
    train_unembed: bool = True
    train_mlp: bool = True   # covers MoE expert MLPs
    train_attn: bool = True
    # NO train_router field
```
```python
# .venv-tinker/.../tinker/lib/public_interfaces/service_client.py
def create_lora_training_client(
    self,
    base_model: str,
    rank: int = 32,
    seed: int | None = None,
    train_mlp: bool = True,
    train_attn: bool = True,
    train_unembed: bool = True,
    user_metadata: dict[str, str] | None = None,
) -> TrainingClient:
```
[VERIFIED: .venv-tinker installed SDK source]

**Decision:** Router gates are FROZEN on Tinker by construction. There is no API surface to target them. `train_mlp=True` covers MoE expert MLPs, not gate weights.

**Justified deviation from GRPO-06 text:** GRPO-06 mentions "router gates" and "router-gate-gradient KL-regularizer." This is infeasible on Tinker. The deviation is:
- Router-gate LoRA: REMOVED (no API)
- KL regularizer on gate gradients: REMOVED (no API)
- Protected-expert monitoring: RETAINED, now using native `ForwardBackwardOutput.metrics` (`e_frac_with_tokens:mean`, `e_max_violation:mean`) from every `forward_backward` call — no external profiler needed for monitoring
- Offline weight export check: available via `save_weights_for_sampler` + reuse of Phase 7 profiler

**Important nuance — routing still shifts:** Frozen gate ≠ routing stable. Phase 7 observed late-layer `E_eff` +7 during SFT because LoRA changes the gate's *inputs* (token representations), shifting which experts are called even without gate weight changes. GRPO-08 per-step monitoring remains meaningful and is achievable via native metrics.

**D-09-06 collapse:** D-09-06's two-tier response (active if router trained) collapses to monitoring-only path. Plan should reflect this.

---

### D-09-03: GSPO Expressibility — RESOLVED: FEASIBLE

**Available loss functions (primary source):**
```python
# .venv-tinker/.../tinker/types/loss_fn_type.py
LossFnType: TypeAlias = Literal[
    "cross_entropy",
    "importance_sampling",
    "ppo",
    "cispo",
    "dro",
]
```
[VERIFIED: .venv-tinker installed SDK source]

**`importance_sampling`** is the GRPO/IS token-level loss. The cookbook `rl/data_processing.py` shows:
```python
advantages_G = rewards_G - rewards_G.mean()  # group-centering
```
And `rl/train.py` passes these as `datum.loss_fn_inputs["advantages"]` to `forward_backward(loss_fn="importance_sampling")`. This is token-level IS (GRPO-style).

**`forward_backward_custom`** provides the sequence-level escape hatch:
```python
training_client.forward_backward_custom(
    data,
    custom_loss_fn,   # receives per-token logprobs
    loss_type_input="logprobs"
)
```
This allows computing a sequence-level IS ratio `exp(sum(log π_θ(aₜ|sₜ)) - sum(log π_β(aₜ|sₜ)))` client-side and returning it as gradients. True GSPO (sequence-level importance sampling) is implementable here. The RSPO stop-gradient floor is also implementable: clip ratio from below at `max(1, r) = 1` (floor) before multiplying by advantage.

**Decision:** PRIMARY path = **GSPO** = `forward_backward_custom` with sequence-level IS aggregation + RSPO stop-gradient floor (honors locked D-09-03; GSPO confirmed FEASIBLE). FALLBACK = `importance_sampling` (GRPO token-level IS, full cookbook infrastructure), used ONLY if GSPO proves unstable. The planner structures this as `use_gspo: bool` defaulting **True**, with `--grpo-fallback`/`--no-gspo` selecting the GRPO fallback path. Single run, NOT a side-by-side comparison.

**Pro-GRPO is infeasible on Tinker:** Pro-GRPO (expand-then-prune expert capacity) requires structural MoE surgery (changing gate routing or expert count). No Tinker API surface supports this. Remove from plan scope — was already deferred, but confirm explicitly.

---

### D-09-05: Claude-Agent Reward Risks — RESOLVED with Procedures

**R1 — Panickssery self-preference bias:**
- Risk: Claude-agent judges reward Claude-stylistic reasoning over correct reasoning, creating sycophantic reward signal.
- Procedure: After every 50 steps, sample 20 rollouts where judge_raw (normalized) diverges from fix_correctness (deterministic) by > 0.3. Human spot-check 5 of these. If > 2/5 show style-reward-over-correctness, flag for investigation.
- Mitigation already in Phase 8 contract: `judge_raw` is capped (anchored to deterministic `fix_correctness`). The cap limits how much pure-style reward can dominate. Phase 9 CONSUMES this contract, not modifying it.
- [ASSUMED] Specific numeric threshold for "systematic bias detected" — propose 40% divergence rate over 100-step window triggers human review.

**R2 — Reward noise → GRPO advantage collapse:**
- Risk: High judge_raw variance → low SNR → flat advantages → policy doesn't learn.
- Measurement: Track per-step `judge_raw` coefficient of variation (CV = std/mean) within each group G. Alert if CV > 1.5 for judge component.
- Mitigation (per CONTEXT canonical_refs — Dr.GRPO / noise-corrected GRPO): use group sizes G ≥ 8 for judge pathway to suppress within-group variance by 1/√G. MO-GRPO normalization (already in reward_pipeline.py `_mo_grpo_norm`) handles cross-signal scale differences.
- [CITED: CONTEXT.md canonical_refs] Noise-corrected GRPO / Dr.GRPO referenced for R2 mitigation. Specific CV threshold is [ASSUMED].

**R3 — Judge agent latency on training critical path:**
- Root cause: After `sampling_client.sample()` returns, Phase 9 must dispatch Claude Code agents for consistency scoring BEFORE calling `forward_backward()`. Training client sits idle during agent wall-clock time.
- Measured risk: If each judge agent takes 30-90s, and we spawn G_judge agents sequentially, a group of 8 judge samples = 4-12 minutes stall per step. This is the dominant loop bottleneck.
- Mitigations (engineering):
  1. **Batch-parallel async dispatch:** Spawn all G_judge agents in parallel via `asyncio.gather()` (same pattern as `run-data-pipeline` SKILL.md). Wall-clock = max(agent_times), not sum.
  2. **Content-hash cache:** Key = `(php_code[:512], critique_text[:512])`. If same (code, critique) seen within session, return cached score. PHP refactors often produce near-identical outputs across rollout groups.
  3. **Double-buffer rollouts (if needed):** Collect next batch's samples while current batch's judge agents run. Requires interleaved.py support — already in cookbook.
  4. **Timeout with imputation:** Set agent timeout = 120s. Timeout → impute from group mean (same strategy already in reward_pipeline.py `judge_imputed_from_group`).

---

## Architecture Patterns

### System Architecture Diagram

```
PHASE 9 RL TRAINING LOOP (Tinker cloud execution)

PROMPT DATASET (wp_gen + wp_judge items, 60/40 ratio)
        │
        ▼
[INTERLEAVED BATCHER] ←── protected_expert_mask.npy (monitor-only)
  rl/interleaved.py
        │ G_gen prompts + G_judge prompts per batch
        ▼
[TINKER SAMPLING CLIENT]
  sampling_client.sample(prompts, G=group_size, temp=0.7)
  Returns: tokens + per-token logprobs
        │
        │ rollout sequences
        ▼
[DUAL REWARD PIPELINE]
 ├── wp_gen samples → reward_pipeline.compute_group_rewards(php_codes, judge_client, judge_model)
 │       └── PHPCS + VeRPO + [Claude-agent judge consistency score]
 │                              │
 │              [Claude Code Agents, async parallel dispatch]
 │              Agent(run_in_background=true) × G_judge
 │              Content-hash cache → bypass if seen
 │              Timeout 120s → impute from group mean
 └── wp_judge samples → reward_pipeline.compute_group_rewards(judge_php_codes, ...)
        └── Same pipeline; fix_correctness anchors judge_raw
        │
        │ RewardResult list, scalar + breakdown_dict
        ▼
[ADVANTAGE COMPUTATION]
  rl/data_processing.py: compute_advantages()
  MO-GRPO norm: (x - mu) / (sigma + 1e-8) per signal independently
  Group centering: advantages_G = rewards_G - rewards_G.mean()
  Zero-advantage groups: skipped (remove_constant_reward_groups)
        │
        ▼
[TINKER TRAINING CLIENT]
  forward_backward_custom(data, gspo_loss_fn)             ← GSPO path (PRIMARY, default)
  OR forward_backward(data, loss_fn="importance_sampling") ← GRPO fallback (--grpo-fallback)
  Returns: ForwardBackwardOutput
   ├── .metrics["e_frac_with_tokens:mean"]  ← MoE routing health
   ├── .metrics["e_max_violation:mean"]     ← Routing concentration
   └── .loss_fn_outputs                     ← Per-datum loss values
        │
        ▼
[POLICY-SHIFT KL COMPUTATION]
  compute_kl_sample_train(data_D, training_logprobs_D)
  Returns: kl_sample_train_v1, kl_sample_train_v2
  AUTO-HALT if kl_v1 > HALT_THRESHOLD (planner sets value)
        │
        ▼
[OPTIMIZER STEP]
  optim_step(adam_params)  ← AdamW, weight_decay=0.0 default
        │
        ▼
[CHECKPOINT (every N steps)]
  save_weights_for_sampler(name="phase9-step-{step}")   ← PERSISTENT
  create_sampling_client(model_path=...)                 ← Reload for next rollout
  Write checkpoint_manifest.json locally
        │
        ▼
[METRICS LOG (every step)]
  Step, reward_mean, reward_std, judge_cv, kl_v1, kl_v2,
  e_frac_with_tokens, e_max_violation, format_collapse_rate,
  judge_impute_rate, wall_clock_per_step
  → wandb or local JSONL (planner decides sink)
```

### Recommended Project Structure

```
scripts/
├── rl_train.py                    # Phase 9 main training loop (new)
├── rl_rollouts.py                 # Rollout + reward collection (new or extend cookbook)
├── rl_judge_dispatch.py           # Claude-agent batch dispatch + cache (new)
├── reward_pipeline.py             # Phase 8 — CONSUME ONLY
└── _prequant_multiuser.sh         # Existing, unrelated
data/
└── rl_prompts/
    ├── wp_gen_train.jsonl         # Generation task prompts
    └── wp_judge_train.jsonl       # Judge task prompts
output/
└── rl_checkpoints/
    ├── checkpoint_manifest.json   # Step → Tinker model_path mapping
    └── metrics/
        └── rl_metrics.jsonl       # Per-step training metrics
.planning/
└── phases/09-gspo-training/       # This phase
```

### Pattern 1: Dual-Mode Interleaved Rollout

```python
# Source: tinker_cookbook/rl/interleaved.py + local orchestration
import asyncio
from tinker_cookbook.rl.interleaved import InterleavedTrainer

# Budget: 60% judge, 40% gen
N_JUDGE = 12   # G_judge samples per batch (planner sets)
N_GEN   = 8    # G_gen samples per batch

async def collect_batch(sampling_client, gen_prompts, judge_prompts):
    # Parallel sample both modes
    gen_futures = [
        sampling_client.sample(p, num_samples=GROUP_SIZE, temperature=0.7)
        for p in gen_prompts[:N_GEN // GROUP_SIZE]
    ]
    judge_futures = [
        sampling_client.sample(p, num_samples=GROUP_SIZE, temperature=0.7)
        for p in judge_prompts[:N_JUDGE // GROUP_SIZE]
    ]
    gen_results, judge_results = await asyncio.gather(
        asyncio.gather(*gen_futures),
        asyncio.gather(*judge_futures),
    )
    return gen_results, judge_results
```

### Pattern 2: Batch-Parallel Judge Agent Dispatch

```python
# Source: adapted from .claude/skills/wp-finetune:run-data-pipeline/SKILL.md pattern
import asyncio

async def score_judge_consistency_batch(samples: list[dict]) -> list[float]:
    """Dispatch Claude Code agents in parallel; content-hash cache; timeout+impute."""
    cache = {}  # content_hash -> score
    tasks = []
    for sample in samples:
        key = hash((sample["php_code"][:512], sample["critique"][:512]))
        if key in cache:
            tasks.append(asyncio.sleep(0, result=cache[key]))  # cached
        else:
            tasks.append(
                asyncio.wait_for(
                    spawn_judge_agent(sample),
                    timeout=120.0
                )
            )
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # Impute timeouts/errors from group mean (matching reward_pipeline.py behavior)
    scores = [r if isinstance(r, float) else None for r in results]
    group_mean = sum(s for s in scores if s is not None) / max(1, sum(1 for s in scores if s is not None))
    return [s if s is not None else group_mean for s in scores]
```

### Pattern 3: GSPO Primary + GRPO Fallback Loss

```python
# GRPO path (FALLBACK ONLY — selected via --grpo-fallback/--no-gspo if GSPO proves unstable)
fb_output = training_client.forward_backward(
    data=batch_data,
    loss_fn="importance_sampling",
    loss_fn_config={},
).result()

# GSPO path (PRIMARY — default, sequence-level IS + RSPO floor, locked D-09-03)
def gspo_loss_fn(logprobs_list):
    """Custom sequence-level IS loss with RSPO stop-gradient floor."""
    results = []
    for datum_logprobs, datum in zip(logprobs_list, batch_data):
        sampling_lp = datum.loss_fn_inputs["logprobs"].to_torch()
        train_lp = datum_logprobs
        seq_ratio = (train_lp - sampling_lp).sum().exp()  # IS ratio (sequence)
        seq_ratio_floored = seq_ratio.clamp(min=1.0)       # RSPO floor
        adv = datum.loss_fn_inputs["advantages"].to_torch().sum()
        loss = -(seq_ratio_floored * adv)
        results.append({"loss": loss})
    return results

if use_gspo:
    fb_output = training_client.forward_backward_custom(
        data=batch_data,
        custom_loss_fn=gspo_loss_fn,
        loss_type_input="logprobs",
    ).result()
```

### Pattern 4: Policy-Shift Monitoring + Auto-Halt

```python
# Source: .venv-tinker/.../tinker_cookbook/rl/metrics.py — compute_kl_sample_train
from tinker_cookbook.rl.metrics import compute_kl_sample_train

kl_metrics = compute_kl_sample_train(data_D, training_logprobs_D)
kl_v1 = kl_metrics["optim/kl_sample_train_v1"]

# MoE routing health from forward_backward output (native, free)
moe_metrics = fb_output.metrics
e_frac = moe_metrics.get("e_frac_with_tokens:mean", None)
e_max = moe_metrics.get("e_max_violation:mean", None)

# Auto-halt conditions (planner sets thresholds)
if kl_v1 > KL_HALT_THRESHOLD:
    raise RuntimeError(f"Policy shift too large: kl_v1={kl_v1:.4f} > {KL_HALT_THRESHOLD}")
if e_frac is not None and e_frac < EFRAC_COLLAPSE_THRESHOLD:
    raise RuntimeError(f"MoE routing collapse: e_frac={e_frac:.4f} < {EFRAC_COLLAPSE_THRESHOLD}")
```

### Pattern 5: Persistent Checkpoint (not ephemeral)

```python
# CORRECT — persistent
training_client.save_weights_for_sampler(name=f"phase9-step-{step}")
# Later: sampling_client = service_client.create_sampling_client(model_path=f"phase9-step-{step}")

# WRONG — ephemeral (from past incident obs 2455, 2457)
# sampling_client = training_client.save_weights_and_get_sampling_client()
# This returns an ephemeral client; weights NOT stored persistently.
```

### Anti-Patterns to Avoid

- **Sequential judge dispatch:** Spawning Claude Code agents one-at-a-time after rollout. Each agent blocks the training client. Always use `asyncio.gather()`.
- **Ephemeral checkpoint:** Using `save_weights_and_get_sampling_client()` for checkpoints. Weights are lost on session end. Use `save_weights_for_sampler(name=...)`.
- **Skipping MoE metrics check:** `ForwardBackwardOutput.metrics` is populated automatically during MoE training; ignoring it means missing routing collapse until it's severe.
- **Pro-GRPO expert expansion:** Not supported on Tinker (no structural MoE API). Do not plan tasks for this.
- **Rebuilding reward logic:** Phase 9 must use `compute_group_rewards` as-is. Modifications to reward_pipeline.py belong to Phase 8 PRs, not Phase 9.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Group-relative advantage | Custom normalization | `rl/data_processing.py compute_advantages()` | Already handles group centering, empty-group skip |
| Interleaved gen/judge batching | Custom scheduling | `rl/interleaved.py InterleavedTrainer` | Staleness tracking, async scheduling already solved |
| KL policy-shift metric | Manual logprob diff | `rl/metrics.py compute_kl_sample_train()` | Action-mask aware, both v1/v2 estimators |
| Checkpoint management | Custom model save | `save_weights_for_sampler(name=...)` + `create_sampling_client(model_path=...)` | Only persistent path; ephemeral alternative is a known failure mode |
| MoE routing health | External profiler | `ForwardBackwardOutput.metrics` keys | Populated automatically on every forward pass; no extra compute |
| Reward combination + normalization | Custom multi-signal | `reward_pipeline.compute_group_rewards()` | Phase 8 deliverable; MO-GRPO norm, VeRPO, security gate all handled |

---

## Common Pitfalls

### Pitfall 1: Ephemeral Checkpoint Loss
**What goes wrong:** Using `save_weights_and_get_sampling_client()` for checkpoints. Weights exist only for the session. Session ends → model state lost. Occurred in Phase 5 (obs 2455, 2457 — ep3 not persistent).
**Why it happens:** Method name implies saving; actually just exports temporarily for inference.
**How to avoid:** Always `save_weights_for_sampler(name="phase9-step-{step}")` for durable checkpoints. Maintain `checkpoint_manifest.json` mapping step → Tinker model path.
**Warning signs:** `sampling_client` returned from `save_weights_and_get_sampling_client()` in checkpoint path.

### Pitfall 2: Sequential Judge Agent Dispatch (Latency Stall)
**What goes wrong:** Spawning judge agents one-at-a-time → O(N × agent_time) stall before `forward_backward`. At 30s/agent × 8 samples = 4min/step. Training becomes impractically slow.
**Why it happens:** Natural to write `for sample in samples: await agent(sample)`.
**How to avoid:** Always `asyncio.gather(*[agent(s) for s in samples])`. Add content-hash cache. Set 120s timeout with group-mean imputation.
**Warning signs:** Step wall-clock > 5 minutes; `judge_impute_rate` = 0 (every agent completing suggests no parallelism issue, but timing still tells truth).

### Pitfall 3: Constant-Reward Group Advantage Collapse
**What goes wrong:** All samples in a group G receive the same reward → advantages = 0 → gradient = 0 → policy doesn't update for that group.
**Why it happens:** Security gate is terminal (all fail → all score 0). Early training, model may consistently fail PHP parse (all 0) or consistently pass VeRPO trivially (all 1).
**How to avoid:** `remove_constant_reward_groups()` (already in cookbook). Monitor fraction of skipped groups per step in metrics. If > 30% groups skipped, reward signal is degraded — investigate component calibration.
**Warning signs:** `frac_skipped_groups` rising; `reward_std` approaching 0.

### Pitfall 4: Format Collapse Regression (Known Baseline)
**What goes wrong:** RL training destabilizes the wp-reasoning-v2 format, reverting to terse JSON collapse. Baseline at training start: 1.3%@greedy, 9.1%@temp0.7.
**Why it happens:** RL updates can shift token distribution toward high-reward-but-format-breaking outputs if format is not rewarded.
**How to avoid:** Log `format_collapse_rate` each step (check for wp-reasoning-v2 XML tags in sampled outputs). If rate exceeds 5%@greedy or 15%@temp0.7, halt and investigate reward signal.
**Warning signs:** Rising proportion of samples without `<wp_gen>` or `<wp_judge>` wrapper tags.

### Pitfall 5: MoE Routing Collapse Not Caught Early
**What goes wrong:** `e_frac_with_tokens:mean` declining slowly → late collapse → expensive restart. Phase 7 showed late-layer experts are already loaded unevenly.
**Why it happens:** Not logging or not alerting on `ForwardBackwardOutput.metrics` MoE keys.
**How to avoid:** Log every step. Set soft alert at `e_frac_with_tokens:mean < 0.7` and hard halt at `< 0.5`. Protected expert mask check (offline, every 100 steps): export weights and run jaccard against Phase 7 mask.

### Pitfall 6: ROADMAP / Skill Stale — DGX References
**What goes wrong:** ROADMAP.md Phase 9 skill text references `dgx.execute("unsloth_studio", ...)` and a per-epoch DGX loop. This is dead. The `run-training.md` skill and `observe-training` 6-agent team were DGX/Unsloth-specific (GPU/thermal telemetry, unsloth-headless container).
**Impact on planning:** New `wp-finetune:run-rl-training` skill must be Tinker-native. The structural template from `run-training.md` (validate → ensure_ready → train loop → status) is reusable but the venue, container, and telemetry sources are all replaced. `observe-training` agents must reframe to: loop/reward/KL/cost metrics (not GPU/thermal).
**What planner must do:** ROADMAP.md Phase 9 description needs a skill-text update task in Wave 0.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| DGX-local Unsloth RL | Tinker cloud RL | Phase 5 pivot (TINKER-PIVOT.md) | No GPU management; Tinker handles compute |
| `save_weights_and_get_sampling_client()` for checkpoints | `save_weights_for_sampler(name)` + `create_sampling_client(model_path)` | Discovered Phase 5 (obs 2455) | Persistent durability |
| Per-step router monitoring via external profiler | Native `ForwardBackwardOutput.metrics` MoE keys | This research (D-09-02 resolution) | Free per-step monitoring, no overhead |
| ROADMAP Phase 9 skill text (DGX) | Tinker-native `run-rl-training` skill | This phase planning | Planner must update ROADMAP skill text |

**Deprecated / outdated:**
- `dgx.execute("unsloth_studio", ...)`: Dead. No DGX in Phase 9.
- `observe-training` GPU/thermal agent team: Reframe to reward/KL/cost telemetry agents.
- `run-training.md` skill as written: Venue is Tinker; structural template is reusable but contents are stale.
- Pro-GRPO: Infeasible on Tinker. Not a plan item.
- D-09-06 active path (gate-gradient KL regularizer): Collapsed to monitoring-only. Plan should not include a "if routing shifts, activate two-tier response" task — monitoring IS the response.

---

## Runtime State Inventory

> Rename/refactor/migration phase: N/A — this is greenfield RL training. No renaming.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| tinker (cloud) | RL training | ✓ | 0.22.3 | — |
| tinker-cookbook | RL scaffold | ✓ | 0.4.1 | — |
| Qwen/Qwen3-30B-A3B on Tinker | Base model | ✓ | confirmed Phase 5 | — |
| scripts/reward_pipeline.py | Reward computation | ✓ | Phase 8 complete | — |
| protected_expert_mask.npy | Routing monitor | ✓ | Phase 7 artifact | — |
| PHPCS (local) | reward_pipeline | ✓ | guard at module load | — |
| Claude Code agents (Agent()) | Judge consistency scoring | ✓ | project standard | — |
| data/rl_prompts/ | Training prompts | ✗ (not yet created) | — | Wave 0 task: assemble from existing seed data |

**Missing dependencies with no fallback:** None that block training.
**Missing dependencies that need Wave 0 tasks:** `data/rl_prompts/wp_gen_train.jsonl` + `wp_judge_train.jsonl` — must be assembled from existing Phase 4 seeds/data before rollout can begin.

---

## Validation Architecture

> workflow.nyquist_validation: absent from config.json — treated as enabled.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (existing, established in Phase 8) |
| Config file | pytest.ini or pyproject.toml (existing) |
| Quick run command | `pytest tests/test_reward_pipeline.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| GRPO-05 | Dual-mode interleaved batch includes both wp_gen and wp_judge samples | unit | `pytest tests/test_rl_train.py::test_dual_mode_batch -x` | ❌ Wave 0 |
| GRPO-05 | Judge sample count ≥ gen sample count in every batch | unit | `pytest tests/test_rl_train.py::test_judge_ge_gen_budget -x` | ❌ Wave 0 |
| GRPO-06 | LoRA targets: train_mlp=True, train_attn=True, train_unembed=True | unit | `pytest tests/test_rl_train.py::test_lora_config -x` | ❌ Wave 0 |
| GRPO-06 | Protected expert mask loaded and checked offline each checkpoint | integration | `pytest tests/test_rl_train.py::test_protected_mask_check -x` | ❌ Wave 0 |
| GRPO-07 | RSPO floor applied when use_gspo=True (ratio clamped min=1.0) | unit | `pytest tests/test_rl_train.py::test_gspo_rspo_floor -x` | ❌ Wave 0 |
| GRPO-07 | GRPO fallback produces non-zero advantages for mixed reward groups | unit | `pytest tests/test_rl_train.py::test_grpo_advantages -x` | ❌ Wave 0 |
| GRPO-08 | Auto-halt raised when kl_v1 > threshold | unit | `pytest tests/test_rl_train.py::test_kl_autohalt -x` | ❌ Wave 0 |
| GRPO-08 | Auto-halt raised when e_frac_with_tokens < threshold | unit | `pytest tests/test_rl_train.py::test_routing_autohalt -x` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/test_rl_train.py -x -q` (unit tests only, < 30s)
- **Per wave merge:** `pytest tests/ -x -q` (full suite including reward_pipeline)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_rl_train.py` — covers GRPO-05 through GRPO-08 (all ❌ above)
- [ ] `data/rl_prompts/wp_gen_train.jsonl` — generation task prompts for rollout
- [ ] `data/rl_prompts/wp_judge_train.jsonl` — judge task prompts for rollout
- [ ] ROADMAP.md Phase 9 skill text update (DGX → Tinker)

---

## Security Domain

> security_enforcement absent from config — treated as enabled.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Tinker auth handled by SDK (token in env) |
| V3 Session Management | no | Stateless training loop |
| V4 Access Control | no | Single-user local execution |
| V5 Input Validation | yes | PHP code inputs to reward_pipeline already sanitized (Phase 8 PHPCS + deterministic checks) |
| V6 Cryptography | no | No crypto primitives hand-rolled |

### Known Threat Patterns for RL Training Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Security-trigger bypass via reward gaming | Tampering | Terminal security gate in reward_pipeline (Phase 8); reward = 0 if triggered; Phase 9 consumes this as-is |
| Prompt injection in judge samples | Tampering | Judge prompts are structured templates; PHP code is enclosed in code blocks; reward_pipeline deterministic mode strips LLM checks |
| Tinker API token exposure | Disclosure | Token loaded from env (not hardcoded); `.env` not committed |
| Training data poisoning via rl_prompts | Tampering | Prompts assembled from existing audited Phase 4 seeds; Wave 0 task must verify provenance |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Numeric threshold for Panickssery bias: 40% divergence rate over 100 steps triggers human review | D-09-05 R1 | Threshold too loose = bias persists; too tight = false positives |
| A2 | Judge-agent timeout 120s sufficient for 95th percentile Claude Code agent runs | D-09-05 R3 | Timeout too tight = excessive imputation; too loose = loop stalls |
| A3 | Reward noise CV threshold 1.5 for judge_raw flags low-SNR episodes | D-09-05 R2 | Wrong threshold = missed degradation or false positives |
| A4 | `cispo` loss in Tinker = Clipped IS Policy Optimization | Standard Stack Alternatives | If wrong, the alternative description is inaccurate; use `forward_backward_custom` instead |
| A5 | Group sizes G_gen=8, G_judge=12 (40/60 budget approximation) are appropriate starting points | Architecture | Wrong group size → poor advantage variance; planner should treat as tunable |
| A6 | KL halt threshold for kl_sample_train_v1: planner must set specific value | Pattern 4 | No halt = training diverges silently; too aggressive halt = premature stop |

**If this table is empty:** All claims in this research were verified or cited — no user confirmation needed. Table is NOT empty — A1-A6 require planner/user decisions at planning time.

---

## Open Questions (RESOLVED at planning)

1. **KL halt threshold (GRPO-07/08)**
   - What we know: `compute_kl_sample_train` returns `kl_sample_train_v1` (mean logprob diff). Increasing values = increasing policy shift.
   - What's unclear: Appropriate numeric halt threshold for Qwen3-30B-A3B at rank-32 LoRA. Typical GRPO papers use 0.1-0.3 for token-level KL.
   - RESOLVED: Plan 09-05 finalizes soft alert at 0.1, hard halt at 0.3 on `kl_sample_train_v1`; tune from first 50 steps.

2. **RL prompt dataset assembly**
   - What we know: `data/rl_prompts/` does not exist. Must be assembled from Phase 4 seeds and existing data.
   - What's unclear: Which specific files to draw from; how to split gen vs judge prompts.
   - RESOLVED: Plan 09-01 builds the audited, val-clean RL prompt corpus (`scripts/build_rl_prompts.py` → `data/rl_prompts/wp_gen_train.jsonl` + `wp_judge_train.jsonl`).

3. **Metrics sink (wandb vs local JSONL)**
   - What we know: Cookbook supports both; `ForwardBackwardOutput.metrics` returns dict.
   - What's unclear: Project preference — wandb was not mentioned in CONTEXT.md.
   - RESOLVED: Plan 09-05 uses local JSONL (`output/rl_checkpoints/metrics/rl_metrics.jsonl`); wandb left as optional future flag.

4. **GSPO vs GRPO as primary**
   - What we know: Both feasible. `importance_sampling` (GRPO) is cookbook-native. `forward_backward_custom` (GSPO) requires custom implementation.
   - What's unclear: Whether the complexity of sequence-level IS ratio is warranted given frozen router.
   - RESOLVED: Plan 09-05 uses **GSPO** `forward_backward_custom` (sequence-level IS + RSPO floor) as the PRIMARY loss, `use_gspo` defaulting **True**, honoring locked D-09-03 (research confirmed GSPO FEASIBLE). GRPO `importance_sampling` is the documented instability FALLBACK via `--grpo-fallback`/`--no-gspo`. D-09-03's "GSPO primary" disposition is LOCKED — NOT Claude's Discretion (CONTEXT.md's Discretion list contains only the interleave ratio and numeric thresholds). NOT a side-by-side comparison.

---

## Sources

### Primary (HIGH confidence — verified from installed SDK source)
- `.venv-tinker/.../tinker/types/lora_config.py` — LoraConfig fields (no train_router)
- `.venv-tinker/.../tinker/lib/public_interfaces/service_client.py` — create_lora_training_client signature
- `.venv-tinker/.../tinker/lib/public_interfaces/training_client.py` — forward_backward, forward_backward_custom, save_weights_for_sampler
- `.venv-tinker/.../tinker/types/loss_fn_type.py` — 5 available loss functions
- `.venv-tinker/.../tinker/types/forward_backward_output.py` — metrics dict with MoE routing keys
- `.venv-tinker/.../tinker_cookbook/rl/metrics.py` — compute_kl_sample_train, compute_post_kl, incorporate_kl_penalty
- `.venv-tinker/.../tinker_cookbook/rl/data_processing.py` — compute_advantages, MO-GRPO norm pattern
- `.venv-tinker/.../tinker_cookbook/rl/train.py` — full RL training framework
- `scripts/reward_pipeline.py` — Phase 8 contract (645 lines, complete implementation)
- `.planning/phases/09-gspo-training/09-CONTEXT.md` — locked decisions, research-gated decisions
- `.planning/phases/08-reward-infrastructure/08-CONTEXT.md` — D-08-04 reward contract
- `.planning/phases/07-router-profiling-protected-expert-set/07-CONTEXT.md` — D-09 CI-aware gate, protected mask

### Secondary (MEDIUM confidence — cited from project documents)
- `.planning/TINKER-PIVOT-RESEARCH.md` — Tinker version confirmation, wp-reasoning-v2 training observations, checkpoint durability incident
- `.planning/ROADMAP.md` — Phase 9 skill text (confirmed stale DGX references)
- `docs/wp-finetune:run-training.md` — structural template for new rl-training skill (venue stale, structure reusable)
- `.claude/skills/wp-finetune:run-data-pipeline/SKILL.md` — Agent dispatch pattern for judge scoring

### Tertiary (LOW confidence — procedure design, thresholds)
- [CITED: CONTEXT.md canonical_refs] Dr.GRPO / noise-corrected GRPO for R2 reward noise mitigation
- [CITED: CONTEXT.md canonical_refs] Panickssery self-preference bias literature for R1
- [ASSUMED] Specific numeric thresholds (CV 1.5, KL 0.1/0.3, EFRAC 0.7/0.5, format collapse 5%/15%) — derived from principles, not measured on this model

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions verified from installed SDK
- Architecture: HIGH — patterns derived directly from SDK source code
- D-09-02 resolution: HIGH — primary source (LoraConfig, service_client.py)
- D-09-03 resolution: HIGH — primary source (loss_fn_type.py, training_client.py, forward_backward_custom)
- D-09-05 R1-R3: MEDIUM — procedures and principles verified; specific thresholds ASSUMED
- ROADMAP staleness: HIGH — confirmed from ROADMAP.md source text

**Research date:** 2026-06-20
**Valid until:** 2026-07-20 (Tinker SDK — 30 days; API stable but version may advance)
