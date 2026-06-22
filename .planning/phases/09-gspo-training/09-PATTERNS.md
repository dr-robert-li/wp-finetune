# Phase 9: GSPO Training — Pattern Map

**Mapped:** 2026-06-20
**Files analyzed:** 11 new/modified files
**Analogs found:** 9 / 11

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `scripts/rl_train.py` | training-driver | event-driven (Tinker step loop) | `scripts/tinker_reasoning_sft.py` | exact |
| `scripts/rl_rollouts.py` | data-transform | batch + transform | `tinker_cookbook/rl/data_processing.py` (SDK) | role-match |
| `scripts/rl_judge_dispatch.py` | service | request-response + cache | `scripts/claude_agent.py` | role-match |
| `.claude/skills/wp-finetune:run-rl-training/SKILL.md` | skill-orchestrator | event-driven (agent lifecycle) | `.claude/skills/wp-finetune:run-training/SKILL.md` | role-match (stale venue) |
| `scripts/tinker_rl_data.py` | data-adapter | transform | `scripts/tinker_reasoning_data.py` | exact |
| `data/rl_prompts/wp_gen_train.jsonl` | data | batch | `data/reasoning_dataset/openai_train.jsonl` schema | schema-match |
| `data/rl_prompts/wp_judge_train.jsonl` | data | batch | `data/reasoning_dataset/openai_train.jsonl` schema | schema-match |
| `output/rl_checkpoints/checkpoint_manifest.json` | artifact | CRUD | `tinker_reasoning_sft.py` manifest pattern (lines 74–78, 227–232) | pattern-match |
| `output/rl_checkpoints/metrics/rl_metrics.jsonl` | sink | streaming | `observe-training/SKILL.md` JSONL schema | schema-match |
| `tests/test_rl_train.py` | test | request-response | `tests/test_reward_pipeline.py` | exact |
| `ROADMAP.md` (Phase 9 skill-text patch) | docs | CRUD | n/a — one-line text correction | no-analog |

---

## Pattern Assignments

### `scripts/rl_train.py` (training-driver, event-driven)

**Analog:** `scripts/tinker_reasoning_sft.py` — same Tinker venue, same model, same SDK 0.22.3

**Imports pattern** (lines 1–26 of analog):
```python
import tinker
from tinker_cookbook import hyperparam_utils as hp
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
```

Phase 9 extension — add these imports on top of the SFT set:
```python
from tinker_cookbook.rl.metrics import compute_kl_sample_train
from tinker_cookbook.rl.data_processing import (
    compute_advantages,
    remove_constant_reward_groups,
    assemble_training_data,
)
from scripts.rl_rollouts import collect_rollouts
from scripts.reward_pipeline import compute_group_rewards
```

**ServiceClient + LoRA client setup** (lines 194–201 of analog — copy verbatim, change loss only):
```python
sc = tinker.ServiceClient()
tc = sc.create_lora_training_client(
    base_model=BASE_MODEL,
    rank=args.rank,
    train_mlp=True,         # MoE expert w1/w2/w3 — always on
    train_attn=args.train_attn,
    train_unembed=args.train_unembed,
    # OMIT train_router — D-09-06: router gates architecturally frozen on Tinker
)
```

**Core RL training loop** (extends analog lines 207–221; replace `cross_entropy` with `importance_sampling`):
```python
for step in range(args.total_steps):
    # 1. Sample rollouts (interleaved wp_gen / wp_judge at ~40/60 ratio)
    rollouts = collect_rollouts(sampling_client, prompts, args)

    # 2. Compute rewards — consume reward_pipeline unmodified
    rewards = compute_group_rewards(
        php_codes=[r.completion for r in rollouts],
        judge_client=judge_client,
        judge_model=args.judge_model,
    )

    # 3. Compute advantages (cookbook call — do not reimplement)
    groups = build_trajectory_groups(rollouts, rewards)
    groups = remove_constant_reward_groups(groups)
    advantages = compute_advantages(groups)
    data, meta = assemble_training_data(groups, advantages)

    # 4. Forward-backward with importance_sampling loss
    fb = tc.forward_backward(
        data=data,
        loss_fn="importance_sampling",
        clip_ratio=args.clip_ratio,          # PPO epsilon; default 0.2
    )
    tc.optim_step(tinker.AdamParams(learning_rate=args.lr))

    # 5. Metrics — KL + MoE health
    out = _res(fb)
    kl_metrics = compute_kl_sample_train(data, out.training_logprobs)
    moe_metrics = out.metrics   # e_frac_with_tokens:mean, e_max_violation:mean/max
    _log_step(step, rewards, kl_metrics, moe_metrics, args)

    # 6. KL halt guard (GRPO-04)
    if kl_metrics["optim/kl_sample_train_v2"] > args.kl_halt_threshold:
        logger.warning("KL halt at step %d — saving emergency checkpoint", step)
        _save_checkpoint(tc, f"{args.save_name}-kl-halt-{step}", manifest)
        break

    # 7. Periodic checkpoint (GRPO-05)
    if (step + 1) % args.checkpoint_every == 0:
        _save_checkpoint(tc, f"{args.save_name}-step{step+1}", manifest)
```

**Future resolver helper** (copy verbatim from analog lines 58–60):
```python
def _res(f):
    return f.result() if hasattr(f, "result") else f
```

**Manifest write helper** (copy verbatim from analog lines 74–78):
```python
def _write_manifest(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
```

**Persistent checkpoint pattern** (copy from analog lines 227–232; `ttl_seconds=None` = persistent):
```python
def _save_checkpoint(tc, name, manifest):
    sampler_path = _res(tc.save_weights_for_sampler(name=name, ttl_seconds=None)).path
    manifest["checkpoints"].append({
        "step": name,
        "sampler_path": sampler_path,
        "ts": datetime.utcnow().isoformat(),
    })
    _write_manifest(MANIFEST_PATH, manifest)
    return sampler_path
```

**Protected-expert monitor** — per-step native metrics only (D-09-06: no Jaccard per-step):
```python
# After each forward_backward:
e_max = moe_metrics.get("e_max_violation:max", 0.0)
if e_max > args.moe_alert_threshold:
    logger.warning("MoE e_max_violation:max=%.4f exceeds threshold at step %d", e_max, step)
# Full Jaccard check every N steps — see Phase 7 analog below
```

**GSPO deviation note:** Research references `forward_backward_custom` for sequence-level IS (RSPO stop-gradient floor, IS ratio `min=1.0`). No project precedent exists for this call. If `forward_backward(loss_fn="importance_sampling")` proves token-level only, planner must consult Tinker SDK docs for `forward_backward_custom` signature. Flag as **No Analog** for that specific call path.

---

### `scripts/rl_rollouts.py` (data-transform, batch + transform)

**Analog:** `tinker_cookbook/rl/data_processing.py` (SDK — read-only, cite by symbol)

**Interleaved prompt sampling** (new code, no project analog):
```python
JUDGE_RATIO = 0.6   # D-09-04: ~60% wp_judge, 40% wp_gen
def sample_interleaved_prompts(gen_pool, judge_pool, batch_size):
    n_judge = round(batch_size * JUDGE_RATIO)
    n_gen   = batch_size - n_judge
    return random.sample(judge_pool, n_judge) + random.sample(gen_pool, n_gen)
```

**Advantage computation** — delegate entirely to cookbook (do not reimplement):
```python
from tinker_cookbook.rl.data_processing import (
    compute_advantages,               # group-centering: A_G = r_G - mean(r_G)
    remove_constant_reward_groups,    # drops zero-gradient groups
    assemble_training_data,           # flat Datum list with group/traj metadata
)
```

**MO-GRPO normalization** — copy from `scripts/reward_pipeline.py` (lines 205–220):
```python
_EPSILON = 1e-8
def _mo_grpo_norm(values: np.ndarray) -> np.ndarray:
    mu    = values.mean()
    sigma = values.std(ddof=0)
    return (values - mu) / (sigma + _EPSILON)
```
Apply per-signal within each group before combining into scalar reward passed to cookbook.

**RewardResult consumption** — `compute_group_rewards` returns `list[RewardResult]`; each has `.scalar` (float) and `.breakdown` (RewardBreakdown). Pass `.scalar` to trajectory group builder; log `.breakdown` to `rl_metrics.jsonl`.

**Data format** (from `tinker_reasoning_data.py` lines 32–46):
```python
# RL prompts use same OpenAI chat schema as SFT data:
# {"messages": [{"role": "user", "content": "..."}, ...]}
# Loaded via FromConversationFileBuilder with same BASE_MODEL + RENDERER_NAME
BASE_MODEL     = "Qwen/Qwen3-30B-A3B"
RENDERER_NAME  = "qwen3_disable_thinking"
```

---

### `scripts/rl_judge_dispatch.py` (service, request-response + cache)

**Analog:** `scripts/claude_agent.py` — THIS is how the project calls LLMs from Python. Not via background `Agent()` (that is orchestrator-only). Python code calls `claude --print` via subprocess.

**Resolved conflict (advisor flag):** `build_antihack_set.py` docstring shows `Agent(run_in_background=True)` — but that is a SKILL orchestrator comment, not Python code. From Python, use `claude_agent.py`'s subprocess pattern.

**Core dispatch pattern** (copy from `scripts/claude_agent.py` lines 46–101):
```python
from scripts.claude_agent import generate_json

def score_judge_consistency(
    php_code: str,
    critique_text: str,
    model: str = "sonnet",
) -> float | None:
    """Score consistency between critique and fix via Claude subprocess."""
    prompt = _build_consistency_prompt(php_code, critique_text)
    result = generate_json(prompt, system=JUDGE_SYSTEM, model=model)
    if result is None:
        return None
    return float(result.get("consistency_score", 0.0))
```

**Content-hash cache** (D-09-05; new code, no analog):
```python
import hashlib, functools

@functools.lru_cache(maxsize=4096)
def _cache_key(php_code: str, critique_text: str) -> str:
    return hashlib.sha256(
        (php_code[:512] + critique_text[:512]).encode()
    ).hexdigest()

_score_cache: dict[str, float] = {}

def score_with_cache(php_code, critique_text, model="sonnet"):
    key = _cache_key(php_code, critique_text)
    if key in _score_cache:
        return _score_cache[key]
    score = score_judge_consistency(php_code, critique_text, model)
    if score is not None:
        _score_cache[key] = score
    return score
```

**Subprocess command** (from `claude_agent.py` lines 126–137):
```python
cmd = [
    "claude",
    "--print",
    "--no-session-persistence",
    "--tools", "",
    "--model", model,
]
# Prompt via stdin always (avoids ARG_MAX). See _generate_via_stdin().
```

**Retry/backoff** (from `claude_agent.py` lines 76–101):
```python
# Retry on: "overloaded", "rate limit", "529", "503" in stderr
# Backoff: min(2 ** attempt, 30) seconds
# After MAX_RETRIES=3: raise RuntimeError
```

**Claude consistency reward is NOT in `reward_pipeline.py`.** `compute_group_rewards(php_codes, judge_client, judge_model)` uses the frozen wp_judge model only (`judge_score_single`). The Claude consistency signal is computed separately in `rl_judge_dispatch.py` and combined with fix-correctness in new Phase 9 code before reward normalization. Do not modify `reward_pipeline.py`.

---

### `.claude/skills/wp-finetune:run-rl-training/SKILL.md` (skill-orchestrator, event-driven)

**Analog:** `.claude/skills/wp-finetune:run-training/SKILL.md` — structural template only. Implementation is STALE (DGX/Docker/Unsloth). Reframe all DGX steps to Tinker.

**Structure to copy** (from `run-training/SKILL.md`):
```
TRIGGER → validate inputs → ensure_ready → spawn telemetry agent →
  train loop (call Python script) → stop telemetry → checkpoint verification →
  write metrics summary → return experiment path
```

**Agent lifecycle pattern** (from `run-training/SKILL.md`):
```
# Spawn long-running monitor before Python script:
Agent(model="sonnet", description="Monitor RL training", run_in_background=true)

# Stop signal file (monitor polls this):
touch {TDIR}/_stop   # end monitor when training finishes
```

**Experiment naming convention** (from `run-training/SKILL.md`):
```
{model_short}_rl_experiment_{N:03d}_{date}
# e.g., qwen3-30b_rl_experiment_001_20260620
```

**Venue reframe for Phase 9** (replace all DGX references):
```
STALE → REPLACE
"DGX node"          → "Tinker cloud via ServiceClient"
"unsloth-headless"  → "tc = sc.create_lora_training_client(...)"
"GPU VRAM check"    → "Tinker token/quota check"
"docker exec"       → "tc.forward_backward() + tc.optim_step()"
"checkpoint dir"    → "tc.save_weights_for_sampler(name=..., ttl_seconds=None)"
"loss curve"        → "rl_reward_mean, kl_sample_train_v2, e_max_violation:max"
```

**Observe agent reframe** (from `observe-training/SKILL.md` — structure reusable):
```
STALE → REPLACE
"container logs"    → "Tinker client health / step metrics dict"
"GPU watts/temp"    → "e_frac_with_tokens:mean, e_max_violation:mean"
"loss"              → "rl_reward_mean, kl_divergence"
"checkpoint file"   → "sampler_path from manifest"
```

---

### `scripts/tinker_rl_data.py` (data-adapter, transform)

**Analog:** `scripts/tinker_reasoning_data.py` — exact structural match (same cookbook, same model)

**Copy this block verbatim** (lines 21–46) and change paths + `train_on_what`:
```python
from tinker_cookbook import renderers
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

BASE_MODEL     = "Qwen/Qwen3-30B-A3B"
RENDERER_NAME  = "qwen3_disable_thinking"
MAX_LENGTH     = 8192
# RL-specific paths:
GEN_TRAIN_PATH   = "data/rl_prompts/wp_gen_train.jsonl"
JUDGE_TRAIN_PATH = "data/rl_prompts/wp_judge_train.jsonl"
```

**RL difference:** For RL rollouts, load prompts only (user turns). Set `train_on_what=TrainOnWhat.NONE` or use raw tokenizer — the model generates completions at sampling time. Confirm cookbook supports prompt-only mode; fallback: load JSONL manually and tokenize with `get_tokenizer(BASE_MODEL)`.

---

### `data/rl_prompts/wp_gen_train.jsonl` + `wp_judge_train.jsonl` (data, batch)

**Schema analog:** `data/reasoning_dataset/openai_train.jsonl` (OpenAI chat format)

**Schema** (copy from `tinker_reasoning_data.py` comment + SFT data):
```json
{"messages": [
  {"role": "user", "content": "<wp_gen> Generate WordPress plugin code for: {task}"},
  {"role": "assistant", "content": ""}
]}
```
For judge prompts:
```json
{"messages": [
  {"role": "user", "content": "<wp_judge> Review and fix this code:\n{php_code}"},
  {"role": "assistant", "content": ""}
]}
```
Assistant content is empty — completions generated at sampling time, not pre-filled.

---

### `output/rl_checkpoints/checkpoint_manifest.json` (artifact, CRUD)

**Analog:** `tinker_reasoning_sft.py` manifest pattern (lines 74–78, 227–232)

**Schema** (extend SFT manifest with RL fields):
```json
{
  "experiment": "qwen3-30b_rl_experiment_001_20260620",
  "base_model": "Qwen/Qwen3-30B-A3B",
  "started_at": "2026-06-20T...",
  "checkpoints": [
    {
      "step": "qwen3-30b_rl_experiment_001_20260620-step100",
      "sampler_path": "/tinker/samplers/...",
      "ts": "2026-06-20T...",
      "rl_reward_mean": 0.42,
      "kl_v2": 0.018
    }
  ],
  "final_sampler_path": "...",
  "status": "complete"
}
```

---

### `output/rl_checkpoints/metrics/rl_metrics.jsonl` (sink, streaming)

**Schema analog:** `observe-training/SKILL.md` JSONL schema (reframed for RL)

**Per-step schema** (one JSON object per line):
```json
{
  "ts": "2026-06-20T10:00:00Z",
  "step": 100,
  "rl_reward_mean": 0.42,
  "rl_reward_std": 0.15,
  "reward_breakdown": {
    "phpcs": 0.31,
    "verpo": 0.48,
    "judge_consistency": 0.55
  },
  "kl_sample_train_v1": 0.012,
  "kl_sample_train_v2": 0.018,
  "entropy": 2.31,
  "e_frac_with_tokens_mean": 0.61,
  "e_max_violation_mean": 0.003,
  "e_max_violation_max": 0.009,
  "n_groups": 8,
  "n_constant_groups_dropped": 1
}
```

---

### `tests/test_rl_train.py` (test, request-response)

**Analog:** `tests/test_reward_pipeline.py` — exact match on test structure

**Class-based with lazy imports** (from `test_reward_pipeline.py` lines 37–80):
```python
class TestRLTrainUnit:
    def test_kl_halt_triggers(self, monkeypatch):
        import scripts.rl_train as rt  # lazy import inside method
        # monkeypatch tc.forward_backward to return mock out
        ...

    def test_checkpoint_manifest_written(self, tmp_path, monkeypatch):
        import scripts.rl_train as rt
        ...
```

**Method names embed -k keywords** (so `pytest -k kl_halt` works):
```python
# test_kl_halt_triggers, test_constant_groups_dropped,
# test_mogrpo_norm, test_cache_hit_skips_subprocess,
# test_manifest_schema_valid
```

**All imports lazy** (inside method body, not at module top — avoids collection failure when Tinker not installed).

**Fixture reuse** (from `tests/conftest.py` lines 1–79):
```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture
def mock_tinker_client():
    tc = MagicMock()
    fb_out = MagicMock()
    fb_out.metrics = {
        "e_frac_with_tokens:mean": 0.6,
        "e_max_violation:mean": 0.002,
        "e_max_violation:max": 0.008,
    }
    fb_out.training_logprobs = []
    tc.forward_backward.return_value = fb_out
    tc.optim_step.return_value = None
    tc.save_weights_for_sampler.return_value.path = "/fake/sampler"
    return tc
```

---

## Shared Patterns

### Tinker Auth / ServiceClient Init
**Source:** `scripts/tinker_reasoning_sft.py` lines 194–201
**Apply to:** `scripts/rl_train.py`

No explicit token management needed — `tinker.ServiceClient()` reads credentials from environment or `~/.tinker/credentials` automatically. Do not hardcode tokens.

### Persistent Checkpoint (ttl_seconds=None)
**Source:** `scripts/tinker_reasoning_sft.py` lines 227–232
**Apply to:** `scripts/rl_train.py` — every `_save_checkpoint()` call

```python
sampler_path = _res(tc.save_weights_for_sampler(name=ep_name, ttl_seconds=None)).path
```
`ttl_seconds=None` = no expiry. Ephemeral alternative (`save_weights_and_get_sampling_client()`) must NOT be used for experiment checkpoints — rollout client is different from checkpoint client.

### Future Resolver
**Source:** `scripts/tinker_reasoning_sft.py` lines 58–60
**Apply to:** All `tc.*` calls that may return a Future

```python
def _res(f):
    return f.result() if hasattr(f, "result") else f
```

### Security Gate / Fail-Closed Pattern
**Source:** `scripts/reward_pipeline.py` lines 149–157
**Apply to:** `scripts/rl_rollouts.py` when consuming RewardResult

```python
# Security gate: if scalar == 0.0, do NOT zero-out advantage — drop entire group
# reward_pipeline returns scalar=0.0 on security trigger; treat as invalid sample
# Use remove_constant_reward_groups() AFTER filtering security-zero groups
```

### Claude Subprocess Dispatch
**Source:** `scripts/claude_agent.py` lines 46–101, 126–170
**Apply to:** `scripts/rl_judge_dispatch.py`

Pattern: `generate_json(prompt, system=..., model="sonnet")` — subprocess `claude --print --no-session-persistence --tools ""`. Retry on overloaded/rate-limit. Do NOT use Anthropic API directly.

### MO-GRPO Normalization
**Source:** `scripts/reward_pipeline.py` lines 205–220
**Apply to:** `scripts/rl_rollouts.py` per-signal normalization before scalar combination

```python
def _mo_grpo_norm(values: np.ndarray) -> np.ndarray:
    mu    = values.mean()
    sigma = values.std(ddof=0)
    return (values - mu) / (sigma + 1e-8)
```

### Protected Expert Jaccard Monitor (every-N-steps)
**Source:** Phase 7 `scripts/extract_protected_mask.py` + `scripts/compute_concentration.py`
**Apply to:** `scripts/rl_train.py` — every-N-step check (not per-step)

D-09-06: router gates frozen on Tinker (no `train_router` in LoraConfig). Monitor-only mode:
- Per-step: read `ForwardBackwardOutput.metrics["e_frac_with_tokens:mean"]`, `e_max_violation:max`
- Every-N steps: load `output/profiling/reasoning-merged-v4/protected_expert_mask.npy` (shape `[48, 128]`, bool — verified Phase 7 artifact location), compute Jaccard of active experts vs mask, log to `rl_metrics.jsonl`
- No enforcement action — alert only (GRPO-08 monitoring)

### KL Halt Guard (GRPO-04)
**Source:** `tinker_cookbook/rl/metrics.py` — `compute_kl_sample_train()` return keys
**Apply to:** `scripts/rl_train.py` main loop

```python
kl = metrics["optim/kl_sample_train_v1"]  # v1 (mean logprob diff) = GRPO-08 halt signal for Phase 9; v2 logged for comparison
if kl > KL_HALT_THRESHOLD:  # Phase 9 finalized: soft 0.1 / hard 0.3 (see 09-05 notes)
    # save emergency checkpoint then break
```

### JSONL Metrics Sink
**Source:** `observe-training/SKILL.md` streaming schema
**Apply to:** `scripts/rl_train.py` `_log_step()`

```python
def _log_step(step, rewards, kl_metrics, moe_metrics, args):
    record = {"ts": datetime.utcnow().isoformat(), "step": step, ...}
    with open(METRICS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
```

---

## No Analog Found

| File / Pattern | Role | Data Flow | Reason |
|---|---|---|---|
| `forward_backward_custom(...)` call for GSPO sequence-level IS | SDK call | streaming | No project use of `forward_backward_custom`; Tinker SDK docs required |
| RSPO stop-gradient IS ratio clamp (`min=1.0`) | algorithm | transform | No sequence-level IS in any project script |
| Interleaved batch sampler (60/40 judge/gen ratio) | utility | batch | No interleaved multi-task RL sampler exists; write from scratch |
| `ROADMAP.md` Phase 9 venue text patch | docs | CRUD | Simple text edit; no pattern needed |

---

## SDK / Cookbook Reference (Read-Only)

These files are in `.venv-tinker` — cite by symbol only, do not copy file paths as project source:

| Symbol | File | Use |
|---|---|---|
| `compute_kl_sample_train(data_D, training_logprobs_D)` | `tinker_cookbook/rl/metrics.py:153` | KL + entropy per step |
| `compute_advantages(trajectory_groups_P)` | `tinker_cookbook/rl/data_processing.py:~80` | group-centered advantages |
| `remove_constant_reward_groups(trajectory_groups_P)` | `tinker_cookbook/rl/data_processing.py:~120` | drop zero-gradient groups |
| `assemble_training_data(trajectory_groups_P, advantages_P)` | `tinker_cookbook/rl/data_processing.py:~160` | flat Datum list with metadata |
| `incorporate_kl_penalty(advantages, kl_coeff, ...)` | `tinker_cookbook/rl/metrics.py:~200` | optional KL penalty on advantages |
| `FromConversationFileBuilder` | `tinker_cookbook/supervised/data.py` | JSONL → cookbook dataset |

---

## Metadata

**Analog search scope:** `scripts/`, `tests/`, `.claude/skills/`, `.venv-tinker/lib/python3.13/site-packages/tinker_cookbook/rl/`
**Project files scanned:** 12 analog files read
**Pattern extraction date:** 2026-06-20

## PATTERN MAPPING COMPLETE
