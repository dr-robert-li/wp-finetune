# Skill: wp-finetune:run-profiling

Run the Phase 7 router profiling pipeline on DGX Spark. Profiles the merged
v1.2 model (`models/qwen3-30b-wp-30_70-reasoning-merged-v4`) to capture per-expert
routing counts split by wp_gen and wp_judge tokens, compute Jaccard subsample
stability (PROF-03), concentration metrics (PROF-04), and extract the D-03
protected-expert mask for Phase 8 reward infrastructure.

## Architecture

```
Skill (this file — intent + decision logic)
  → scripts/profile_merged_model.py   (GPU forward pass, hook registration, Jaccard sidecar)
  → scripts/compute_concentration.py  (PROF-04 metrics + bootstrap CI + PROF-03 CI gate)
  → scripts/extract_protected_mask.py (D-03 co-activation mask + D-04 sensitivity table)
    → Output: output/profiling/reasoning-merged-v4/
      - routing_report.jsonl           (per-layer expert counts + E_eff)
      - jaccard_stability.json         (raw 48-element per-layer Jaccard array)
      - concentration_report.json      (PROF-04 metrics + jaccard_ci_lower gate)
      - protected_expert_mask.npy      ([48, 128] bool array)
      - protected_expert_mask.json     ({layer_idx: [expert_ids]} sidecar)
      - sensitivity_table.json         (D-04 three-threshold comparison)
      - protected_mask_result.json     (full D-03 analysis)
```

## Telemetry

> **Default: Lightweight monitor only.** The observe-evaluation agent team (3 agents,
> ~1.2 GB overhead) should NOT be spawned during profiling if memory headroom is <25 GB.
> Profiling loads the 30B merged model **directly** (no vLLM, no LoRA), consuming ~60 GB
> VRAM — even tighter than evaluation. Every Python agent (~0.4 GB each) competes
> directly for the same unified memory pool.
>
> The lightweight monitor captures watts, temp, util, and memory — sufficient for
> profiling telemetry. Only spawn full observe agents if profiling is running standalone
> with >25 GB headroom after model load.
>
> **Memory impact reference:**
> | Mode | Processes | Memory | When to use |
> |------|-----------|--------|-------------|
> | Lightweight | 1 shell script | ~5 MB | Default — always safe |
> | Observe | 3 Python agents | ~1.2 GB | Standalone profiling, >25 GB headroom post-load |

## Trigger

User says: "run profiling", "profile the merged model", "run router profiling", "/run-profiling"

## Process

### Step 0: Readiness Checks

#### 0a. Verify DGX GPU availability

```bash
nvidia-smi 2>&1 | head -1
python3 -c "import torch; print('cuda:', torch.cuda.is_available())"
```

All three profiling scripts require CUDA. The `--allow-cpu` override exists but is NOT
recommended for a 30B forward pass. If CUDA is unavailable, stop and direct user to
open the DGX `ngc-pytorch` container before proceeding.

#### 0b. Verify merged model and baseline exist

```bash
ls models/qwen3-30b-wp-30_70-reasoning-merged-v4/config.json 2>/dev/null \
  && echo "merged model: ready" || echo "merged model: MISSING"

ls output/profiling/base_model_eeff.jsonl 2>/dev/null \
  && echo "baseline: ready" || echo "baseline: MISSING — run profile_base_model.py first"
```

The baseline `base_model_eeff.jsonl` must exist for the D-08 E_eff delta join in
`compute_concentration.py`. If absent, run `profile_base_model.py` on the base model first.

#### 0c. Check idempotency markers (resume from last completed step)

```bash
ls output/profiling/reasoning-merged-v4/.profile_complete 2>/dev/null \
  && echo "profile: DONE" || echo "profile: PENDING"

ls output/profiling/reasoning-merged-v4/.concentration_complete 2>/dev/null \
  && echo "concentration: DONE" || echo "concentration: PENDING"

ls output/profiling/reasoning-merged-v4/.mask_complete 2>/dev/null \
  && echo "mask: DONE" || echo "mask: PENDING"
```

Re-running the skill resumes from the last incomplete step automatically.
Use `--force` equivalent: delete the `.complete` markers to re-run a step.

### Step 1: Run Profile Pass (inside DGX ngc-pytorch container)

**CRITICAL: `profile_merged_model.py` runs INSIDE the DGX `ngc-pytorch` container (CUDA required).
Orchestration, `.complete` checks, and monitoring run on HOST.**

```bash
# Run inside container — DGX Toolbox manages lifecycle
bash deps/dgx-toolbox/containers/ngc-pytorch.sh python3 -m scripts.profile_merged_model \
  --model-path models/qwen3-30b-wp-30_70-reasoning-merged-v4 \
  --ratio ratio_30_70 \
  --output-dir output/profiling/reasoning-merged-v4
```

This runs the FULL-set reference pass (~34K examples of ratio_30_70) followed by a
single 10% subsample pass for per-layer Jaccard stability (D-06 literal). Both passes
use the same matched stimulus that produced `base_model_eeff.jsonl`.

On completion, touch the idempotency marker from HOST:

```bash
touch output/profiling/reasoning-merged-v4/.profile_complete
```

**Lightweight monitor (run in background on HOST during Step 1):**

```bash
# Start monitor in background before launching container
while true; do
  nvidia-smi --query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used \
    --format=csv,noheader,nounits >> output/profiling/reasoning-merged-v4/gpu_monitor.log
  sleep 30
done &
MONITOR_PID=$!

# ... run container step above ...

kill $MONITOR_PID 2>/dev/null
```

**Decision Gate: Jaccard Stability (PROF-03 point estimate)**

After Step 1 completes, check the raw Jaccard sidecar:

```bash
python3 -c "
import json, numpy as np
d = json.load(open('output/profiling/reasoning-merged-v4/jaccard_stability.json'))
j = np.array(d['per_layer_jaccard'])
print(f'Jaccard: mean={j.mean():.4f}, min={j.min():.4f}')
print(f'Point gate (>=0.94): {np.all(j >= 0.94)}')
"
```

The CI-aware disposition (`jaccard_ci_lower >= 0.94`) is computed in Step 2
(`compute_concentration.py`). A point-estimate failure here is an early warning;
the CI gate in Step 2 is definitive.

### Step 2: Compute Concentration Metrics (HOST — no GPU required)

```bash
python3 -m scripts.compute_concentration \
  --merged-jsonl output/profiling/reasoning-merged-v4/routing_report.jsonl \
  --base-jsonl output/profiling/base_model_eeff.jsonl \
  --jaccard-json output/profiling/reasoning-merged-v4/jaccard_stability.json \
  --output output/profiling/reasoning-merged-v4/concentration_report.json
```

This step computes PROF-04 metrics (CV, cumulative coverage, depth skew, E_eff delta)
and the CI-aware PROF-03 gate (`jaccard_ci_lower >= 0.94`). If the gate fails:

- `concentration_report.json` will contain `"jaccard_gate_passes": false`
- **D-06 fallback**: re-run Step 1 with a larger subsample (e.g. `--subsample 0.25` or
  `--subsample 1.0` = full-set-only reference with 25% test subsample)
- Do NOT proceed to Step 3 until the PROF-03 CI gate passes

On pass:

```bash
touch output/profiling/reasoning-merged-v4/.concentration_complete
```

**Present concentration summary:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PROFILING ► CONCENTRATION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PROF-03 Jaccard CI gate: {PASS/FAIL}
  jaccard_ci_lower: {val} (threshold: >= 0.94)

  E_eff total:   mean={val}  max={val}  CI=[{lo}, {hi}]
  E_eff delta:   mean={val}  (negative = more concentrated than base)
  CV mean:       {val}  CI=[{lo}, {hi}]
  Layer-depth skew (early/late CV): {val}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Step 3: Extract Protected-Expert Mask (HOST — no GPU required)

```bash
python3 -m scripts.extract_protected_mask \
  --merged-jsonl output/profiling/reasoning-merged-v4/routing_report.jsonl \
  --output-dir output/profiling/reasoning-merged-v4
```

This implements the D-03 conservative co-activation rule and the D-04 sensitivity table.
Outputs: `protected_expert_mask.npy`, `protected_expert_mask.json`, `sensitivity_table.json`.

On completion:

```bash
touch output/profiling/reasoning-merged-v4/.mask_complete
```

**Present mask summary:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PROFILING ► PROTECTED EXPERT MASK (D-03)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Rule: D-03 conservative co-activation (above mean in BOTH gen AND judge)
  Total protected: {n} experts across 48 layers
  Mean per layer:  {n:.1f} / 128

  Sensitivity (D-04):
    mean_threshold:    {n} total (chosen)
    median_threshold:  {n} total
    top-16 intersection: {n} total

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Step 4: Verification and Handoff

After all three steps complete:

```bash
# Verify all required outputs exist
ls output/profiling/reasoning-merged-v4/ | grep -E 'routing_report|jaccard_stability|concentration_report|protected_expert_mask'
```

Expected files:
- `routing_report.jsonl` — per-layer expert counts
- `jaccard_stability.json` — raw Jaccard array (PROF-03 input)
- `concentration_report.json` — PROF-04 + CI gate result
- `protected_expert_mask.npy` and `.json` — D-03 mask
- `sensitivity_table.json` — D-04 variants
- `.profile_complete`, `.concentration_complete`, `.mask_complete`

**Phase 8 handoff:** The protected-expert mask is a hard input to Phase 8 reward
infrastructure. Confirm `protected_expert_mask.npy` shape is `(48, 128)` before signaling
Phase 7 complete.

## CLI Reference

| Script | Key flags | Purpose |
|--------|-----------|---------|
| `profile_merged_model.py` | `--model-path`, `--ratio ratio_30_70`, `--subsample 0.10`, `--output-dir` | GPU forward pass, Jaccard sidecar |
| `compute_concentration.py` | `--merged-jsonl`, `--base-jsonl`, `--jaccard-json`, `--output` | PROF-04 + PROF-03 CI gate |
| `extract_protected_mask.py` | `--merged-jsonl`, `--output-dir`, `--top-k-mask` | D-03 mask + D-04 sensitivity |

All scripts support `--allow-cpu` (profile only) or run on HOST without GPU (concentration, mask).

## Key Rules

1. `profile_merged_model.py` runs INSIDE the DGX `ngc-pytorch` container; all other scripts run on HOST.
2. The PROF-03 CI gate (`jaccard_ci_lower >= 0.94`) is definitive — point-estimate gate is informational only.
3. On PROF-03 gate failure, re-run Step 1 with larger `--subsample` before proceeding (D-06 fallback).
4. The protected-expert mask is immutable once written; subsequent Phase 8/11 runs must not regenerate it unless explicitly re-running Phase 7.
5. Never write to `output/profiling/base_model_eeff.jsonl` (base path collision guard in `profile_merged_model.py`).
