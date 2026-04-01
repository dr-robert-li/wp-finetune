# Skill: wp-finetune:adaptive-planner

Adaptive resource planning (v4.0). Invoked by run-training between runs.
Thin wrapper around `scripts/adaptive_planner.py` — all decision logic is in tested Python.

**Not user-invocable.** Called by run-training at Step 8.5.

## Trigger

Called by run-training with variables:
- `$THERMAL_LOG` — path to the canonical JSONL file for the completed run
- `$RATIO` — current ratio just completed (e.g., "30_70")
- `$TELEMETRY` — boolean; if false, skip immediately

## Step 1: Guard clause

If `$TELEMETRY` is false, print skip message and exit:
```python
print("Telemetry disabled — skipping adaptive planning")
sys.exit(0)
```

Verify telemetry modules are importable:
```python
import sys
try:
    from telemetry.probe import prepare_probe, evaluate_probe
    from telemetry.anchor_store import AnchorStore
    from telemetry.sampler import GPUSampler
    from telemetry.failure_classifier import classify_failure
except ImportError as e:
    print(f"ERROR: telemetry package not available: {e}")
    print("Check that ~/dgx-toolbox/telemetry is mounted at /workspace/dgx-toolbox/telemetry")
    print("and PYTHONPATH includes /workspace/dgx-toolbox")
    sys.exit(1)
```

## Step 2: Parse telemetry and load config

```python
from scripts.adaptive_planner import parse_telemetry_jsonl, classify_power_zone, apply_ladder, compute_batch_ceiling
import yaml
import json
from pathlib import Path

# Parse canonical JSONL
summary = parse_telemetry_jsonl(Path(THERMAL_LOG))
if summary is None:
    print("Insufficient telemetry (fewer than 3 valid readings after warm-up skip) -- skipping adaptive planning")
    sys.exit(0)

# Load configs
config = yaml.safe_load(Path("config/train_config.yaml").read_text())
thresholds = yaml.safe_load(Path("config/adaptive_planning.yaml").read_text())
```

## Step 3: Check Unsloth overrides (BTCH-03)

Unsloth may silently change batch_size/grad_accum at build time to satisfy hardware constraints.
If actuals differ from config, use the actual values as the basis for this planning cycle:

```python
actuals_path = Path("telemetry/training/_unsloth_actuals.json")
if actuals_path.exists():
    actuals = json.loads(actuals_path.read_text())
    config["training"]["per_device_train_batch_size"] = actuals["actual_batch"]
    config["training"]["gradient_accumulation_steps"] = actuals["actual_grad_accum"]
    print(f"Unsloth actuals applied: batch={actuals['actual_batch']}, grad_accum={actuals['actual_grad_accum']}")
    actuals_path.unlink()  # consumed — clear for next run
```

## Step 4: Read failure classification (TELE-03)

```python
from telemetry.failure_classifier import classify_failure

class_path = Path("telemetry/training/_run_classification.json")
if class_path.exists():
    classification = json.loads(class_path.read_text())
    failure = classification["classification"]
    class_path.unlink()  # consumed
else:
    failure = "clean"

# HANG: driver-level issue, NOT memory-related — log only, no config change (TELEM-14)
if failure == "hang":
    print("HANG detected -- this is a driver-level issue, not OOM. No config change.")
    from telemetry.anchor_store import AnchorStore
    store = AnchorStore(Path(thresholds["anchors"]["store_path"]))
    hash_config = {
        "model_id": config["model"]["name"],
        "quant_mode": "bf16",
        "framework": "pytorch",
        "grad_ckpt": "full",
        "lora_rank": config["lora"]["r"],
        "seq_len": config["model"]["max_seq_length"],
        "optimizer": "adamw",
        "batch_size": config["training"]["per_device_train_batch_size"],
        "grad_accum": config["training"]["gradient_accumulation_steps"],
    }
    h = store.compute_config_hash(hash_config)
    # HANG record: no batch_cap key (per TELEM-14 — batch cap is inapplicable for hangs)
    store.apply_override(h, "HANG",
                         batch_size=config["training"]["per_device_train_batch_size"],
                         tier_cap=0)
    sys.exit(0)
```

## Step 5: Classify power zone and apply ladder

```python
from scripts.adaptive_planner import compute_batch_ceiling

ceiling = compute_batch_ceiling(config["model"], config["lora"])
batch = config["training"]["per_device_train_batch_size"]
has_batch_headroom = batch < ceiling["batch_cap"]
has_mem_headroom = summary["effective_headroom_gb"] > thresholds["memory"]["jitter_margin_gb"]

zone = classify_power_zone(
    avg_watts=summary.get("avg_watts"),
    peak_temp=summary["peak_temp"],
    avg_gpu_util=summary["avg_gpu_util"],
    thresholds=thresholds,
    has_batch_headroom=has_batch_headroom,
    has_mem_headroom=has_mem_headroom,
)

print(f"Power zone: {zone} (avg_watts={summary.get('avg_watts')}, peak_temp={summary['peak_temp']}C, "
      f"avg_gpu_util={summary['avg_gpu_util']:.1f}%)")

# Build anchor_lookup_fn for apply_ladder (PROB-02)
def anchor_lookup(proposed_batch: int):
    from telemetry.anchor_store import AnchorStore
    store = AnchorStore(Path(thresholds["anchors"]["store_path"]))
    current_effective = batch * config["training"]["gradient_accumulation_steps"]
    # Compute new grad_accum for proposed batch (round-based, per BTCH-01)
    from scripts.adaptive_planner import couple_batch_grad_accum
    new_grad_accum = couple_batch_grad_accum(current_effective, proposed_batch)
    hash_config = {
        "model_id": config["model"]["name"],
        "quant_mode": "bf16",
        "framework": "pytorch",
        "grad_ckpt": "full",
        "lora_rank": config["lora"]["r"],
        "seq_len": config["model"]["max_seq_length"],
        "optimizer": "adamw",
        "batch_size": proposed_batch,
        "grad_accum": new_grad_accum,
    }
    h = store.compute_config_hash(hash_config)
    return store.lookup(h)

# Apply thermal exploitation ladder — all rung logic is in Python (not here)
changes = apply_ladder(
    current_config={
        "per_device_train_batch_size": config["training"]["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "dataloader_prefetch_factor": config["training"].get("dataloader_prefetch_factor", 2),
        "dataloader_num_workers": config["training"].get("dataloader_num_workers", 2),
        "save_steps": config["training"].get("save_steps", 200),
        "eval_steps": config["training"].get("eval_steps", 100),
        "dataloader_pin_memory": config["training"].get("dataloader_pin_memory", False),
        "dataloader_persistent_workers": config["training"].get("dataloader_persistent_workers", False),
    },
    power_zone=zone,
    thresholds=thresholds,
    telemetry_summary=summary,
    batch_ceiling=ceiling["batch_cap"],
    anchor_lookup_fn=anchor_lookup,
)

print(f"Rungs applied: {changes.get('rungs_applied', [])}")
```

## Step 6: Warmup probe protocol (PROB-01)

Batch size increases require a warmup probe before full training (UMA deadlock risk on DGX Spark).

**Check for pending probe results first (from a previous run's probe):**

```python
probe_results_path = Path("telemetry/training/_probe_results.json")
if probe_results_path.exists():
    from telemetry.probe import evaluate_probe
    from telemetry.sampler import GPUSampler

    results_meta = json.loads(probe_results_path.read_text())
    baseline_sample = GPUSampler().sample()
    eval_result = evaluate_probe(
        results_path=Path(results_meta["results_path"]),
        baseline={"mem_available_gb": baseline_sample["mem_available_gb"]},
        tier_headroom_pct=ceiling["min_headroom_pct"],
        jitter_margin_gb=thresholds["memory"]["jitter_margin_gb"],
    )
    probe_results_path.unlink()  # consumed

    if eval_result["action"] == "commit":
        print(f"Probe PASSED: {eval_result['reason']}")
        # changes is already the probed config — proceed to Step 7
    else:
        print(f"Probe FAILED: {eval_result['reason']} -- reverting, no config change")
        changes = {}  # clear changes — stay on current config
```

**Flag a new probe if batch increase is needed:**

```python
if changes.get("per_device_train_batch_size") and \
        changes["per_device_train_batch_size"] > config["training"]["per_device_train_batch_size"]:
    from telemetry.probe import prepare_probe

    probe_result = prepare_probe(
        current_config={
            "batch_size": batch,
            "grad_accum": config["training"]["gradient_accumulation_steps"],
        },
        proposed_changes={
            "batch_size": changes["per_device_train_batch_size"],
            "grad_accum": changes["gradient_accumulation_steps"],
        },
    )
    # Write probe flag for run-training Step 6 to execute 3-5 steps
    Path("telemetry/training/_warmup_probe_required").write_text(json.dumps({
        "probe_config_path": str(probe_result["probe_config_path"]),
        "rollback_config_path": str(probe_result["rollback_config_path"]),
        "results_path": str(probe_result["results_path"]),
        "proposed_batch": changes["per_device_train_batch_size"],
        "proposed_accum": changes["gradient_accumulation_steps"],
    }))
    print(f"Warmup probe flagged: batch {batch} -> {changes['per_device_train_batch_size']}")
    print("run-training Step 6 will execute 3-5 steps before the next full training run")
    # Do NOT apply batch change yet — wait for probe to commit or revert
    changes.pop("per_device_train_batch_size", None)
    changes.pop("gradient_accumulation_steps", None)
```

## Step 7: Write config and record anchor (PROB-02)

```python
if changes and changes.get("rungs_applied"):
    # Apply all non-batch changes from apply_ladder result
    if "per_device_train_batch_size" in changes:
        config["training"]["per_device_train_batch_size"] = changes["per_device_train_batch_size"]
    if "gradient_accumulation_steps" in changes:
        config["training"]["gradient_accumulation_steps"] = changes["gradient_accumulation_steps"]
    if "dataloader_prefetch_factor" in changes:
        config["training"]["dataloader_prefetch_factor"] = changes["dataloader_prefetch_factor"]
    if "dataloader_num_workers" in changes:
        config["training"]["dataloader_num_workers"] = changes["dataloader_num_workers"]
    if "save_steps" in changes:
        config["training"]["save_steps"] = changes["save_steps"]
    if "eval_steps" in changes:
        config["training"]["eval_steps"] = changes["eval_steps"]
    if "dataloader_pin_memory" in changes:
        config["training"]["dataloader_pin_memory"] = changes["dataloader_pin_memory"]
    if "dataloader_persistent_workers" in changes:
        config["training"]["dataloader_persistent_workers"] = changes["dataloader_persistent_workers"]

    # Write updated config
    with open("config/train_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=True)
    print("Updated config/train_config.yaml")

    # Record in anchor store (PROB-02)
    from telemetry.anchor_store import AnchorStore
    store = AnchorStore(Path(thresholds["anchors"]["store_path"]))
    hash_config = {
        "model_id": config["model"]["name"],
        "quant_mode": "bf16",
        "framework": "pytorch",
        "grad_ckpt": "full",
        "lora_rank": config["lora"]["r"],
        "seq_len": config["model"]["max_seq_length"],
        "optimizer": "adamw",
        "batch_size": config["training"]["per_device_train_batch_size"],
        "grad_accum": config["training"]["gradient_accumulation_steps"],
    }
    h = store.compute_config_hash(hash_config)
    status = failure.upper() if failure != "clean" else "COMPLETED"
    store.apply_override(h, status,
                         batch_size=config["training"]["per_device_train_batch_size"],
                         tier_cap=ceiling["batch_cap"])

    # Log to adaptive_adjustments.md
    import datetime
    log_entry = (
        f"\n### Adaptive adjustment after {RATIO} "
        f"({datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')})\n"
        f"- Power zone: {zone}\n"
        f"- Failure classification: {failure}\n"
        f"- Rungs applied: {changes.get('rungs_applied', [])}\n"
    )
    # Append reason lines
    for key in changes:
        if key.startswith("reason_"):
            log_entry += f"- {key[7:]}: {changes[key]}\n"
    log_entry += f"- Telemetry: avg_watts={summary.get('avg_watts')}, peak_temp={summary['peak_temp']}C, avg_gpu_util={summary['avg_gpu_util']:.1f}%\n"

    with open("telemetry/training/adaptive_adjustments.md", "a") as f:
        f.write(log_entry)
    print(log_entry)
else:
    print(f"No config changes needed (zone={zone}, rungs_applied=[])")

print("Adaptive planning complete")
```

## Troubleshooting

**`ImportError: No module named 'telemetry'`**
- PYTHONPATH must include `/workspace/dgx-toolbox` (set via `container_env` in dgx_toolbox.yaml)
- The telemetry package lives at `~/dgx-toolbox/telemetry/telemetry/`
- Mount path: `~/dgx-toolbox/telemetry:/workspace/dgx-toolbox/telemetry` (see `extra_mounts.dgx_telemetry`)

**`ImportError: No module named 'scripts.adaptive_planner'`**
- PYTHONPATH must include `/workspace/wp-finetune`
- This is the project root; the scripts/ directory contains adaptive_planner.py

**AnchorStore AttributeError on write operations**
- The store only exposes: `apply_override(hash, status, batch_size, tier_cap)` to write
- Use `lookup(hash)` to query existing records
- Use `compute_config_hash(config_dict)` to get the hash
- No other mutation methods exist on this class

**`AttributeError: 'telemetry.probe' has no attribute '...'`**
- The probe module only exposes two functions: `prepare_probe(current_config, proposed_changes)` and `evaluate_probe(results_path, baseline, ...)`
- Do not call any other function on telemetry.probe — only these two exist

**`telemetry/training/_warmup_probe_required` not cleared**
- This file is created by adaptive-planner and consumed by run-training Step 6
- If training was aborted before Step 6 ran, delete it manually before re-running
