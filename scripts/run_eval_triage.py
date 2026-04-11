"""Phase 4 Eval Triage Orchestrator.

Full pipeline for base-model profiling + sequential adapter eval + triage decision.

Steps:
    0. Setup (output dirs, wp-bench clone)
    1. Base-model E_eff profiling (gradient-free, all experiment datasets)
    2. Sequential adapter eval (vLLM merged-model serving per experiment + full eval suite + wp-bench)
    3. Triage decision (load_eval_results -> triage_ratios -> write_triage_decision)

Adapters and experiments are auto-discovered from disk:
    - adapters/ directory: any subdirectory with adapter_config.json
    - output/eval_triage/ directory: any subdirectory (for resuming)

Idempotency: every major step writes a completion marker (.complete file).
On re-run, steps with existing markers are skipped unless --force is passed.

Usage:
    python scripts/run_eval_triage.py
    python scripts/run_eval_triage.py --skip-wpbench
    python scripts/run_eval_triage.py --experiments qwen3-30b-wp-30_70,qwen3-30b-wp-50_50
    python scripts/run_eval_triage.py --force
    python scripts/run_eval_triage.py --skip-profiling
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("run_eval_triage")

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure project root is on sys.path so `eval` package is importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ADAPTERS_DIR = PROJECT_ROOT / "adapters"
DATASET_BASE = "data/final_dataset"
OUTPUT_DIR = PROJECT_ROOT / "output"
PROFILING_DIR = OUTPUT_DIR / "profiling"
EVAL_TRIAGE_DIR = OUTPUT_DIR / "eval_triage"
WP_BENCH_DIR = PROJECT_ROOT / "wp-bench"

VLLM_HEALTH_TIMEOUT_S = 600   # default: 10 minutes (30B MoE loads ~8 min)
VLLM_HEALTH_POLL_S = 5
EVAL_RETRY_DELAY_S = 30

# Completion markers (use .format(experiment=...) for per-experiment markers)
COMPLETION_MARKERS = {
    "profiling": str(PROFILING_DIR / ".complete"),
    "eval_experiment": str(EVAL_TRIAGE_DIR / "{experiment}" / ".complete"),
    "wpbench_experiment": str(EVAL_TRIAGE_DIR / "{experiment}" / ".wpbench_complete"),
    "triage": str(OUTPUT_DIR / ".triage_complete"),
}


def discover_adapters(adapters_dir: Path = None) -> list[str]:
    """Auto-discover trained adapter directories.

    Returns sorted list of adapter directory names that contain adapter_config.json.
    """
    if adapters_dir is None:
        adapters_dir = ADAPTERS_DIR
    if not adapters_dir.exists():
        return []
    return sorted([
        d.name for d in adapters_dir.iterdir()
        if d.is_dir() and (d / "adapter_config.json").exists()
    ])


def discover_experiments(eval_dir: Path = None) -> list[str]:
    """Auto-discover experiment directories under eval triage output.

    Returns sorted list of experiment directory names (for resuming prior runs).
    """
    if eval_dir is None:
        eval_dir = EVAL_TRIAGE_DIR
    if not eval_dir.exists():
        return []
    return sorted([
        d.name for d in eval_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])


def step_complete(marker_path: str) -> bool:
    """Return True if the completion marker file exists."""
    return Path(marker_path).exists()


def mark_complete(marker_path: str) -> None:
    """Write a completion marker file with timestamp."""
    p = Path(marker_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"completed: {datetime.now().isoformat()}\n")
    logger.info(f"Marked complete: {marker_path}")


def clear_marker(marker_path: str) -> None:
    """Remove a completion marker if it exists."""
    p = Path(marker_path)
    if p.exists():
        p.unlink()
        logger.info(f"Cleared marker: {marker_path}")


def _clean_stale_results(experiments: list[str]) -> None:
    """Remove stale result files and markers so --force starts genuinely fresh.

    Clears per-experiment result files (JSON, JSONL, markers, tmp configs) and the
    triage decision. Profiling results are NOT cleared (they're expensive and
    experiment-independent).
    """
    logger.info("--force: cleaning stale result files ...")
    cleaned = 0

    # Per-experiment result files
    for experiment in experiments:
        exp_dir = EVAL_TRIAGE_DIR / experiment
        if exp_dir.exists():
            for f in exp_dir.iterdir():
                f.unlink()
                cleaned += 1

    # Triage decision
    triage_md = OUTPUT_DIR / "triage_decision.md"
    if triage_md.exists():
        triage_md.unlink()
        cleaned += 1

    # Triage completion marker
    triage_marker = Path(COMPLETION_MARKERS["triage"])
    if triage_marker.exists():
        triage_marker.unlink()
        cleaned += 1

    logger.info(f"--force: removed {cleaned} stale files")


# ---------------------------------------------------------------------------
# Step 0: Setup
# ---------------------------------------------------------------------------


def setup_output_dirs(experiments: list[str]) -> None:
    """Create all output directories."""
    PROFILING_DIR.mkdir(parents=True, exist_ok=True)
    for experiment in experiments:
        (EVAL_TRIAGE_DIR / experiment).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directories ready: {OUTPUT_DIR}")


def _get_training_container_name() -> str:
    """Resolve the training container name from dgx_toolbox.yaml.

    Falls back to 'unsloth-headless' if config lookup fails.
    """
    try:
        from scripts.dgx_toolbox import get_toolbox
        dgx = get_toolbox()
        return dgx._containers.get("unsloth_studio", {}).get("container_name", "unsloth-headless")
    except Exception:
        return "unsloth-headless"


def _get_vllm_container_name() -> str:
    """Resolve the vLLM container name from dgx_toolbox.yaml.

    Falls back to 'vllm' if config lookup fails.
    """
    try:
        from scripts.dgx_toolbox import get_toolbox
        dgx = get_toolbox()
        return dgx._containers.get("vllm", {}).get("container_name", "vllm")
    except Exception:
        return "vllm"


def pre_merge_adapters(experiments: list[str]) -> dict[str, bool]:
    """Pre-merge all adapters on HOST before the eval loop.

    LoRA serving always fails for Qwen3-30B-A3B due to modules_to_save tensors.
    Instead of failing per-experiment inside the eval loop, merge all adapters upfront
    using device_map=cpu (no GPU or container required).

    Returns dict mapping experiment -> success bool.
    """
    logger.info("=" * 60)
    logger.info("PRE-MERGE: Merging adapters for all experiments")
    logger.info("=" * 60)

    merge_script = PROJECT_ROOT / "scripts" / "merge_adapter.py"
    results = {}

    for experiment in experiments:
        merged_path = PROJECT_ROOT / "models" / f"merged-{experiment}"
        adapter_path = ADAPTERS_DIR / experiment

        # Check if already merged and verified
        if (merged_path / "config.json").exists():
            logger.info(f"  {experiment}: merged model already exists at {merged_path}, verifying ...")
            # Quick token verification via merge_adapter.py idempotency check
            result = subprocess.run(
                [sys.executable, str(merge_script),
                 "--adapter-dir", str(adapter_path),
                 "--output-dir", str(merged_path)],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT),
                timeout=120,
            )
            if result.returncode == 0:
                logger.info(f"  {experiment}: verified OK")
                results[experiment] = True
                continue
            else:
                logger.warning(f"  {experiment}: verification failed, re-merging ...")

        # Check adapter exists
        if not (adapter_path / "adapter_config.json").exists():
            logger.error(f"  {experiment}: adapter not found at {adapter_path}")
            results[experiment] = False
            continue

        logger.info(f"  {experiment}: merging {adapter_path} -> {merged_path} (HOST, device_map=cpu) ...")
        result = subprocess.run(
            [sys.executable, str(merge_script),
             "--adapter-dir", str(adapter_path),
             "--output-dir", str(merged_path)],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            timeout=1200,  # 20 min max per merge (30B model is large)
        )
        if result.returncode == 0:
            logger.info(f"  {experiment}: merge succeeded")
            results[experiment] = True
        else:
            logger.error(f"  {experiment}: merge FAILED: {result.stderr[:500]}")
            if result.stdout:
                logger.error(f"  {experiment}: stdout: {result.stdout[:500]}")
            results[experiment] = False

    # Summary
    succeeded = [r for r, ok in results.items() if ok]
    failed = [r for r, ok in results.items() if not ok]
    logger.info(f"Pre-merge complete: {len(succeeded)} succeeded, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed merges: {failed} — these experiments will be skipped during eval")

    return results


def setup_wpbench() -> bool:
    """Clone and set up wp-bench.

    Returns True if wp-bench is available, False otherwise.
    wp-bench failure is non-fatal (D-09: static gates are primary).
    """
    if WP_BENCH_DIR.exists():
        logger.info(f"wp-bench already present at {WP_BENCH_DIR}")
        return True

    logger.info("Cloning wp-bench from github.com/WordPress/wp-bench ...")
    try:
        result = subprocess.run(
            ["git", "clone", "https://github.com/WordPress/wp-bench.git", str(WP_BENCH_DIR)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            logger.error(f"wp-bench clone failed: {result.stderr[:500]}")
            return False
        logger.info("wp-bench cloned successfully")
    except Exception as e:
        logger.error(f"wp-bench clone error: {e}")
        return False

    # Run wp-bench setup to build Docker WordPress runtime image
    setup_script = WP_BENCH_DIR / "setup.sh"
    if not setup_script.exists():
        # Try alternative setup scripts
        for candidate in ["scripts/setup.sh", "Makefile"]:
            candidate_path = WP_BENCH_DIR / candidate
            if candidate_path.exists():
                setup_script = candidate_path
                break

    if setup_script.exists() and setup_script.suffix == ".sh":
        logger.info("Running wp-bench setup script ...")
        try:
            result = subprocess.run(
                ["bash", str(setup_script)],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(WP_BENCH_DIR),
            )
            if result.returncode != 0:
                logger.warning(f"wp-bench setup returned non-zero: {result.stderr[:500]}")
                logger.warning("Continuing -- wp-bench may still work without setup")
        except Exception as e:
            logger.warning(f"wp-bench setup error: {e}")
            logger.warning("Continuing -- wp-bench may still work without setup")
    else:
        logger.info("No setup.sh found for wp-bench -- skipping setup step")

    return True


# ---------------------------------------------------------------------------
# Step 1: Base-Model Profiling
# ---------------------------------------------------------------------------


def run_profiling(
    model_path: str = "models/Qwen3-30B-A3B",
    tokenizer_path: str = "adapters/tokenizer",
    force: bool = False,
) -> Optional[dict]:
    """Run base-model E_eff profiling over all 5 ratio datasets.

    Returns the all_ratio_eeffs dict for downstream use, or None if skipped.

    GPU memory is explicitly reclaimed after profiling completes.
    """
    marker = COMPLETION_MARKERS["profiling"]

    if not force and step_complete(marker):
        logger.info(f"Skipping profiling (already complete, marker: {marker})")
        return None

    logger.info("=" * 60)
    logger.info("STEP 1: Base-model E_eff profiling")
    logger.info("=" * 60)

    # Lazy imports (GPU not needed at import time)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        logger.error(f"Cannot import torch/transformers: {e}")
        logger.error("Profiling requires a GPU environment. Skipping.")
        return None

    from scripts.profile_base_model import (
        discover_dataset_dirs,
        has_downward_eeff_trend,
        profile_base_model,
    )

    abs_model_path = PROJECT_ROOT / model_path
    abs_tokenizer_path = PROJECT_ROOT / tokenizer_path

    if not abs_model_path.exists():
        logger.error(f"Model not found at {abs_model_path}. Cannot profile.")
        raise RuntimeError(f"Model path does not exist: {abs_model_path}")

    if not abs_tokenizer_path.exists():
        logger.warning(
            f"Extended tokenizer not found at {abs_tokenizer_path}. "
            f"Falling back to base tokenizer from model path."
        )
        abs_tokenizer_path = abs_model_path

    # Auto-discover dataset directories
    ratio_data_paths = discover_dataset_dirs(PROJECT_ROOT / DATASET_BASE)

    if not ratio_data_paths:
        raise RuntimeError("No dataset directories found. Cannot profile.")

    logger.info(f"Loading extended tokenizer from {abs_tokenizer_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(abs_tokenizer_path))

    logger.info(f"Loading model from {abs_model_path} (bfloat16) ...")
    # Log GPU memory before loading
    if torch.cuda.is_available():
        free_before = torch.cuda.mem_get_info()[0] / (1024**3)
        logger.info(f"GPU free memory before model load: {free_before:.1f} GB")

    model = AutoModelForCausalLM.from_pretrained(
        str(abs_model_path),
        dtype=torch.bfloat16,
        device_map="auto",
    )

    try:
        logger.info("Starting profiling forward passes ...")
        all_ratio_eeffs = profile_base_model(
            model=model,
            tokenizer=tokenizer,
            ratio_data_paths=ratio_data_paths,
            subsample_frac=0.10,
            batch_size=1,
            max_seq_len=2048,
            output_dir=str(PROFILING_DIR),
        )

        # E_eff trend analysis
        import math
        import numpy as np

        means_total = []
        for experiment in sorted(all_ratio_eeffs.keys()):
            vals = [
                v for v in all_ratio_eeffs[experiment]["eeff_total"]
                if not (isinstance(v, float) and math.isnan(v))
            ]
            means_total.append(float(np.nanmean(vals)) if vals else float("nan"))

        if has_downward_eeff_trend(means_total):
            logger.info(
                "E_eff DOWNWARD TREND DETECTED -- 60/40 training is warranted (D-05). "
                "Start 60/40 training now; its eval runs when training completes (~2 days)."
            )
            print("\n*** E_eff DOWNWARD TREND: 60/40 training warranted (see D-05) ***\n")
        else:
            logger.info("E_eff trend: flat or increasing -- no additional training triggered.")

    finally:
        # GPU memory reclamation (critical -- vLLM needs full 128GB)
        logger.info("Reclaiming GPU memory after profiling ...")
        if torch.cuda.is_available():
            used_before_cleanup = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"GPU memory allocated before cleanup: {used_before_cleanup:.1f} GB")

        del model
        del tokenizer
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_after = torch.cuda.mem_get_info()[0] / (1024**3)
            logger.info(f"GPU free memory after cleanup: {free_after:.1f} GB")

        logger.info("GPU memory reclamation complete")

    mark_complete(marker)
    return all_ratio_eeffs


# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------


def _get_vllm_endpoint() -> str:
    """Get vLLM endpoint URL from DGX Toolbox config."""
    try:
        from scripts.dgx_toolbox import get_toolbox
        dgx = get_toolbox()
        return dgx.vllm_endpoint()
    except Exception:
        return "http://localhost:8020/v1"


def _vllm_health_check(endpoint: str) -> bool:
    """Return True if vLLM health endpoint responds OK."""
    try:
        import urllib.request
        health_url = endpoint.rstrip("/v1").rstrip("/") + "/health"
        req = urllib.request.urlopen(health_url, timeout=5)
        return req.status == 200
    except Exception:
        return False


def _is_vllm_crash_looping() -> bool:
    """Return True if the vLLM container has crashed and restarted.

    Checks RestartCount >= 1 (container already failed once) or docker logs
    contain known fatal errors like LoRA validation failures.
    """
    cname = _get_vllm_container_name()
    try:
        # Check restart count
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}} {{.RestartCount}}", cname],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if len(parts) >= 2:
                status, restart_count = parts[0], int(parts[1])
                if status == "restarting" or restart_count >= 1:
                    logger.info(f"vLLM container status={status} restarts={restart_count}")
                    return True

        # Check logs for known fatal errors (LoRA validation, OOM)
        logs = subprocess.run(
            ["docker", "logs", "--tail", "50", cname],
            capture_output=True, text=True, timeout=10,
        )
        if logs.returncode == 0:
            output = logs.stdout + logs.stderr
            fatal_patterns = ["peft_helper.validate_legal", "modules_to_save", "CUDA out of memory"]
            for pattern in fatal_patterns:
                if pattern in output:
                    logger.info(f"vLLM fatal error detected in logs: {pattern}")
                    return True
    except Exception:
        pass
    return False


def _wait_for_vllm(endpoint: str, timeout_s: int = VLLM_HEALTH_TIMEOUT_S) -> bool:
    """Poll vLLM health endpoint until ready or timeout.

    Returns True if vLLM is ready, False on timeout or crash-loop detection.
    """
    deadline = time.time() + timeout_s
    attempt = 0
    while time.time() < deadline:
        if _vllm_health_check(endpoint):
            logger.info(f"vLLM is healthy at {endpoint} (attempt {attempt + 1})")
            return True
        attempt += 1
        logger.debug(f"Waiting for vLLM at {endpoint} (attempt {attempt}) ...")

        # Check for crash-loop every 12 attempts (~60s)
        if attempt % 12 == 0 and _is_vllm_crash_looping():
            logger.error(
                "vLLM container is crash-looping (restarting repeatedly). "
                "Likely LoRA adapter incompatibility. Aborting health wait."
            )
            return False

        time.sleep(VLLM_HEALTH_POLL_S)

    cname = _get_vllm_container_name()
    logger.error(
        f"vLLM did not become healthy within {timeout_s}s. "
        f"Check: docker logs {cname} | tail -50"
    )
    return False


def _get_vllm_models(endpoint: str) -> list[str]:
    """GET /v1/models and return list of model IDs. Empty list on error."""
    try:
        import urllib.request
        models_url = endpoint.rstrip("/") + "/models"
        req = urllib.request.urlopen(models_url, timeout=10)
        data = json.loads(req.read().decode())
        models = [m.get("id", "") for m in data.get("data", [])]
        logger.info(f"vLLM /v1/models response: {models}")
        return models
    except Exception as e:
        logger.warning(f"Could not query /v1/models: {e}")
        return []


def _start_vllm_with_lora(experiment: str) -> subprocess.Popen:
    """Start vLLM with LoRA adapter for the given experiment.

    Returns the process handle. Caller is responsible for stopping it via _stop_vllm.

    Uses DGX Toolbox start-vllm.sh which handles container setup, mounts, and GPU config.
    """
    from scripts.dgx_toolbox import get_toolbox

    dgx = get_toolbox()
    adapter_path = f"/workspace/wp-finetune/adapters/{experiment}"
    model_path = "/workspace/wp-finetune/models/Qwen3-30B-A3B"

    vllm_script = dgx.resolve("vllm")
    extra_args = [
        "--enable-lora",
        f"--lora-modules=qwen3-wp={adapter_path}",
        "--max-lora-rank=64",
        "--max-model-len=4096",
        "--gpu-memory-utilization=0.92",
    ]

    cmd = [str(vllm_script), model_path] + extra_args
    logger.info(f"Starting vLLM via DGX Toolbox for experiment {experiment}")
    logger.debug(f"vLLM cmd: {' '.join(cmd)}")

    # Set EXTRA_MOUNTS so start-vllm.sh mounts the project directory
    env = os.environ.copy()
    env["EXTRA_MOUNTS"] = f"{PROJECT_ROOT}:/workspace/wp-finetune"

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    return proc


def _stop_vllm(proc: Optional[subprocess.Popen]) -> None:
    """Stop a running vLLM process and clean up its container."""
    if proc is None:
        return
    if proc.poll() is None:
        logger.info(f"Stopping vLLM process (pid={proc.pid}) ...")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM did not stop gracefully; killing ...")
            proc.kill()
            proc.wait()
    # Also clean up the docker container in case it's still running
    cname = _get_vllm_container_name()
    subprocess.run(["docker", "rm", "-f", cname], capture_output=True, timeout=15)
    logger.info("vLLM process stopped")


def _fallback_merge_and_serve(experiment: str) -> Optional[subprocess.Popen]:
    """Fallback: serve pre-merged model without LoRA.

    Pre-merge step (run at pipeline start) should have already merged all adapters.
    If merged model doesn't exist, attempts HOST merge as last resort.

    Returns vLLM process handle for merged model, or None on failure.
    """
    logger.info(f"LoRA fallback: serving merged model for experiment {experiment} ...")
    merged_model_path = PROJECT_ROOT / "models" / f"merged-{experiment}"

    if not merged_model_path.exists():
        # Pre-merge should have handled this, but attempt HOST merge as last resort
        logger.warning(f"Merged model not found at {merged_model_path} — attempting HOST merge ...")
        merge_script = PROJECT_ROOT / "scripts" / "merge_adapter.py"
        adapter_path = ADAPTERS_DIR / experiment
        result = subprocess.run(
            [sys.executable, str(merge_script),
             "--adapter-dir", str(adapter_path),
             "--output-dir", str(merged_model_path)],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            timeout=1200,
        )
        if result.returncode != 0:
            logger.error(f"HOST merge failed for experiment {experiment}: {result.stderr[:500]}")
            return None
        logger.info(f"HOST merge succeeded for experiment {experiment}")

    # Serve merged model as full model (no --lora-modules)
    container_merged_path = f"/workspace/wp-finetune/models/merged-{experiment}"
    extra_args = [
        "--max-model-len=4096",
        "--gpu-memory-utilization=0.92",
    ]

    from scripts.dgx_toolbox import get_toolbox
    dgx = get_toolbox()
    vllm_script = dgx.resolve("vllm")

    cmd = [str(vllm_script), container_merged_path] + extra_args
    logger.info(f"Serving merged model for experiment {experiment} via DGX Toolbox")

    env = os.environ.copy()
    env["EXTRA_MOUNTS"] = f"{PROJECT_ROOT}:/workspace/wp-finetune"

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    return proc


# ---------------------------------------------------------------------------
# Step 2a: Eval for a single ratio
# ---------------------------------------------------------------------------


def run_eval_for_experiment(
    experiment: str,
    dataset_path: str = "data/final_dataset/openai_test.jsonl",
    force: bool = False,
    limit: int = None,
) -> bool:
    """Start vLLM, run eval_gen + eval_judge + eval_gate, stop vLLM.

    Returns True if eval completed successfully (or was skipped due to marker).
    Returns False if eval failed -- pipeline continues with next experiment.
    """
    marker = COMPLETION_MARKERS["eval_experiment"].format(experiment=experiment)

    if not force and step_complete(marker):
        logger.info(f"Skipping eval for experiment {experiment} (already complete, marker: {marker})")
        return True

    logger.info("=" * 60)
    logger.info(f"STEP 2: Eval for experiment {experiment}")
    logger.info("=" * 60)

    from eval import eval_gen, eval_judge, eval_gate

    out_dir = EVAL_TRIAGE_DIR / experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoint = _get_vllm_endpoint()
    vllm_proc: Optional[subprocess.Popen] = None
    eval_succeeded = False

    try:
        # Start vLLM with LoRA adapter
        try:
            vllm_proc = _start_vllm_with_lora(experiment)
        except Exception as e:
            logger.error(f"Failed to start vLLM for experiment {experiment}: {e}")
            return False

        # Wait for vLLM health
        if not _wait_for_vllm(endpoint, timeout_s=VLLM_HEALTH_TIMEOUT_S):
            # Check if it's a LoRA loading error and attempt fallback
            logger.warning(
                f"vLLM failed to start for experiment {experiment}. "
                f"Attempting merge-and-serve fallback ..."
            )
            _stop_vllm(vllm_proc)
            vllm_proc = _fallback_merge_and_serve(experiment)
            if vllm_proc is None:
                logger.error(f"Fallback also failed for experiment {experiment}. Skipping eval.")
                return False
            if not _wait_for_vllm(endpoint, timeout_s=VLLM_HEALTH_TIMEOUT_S):
                raise RuntimeError(
                    f"vLLM (merged fallback) failed to become healthy for experiment {experiment} "
                    f"within {VLLM_HEALTH_TIMEOUT_S}s. "
                    f"Diagnostics: docker logs vllm | tail -100"
                )

        # Verify model name (Pitfall 3 from RESEARCH.md)
        available_models = _get_vllm_models(endpoint)
        if available_models and "qwen3-wp" not in " ".join(available_models):
            logger.warning(
                f"Expected model 'qwen3-wp' not found in vLLM models: {available_models}. "
                f"Eval scripts use model name 'openai/qwen3-wp' -- check LoRA module naming."
            )
        else:
            logger.info(f"Model name verified. Available: {available_models}")

        # Resolve model name for eval scripts (auto-detect from vLLM)
        resolved_model = available_models[0] if available_models else None

        abs_dataset = str(PROJECT_ROOT / dataset_path)

        # Run eval_gen with retry
        gen_output_path = str(out_dir / "eval_gen_results.json")
        gen_success = False
        for attempt in range(2):
            try:
                logger.info(f"Running eval_gen (attempt {attempt + 1}) ...")
                eval_gen.run_eval(
                    dataset_path=abs_dataset,
                    limit=limit,
                    output_path=gen_output_path,
                    model=resolved_model,
                )
                gen_success = True
                break
            except Exception as e:
                logger.error(f"eval_gen failed (attempt {attempt + 1}): {e}")
                if attempt == 0:
                    logger.info(f"Retrying after {EVAL_RETRY_DELAY_S}s ...")
                    time.sleep(EVAL_RETRY_DELAY_S)

        if not gen_success:
            logger.error(f"eval_gen failed after 2 attempts for experiment {experiment}. Marking as eval_failed.")
            # Write a failure marker for diagnosis
            (out_dir / ".eval_gen_failed").write_text(f"failed: {datetime.now().isoformat()}\n")
            return False

        # Run eval_judge with retry
        judge_output_path = str(out_dir / "eval_judge_results.json")
        judge_success = False
        for attempt in range(2):
            try:
                logger.info(f"Running eval_judge (attempt {attempt + 1}) ...")
                eval_judge.run_eval(
                    dataset_path=abs_dataset,
                    limit=limit,
                    output_path=judge_output_path,
                    model=resolved_model,
                )
                judge_success = True
                break
            except Exception as e:
                logger.error(f"eval_judge failed (attempt {attempt + 1}): {e}")
                if attempt == 0:
                    logger.info(f"Retrying after {EVAL_RETRY_DELAY_S}s ...")
                    time.sleep(EVAL_RETRY_DELAY_S)

        if not judge_success:
            logger.error(f"eval_judge failed after 2 attempts for experiment {experiment}. Marking as eval_failed.")
            (out_dir / ".eval_judge_failed").write_text(f"failed: {datetime.now().isoformat()}\n")
            return False

        # Run eval_gate (informational -- does not block pipeline)
        try:
            logger.info("Running eval_gate ...")
            passed, gate_rows = eval_gate.run_gate(
                results_dir=str(out_dir),
            )
            gate_status = "PASS" if passed else "FAIL"
            logger.info(f"Eval gate for experiment {experiment}: {gate_status}")
            # Write gate summary
            gate_summary = {"experiment": experiment, "passed": passed, "gate_rows": gate_rows}
            (out_dir / "eval_gate_results.json").write_text(
                json.dumps(gate_summary, indent=2)
            )
        except Exception as e:
            logger.warning(f"eval_gate error (non-fatal): {e}")

        eval_succeeded = True
        mark_complete(marker)

    finally:
        _stop_vllm(vllm_proc)

    return eval_succeeded


# ---------------------------------------------------------------------------
# Step 2b: wp-bench for a single ratio (called while vLLM is running)
# ---------------------------------------------------------------------------


def run_wpbench_for_experiment(
    experiment: str,
    force: bool = False,
) -> bool:
    """Run wp-bench against the live vLLM endpoint for the given experiment.

    IMPORTANT: This function assumes vLLM is already running.
    It must be called while vLLM is up (before _stop_vllm).

    Returns True if wp-bench completed (or was skipped), False on fatal error.
    """
    marker = COMPLETION_MARKERS["wpbench_experiment"].format(experiment=experiment)

    if not force and step_complete(marker):
        logger.info(f"Skipping wp-bench for experiment {experiment} (already complete, marker: {marker})")
        return True

    if not WP_BENCH_DIR.exists():
        logger.warning(f"wp-bench not found at {WP_BENCH_DIR}. Skipping wp-bench for experiment {experiment}.")
        return False

    logger.info(f"Running wp-bench for experiment {experiment} ...")

    out_dir = EVAL_TRIAGE_DIR / experiment
    wp_bench_output = out_dir / "wp_bench_results.json"

    # Configure wp-bench output path
    wp_bench_config = PROJECT_ROOT / "config" / "wp-bench.yaml"
    if not wp_bench_config.exists():
        logger.warning(f"wp-bench config not found at {wp_bench_config}. Skipping wp-bench.")
        return False

    try:
        import yaml as _yaml
        with open(wp_bench_config) as f:
            config = _yaml.safe_load(f) or {}

        # Update output path
        config["output_path"] = str(wp_bench_output)

        # Write temporary config for this experiment
        tmp_config = out_dir / "wp_bench_config_tmp.yaml"
        with open(tmp_config, "w") as f:
            _yaml.dump(config, f)

        # Run wp-bench via the installed CLI entry point.
        # wp-bench is installed as `pip install -e wp-bench/python` so the
        # `wp-bench` console script is on PATH.  The Python package has no
        # `wp_bench.run` module — the entry point is `wp_bench.cli:app`.
        result = subprocess.run(
            ["wp-bench", "run", "--config", str(tmp_config)],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
            cwd=str(WP_BENCH_DIR),
        )

        if result.returncode != 0:
            error_detail = result.stderr[:500] if result.stderr else result.stdout[:500]
            logger.warning(
                f"wp-bench returned non-zero for experiment {experiment} (exit={result.returncode}): "
                f"{error_detail}"
            )
            logger.warning(f"Continuing without wp-bench for experiment {experiment}.")
            wp_bench_output.write_text(json.dumps({
                "wpbench_score": None,
                "error": f"exit code {result.returncode}",
                "detail": error_detail,
            }))
            return False

        logger.info(f"wp-bench completed for experiment {experiment}: {wp_bench_output}")
        mark_complete(marker)
        return True

    except subprocess.TimeoutExpired:
        logger.warning(f"wp-bench timed out for experiment {experiment}. Skipping.")
        return False
    except Exception as e:
        logger.warning(f"wp-bench error for experiment {experiment} (non-fatal): {e}")
        return False


# ---------------------------------------------------------------------------
# Combined eval + wpbench loop
# ---------------------------------------------------------------------------


def run_eval_and_wpbench_for_experiment(
    experiment: str,
    dataset_path: str = "data/final_dataset/openai_test.jsonl",
    skip_wpbench: bool = False,
    force: bool = False,
    health_timeout: int = VLLM_HEALTH_TIMEOUT_S,
    limit: int = None,
) -> bool:
    """Run eval + wp-bench for an experiment, keeping vLLM alive between them.

    This is the correct integration: vLLM starts, eval runs, wp-bench runs
    against the live endpoint, then vLLM stops.
    """
    eval_marker = COMPLETION_MARKERS["eval_experiment"].format(experiment=experiment)
    wpbench_marker = COMPLETION_MARKERS["wpbench_experiment"].format(experiment=experiment)

    eval_already_done = not force and step_complete(eval_marker)
    wpbench_already_done = skip_wpbench or (not force and step_complete(wpbench_marker))

    if eval_already_done and wpbench_already_done:
        logger.info(f"Skipping experiment {experiment} (both eval and wp-bench already complete)")
        return True

    from eval import eval_gen, eval_judge, eval_gate

    out_dir = EVAL_TRIAGE_DIR / experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoint = _get_vllm_endpoint()
    vllm_proc: Optional[subprocess.Popen] = None
    eval_succeeded = False

    try:
        # Start vLLM with LoRA adapter (unless eval is already done and we only need wp-bench)
        if not eval_already_done:
            try:
                vllm_proc = _start_vllm_with_lora(experiment)
            except Exception as e:
                logger.error(f"Failed to start vLLM for experiment {experiment}: {e}")
                return False

            # Wait for vLLM
            if not _wait_for_vllm(endpoint, timeout_s=health_timeout):
                logger.warning(
                    f"vLLM failed to start for experiment {experiment}. "
                    f"Attempting merge-and-serve fallback ..."
                )
                _stop_vllm(vllm_proc)
                vllm_proc = _fallback_merge_and_serve(experiment)
                if vllm_proc is None:
                    logger.error(f"Fallback also failed for experiment {experiment}. Skipping.")
                    return False
                if not _wait_for_vllm(endpoint, timeout_s=health_timeout):
                    raise RuntimeError(
                        f"vLLM (merged fallback) failed for experiment {experiment} within "
                        f"{health_timeout}s. Check: docker logs vllm | tail -100"
                    )

            # Verify model name (Pitfall 3)
            available_models = _get_vllm_models(endpoint)
            if available_models and "qwen3-wp" not in " ".join(available_models):
                logger.warning(
                    f"Expected 'qwen3-wp' not in available models: {available_models}. "
                    f"Continuing anyway -- eval scripts use 'openai/qwen3-wp'."
                )

            # Resolve model name for eval scripts (auto-detect from vLLM)
            resolved_model = available_models[0] if available_models else None

            abs_dataset = str(PROJECT_ROOT / dataset_path)

            # eval_gen with retry
            gen_output_path = str(out_dir / "eval_gen_results.json")
            gen_success = False
            for attempt in range(2):
                try:
                    logger.info(f"Running eval_gen for experiment {experiment} (attempt {attempt + 1}) ...")
                    eval_gen.run_eval(
                        dataset_path=abs_dataset,
                        limit=limit,
                        output_path=gen_output_path,
                        model=resolved_model,
                    )
                    gen_success = True
                    break
                except Exception as e:
                    logger.error(f"eval_gen failed (attempt {attempt + 1}): {e}")
                    if attempt == 0:
                        logger.info(f"Retrying after {EVAL_RETRY_DELAY_S}s ...")
                        time.sleep(EVAL_RETRY_DELAY_S)

            if not gen_success:
                logger.error(f"eval_gen failed after 2 attempts for experiment {experiment}.")
                (out_dir / ".eval_gen_failed").write_text(f"failed: {datetime.now().isoformat()}\n")
                return False

            # eval_judge with retry
            judge_output_path = str(out_dir / "eval_judge_results.json")
            judge_success = False
            for attempt in range(2):
                try:
                    logger.info(f"Running eval_judge for experiment {experiment} (attempt {attempt + 1}) ...")
                    eval_judge.run_eval(
                        dataset_path=abs_dataset,
                        limit=limit,
                        output_path=judge_output_path,
                        model=resolved_model,
                    )
                    judge_success = True
                    break
                except Exception as e:
                    logger.error(f"eval_judge failed (attempt {attempt + 1}): {e}")
                    if attempt == 0:
                        logger.info(f"Retrying after {EVAL_RETRY_DELAY_S}s ...")
                        time.sleep(EVAL_RETRY_DELAY_S)

            if not judge_success:
                logger.error(f"eval_judge failed after 2 attempts for experiment {experiment}.")
                (out_dir / ".eval_judge_failed").write_text(f"failed: {datetime.now().isoformat()}\n")
                return False

            # eval_gate (informational)
            try:
                passed, gate_rows = eval_gate.run_gate(results_dir=str(out_dir))
                gate_status = "PASS" if passed else "FAIL"
                logger.info(f"Eval gate for experiment {experiment}: {gate_status}")
                gate_summary = {"experiment": experiment, "passed": passed, "gate_rows": gate_rows}
                (out_dir / "eval_gate_results.json").write_text(
                    json.dumps(gate_summary, indent=2)
                )
            except Exception as e:
                logger.warning(f"eval_gate error (non-fatal): {e}")

            eval_succeeded = True
            mark_complete(eval_marker)

        else:
            eval_succeeded = True
            logger.info(f"Eval already done for experiment {experiment}, starting vLLM for wp-bench only ...")
            # Still need vLLM for wp-bench.
            # Use the pre-merged model (not LoRA serving) so wp-bench runs
            # against the same weights that will ship — LoRA serving requires
            # --enable-lora which is unnecessary here and has caused instability.
            if not skip_wpbench and not wpbench_already_done:
                try:
                    vllm_proc = _fallback_merge_and_serve(experiment)
                    if vllm_proc is None or not _wait_for_vllm(endpoint, timeout_s=health_timeout):
                        logger.warning(f"vLLM (merged) failed to start for wp-bench on experiment {experiment}. Skipping wp-bench.")
                        return eval_succeeded
                except Exception as e:
                    logger.warning(f"Could not start vLLM for wp-bench: {e}")
                    return eval_succeeded

        # Run wp-bench while vLLM is still running
        if not skip_wpbench and not wpbench_already_done and vllm_proc is not None:
            run_wpbench_for_experiment(experiment=experiment, force=force)

    finally:
        _stop_vllm(vllm_proc)

    return eval_succeeded


# ---------------------------------------------------------------------------
# Step 3: Triage decision
# ---------------------------------------------------------------------------


def run_triage(force: bool = False) -> bool:
    """Load eval results and write triage_decision.md.

    The triage marker is re-evaluated if any eval marker is newer.
    Returns True if triage completed (or was skipped due to marker).
    """
    triage_marker = COMPLETION_MARKERS["triage"]

    if not force and step_complete(triage_marker):
        # Check if any eval marker is newer than triage marker
        triage_mtime = Path(triage_marker).stat().st_mtime
        for experiment in discover_experiments():
            eval_marker = COMPLETION_MARKERS["eval_experiment"].format(experiment=experiment)
            if Path(eval_marker).exists():
                eval_mtime = Path(eval_marker).stat().st_mtime
                if eval_mtime > triage_mtime:
                    logger.info(
                        f"Eval marker for experiment {experiment} is newer than triage marker. "
                        f"Re-running triage."
                    )
                    clear_marker(triage_marker)
                    break
        else:
            logger.info(f"Skipping triage (already complete, marker: {triage_marker})")
            return True

    logger.info("=" * 60)
    logger.info("STEP 3: Triage decision")
    logger.info("=" * 60)

    from scripts.triage_ratios import load_eval_results, triage_ratios, write_triage_decision

    eval_results = load_eval_results(str(EVAL_TRIAGE_DIR))
    if not eval_results:
        logger.error(
            f"No eval results found in {EVAL_TRIAGE_DIR}. "
            f"Triage cannot proceed. Run eval first."
        )
        return False

    logger.info(f"Loaded eval results for ratios: {list(eval_results.keys())}")

    # Load profiling summary if available
    profiling_summary = None
    profiling_jsonl = PROFILING_DIR / "base_model_eeff.jsonl"
    if profiling_jsonl.exists():
        try:
            import math
            import numpy as np

            ratio_eeffs: dict[str, list] = {}
            with open(profiling_jsonl) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    r = rec.get("ratio", "")
                    if r not in ratio_eeffs:
                        ratio_eeffs[r] = []
                    v = rec.get("eeff_total")
                    if v is not None:
                        ratio_eeffs[r].append(float(v))

            profiling_summary = {
                ratio: {
                    "mean_eeff_total": float(np.nanmean(vals)) if vals else None,
                    "max_eeff_total": float(np.nanmax(vals)) if vals else None,
                }
                for ratio, vals in ratio_eeffs.items()
            }
            logger.info(f"Loaded profiling summary for ratios: {list(profiling_summary.keys())}")
        except Exception as e:
            logger.warning(f"Could not load profiling summary: {e}")

    triage_result = triage_ratios(eval_results, profiling_summary=profiling_summary)

    triage_output = OUTPUT_DIR / "triage_decision.md"
    write_triage_decision(triage_result, profiling_summary=profiling_summary, out_path=str(triage_output))

    # Print summary
    print("\n" + "=" * 60)
    print(f"TRIAGE RESULT: STATUS={triage_result.status}")
    print(f"Survivors:  {', '.join(triage_result.survivors) if triage_result.survivors else 'NONE'}")
    print(f"Best ratio: {triage_result.best_ratio or 'N/A'}")
    if triage_result.eliminated:
        print("Eliminated:")
        for e in triage_result.eliminated:
            print(f"  - {e['ratio']}: {e['reason']}")
    print(f"wp-bench available: {triage_result.wpbench_available}")
    print(f"Triage decision written to: {triage_output}")
    print("=" * 60 + "\n")

    if triage_result.status == "NO_SURVIVORS":
        print(
            "\n*** WARNING: NO_SURVIVORS -- all ratios failed hard gates. ***\n"
            "Do NOT panic. Plan 03 (human review) handles this contingency.\n"
            "Record this result and consult the human for next steps.\n"
            "Recommendation options (from triage_decision.md):\n"
            "  1. Re-examine training data quality\n"
            "  2. Investigate specific failure dimensions\n"
            "  3. Lower gate thresholds if domain warrants\n"
        )

    mark_complete(triage_marker)
    return True


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------


def run_full_triage(
    experiments: list[str] = None,
    skip_profiling: bool = False,
    skip_wpbench: bool = False,
    force: bool = False,
    model_path: str = "models/Qwen3-30B-A3B",
    tokenizer_path: str = "adapters/tokenizer",
    dataset_path: str = "data/final_dataset/openai_test.jsonl",
    health_timeout: int = VLLM_HEALTH_TIMEOUT_S,
    limit: int = None,
) -> None:
    """Run the full Phase 4 pipeline: profiling -> eval -> triage.

    Args:
        experiments: Adapter experiments to evaluate (default: auto-discovered from adapters/).
        skip_profiling: Skip Step 1 (profiling). Still checks for existing results.
        skip_wpbench: Skip wp-bench entirely.
        force: Bypass all completion markers and re-run everything.
        model_path: Path to base model (relative to project root).
        tokenizer_path: Path to extended tokenizer (relative to project root).
        dataset_path: Path to test dataset (relative to project root).
        health_timeout: Seconds to wait for vLLM health check.
        limit: Max examples for eval_gen/eval_judge (None = all). Does not affect wp-bench.
    """
    if experiments is None:
        experiments = discover_adapters()
        if not experiments:
            logger.error(
                f"No adapters found in {ADAPTERS_DIR}. "
                f"Each adapter directory must contain adapter_config.json."
            )
            sys.exit(1)

    logger.info("Phase 4 Eval Triage pipeline starting ...")
    logger.info(f"Experiments: {experiments}")
    logger.info(f"Skip profiling: {skip_profiling}")
    logger.info(f"Skip wp-bench: {skip_wpbench}")
    logger.info(f"Force re-run: {force}")

    start_time = time.time()

    # Force: clean stale result files so monitors and idempotency checks start fresh
    if force:
        _clean_stale_results(experiments)

    # Step 0: Setup
    setup_output_dirs(experiments)
    wpbench_available = False
    if not skip_wpbench:
        wpbench_available = setup_wpbench()
        if not wpbench_available:
            logger.warning("wp-bench setup failed -- will skip wp-bench for all experiments")
            skip_wpbench = True

    # Step 1: Base-model profiling
    if not skip_profiling:
        run_profiling(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            force=force,
        )
    else:
        logger.info("Skipping profiling (--skip-profiling flag set)")

    # Step 1.5: Pre-merge all adapters (LoRA serving always fails for Qwen3-30B-A3B)
    merge_results = pre_merge_adapters(experiments)
    merge_failures = [e for e, ok in merge_results.items() if not ok]
    if merge_failures:
        logger.warning(f"Pre-merge failures: {merge_failures} — these experiments will be skipped")
        # Remove failed experiments from eval list
        experiments = [e for e in experiments if merge_results.get(e, False)]
        if not experiments:
            logger.error("All adapter merges failed. Cannot proceed with eval.")
            sys.exit(1)

    # Step 2: Sequential adapter eval + wp-bench
    eval_failures = []
    for experiment in experiments:
        logger.info(f"--- Evaluating experiment {experiment} ---")
        success = run_eval_and_wpbench_for_experiment(
            experiment=experiment,
            dataset_path=dataset_path,
            skip_wpbench=skip_wpbench,
            force=force,
            health_timeout=health_timeout,
            limit=limit,
        )
        if not success:
            eval_failures.append(experiment)
            logger.warning(f"Eval failed for experiment {experiment} -- continuing with remaining experiments")

    if eval_failures:
        logger.warning(f"Eval failures: {eval_failures}. Triage will proceed with available results.")

    # Step 3: Triage decision
    triage_ok = run_triage(force=force)
    if not triage_ok:
        logger.error("Triage failed. Check logs and re-run after fixing eval results.")
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info(f"Phase 4 pipeline complete in {elapsed:.0f}s ({elapsed / 60:.1f} min)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 Eval Triage: profiling + sequential adapter eval + triage decision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_eval_triage.py                                          # Full pipeline (auto-discover adapters)
    python scripts/run_eval_triage.py --skip-wpbench                          # Skip wp-bench
    python scripts/run_eval_triage.py --skip-profiling                        # Skip profiling
    python scripts/run_eval_triage.py --experiments qwen3-30b-wp-30_70        # Eval only one adapter
    python scripts/run_eval_triage.py --force                                 # Force full re-run
""",
    )
    parser.add_argument(
        "--skip-profiling",
        action="store_true",
        help="Skip base-model profiling (Step 1). Use existing results if present.",
    )
    parser.add_argument(
        "--skip-wpbench",
        action="store_true",
        help="Skip wp-bench for all experiments. Triage will use static eval gates only.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated list of adapter experiment names to evaluate "
             "(default: auto-discover from adapters/ directory). "
             "Each name must match a subdirectory of adapters/ containing adapter_config.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass all completion markers and re-run all steps from scratch.",
    )
    parser.add_argument(
        "--model-path",
        default="models/Qwen3-30B-A3B",
        help="Path to base model directory (relative to project root).",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="adapters/tokenizer",
        help="Path to extended tokenizer (relative to project root).",
    )
    parser.add_argument(
        "--dataset",
        default="data/final_dataset/openai_test.jsonl",
        help="Path to test dataset (relative to project root).",
    )
    parser.add_argument(
        "--health-timeout",
        type=int,
        default=VLLM_HEALTH_TIMEOUT_S,
        help=f"Seconds to wait for vLLM health check (default: {VLLM_HEALTH_TIMEOUT_S}). "
             "Increase for larger models, decrease for smaller ones.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max examples for eval_gen/eval_judge per experiment (default: all). "
             "Use 500 for a representative sample (~2.8h per experiment instead of ~58h). "
             "Does not affect wp-bench (always runs full suite).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    experiments = None
    if args.experiments is not None:
        experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
        if not experiments:
            parser.error("--experiments cannot be empty")

    run_full_triage(
        experiments=experiments,
        skip_profiling=args.skip_profiling,
        skip_wpbench=args.skip_wpbench,
        force=args.force,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        dataset_path=args.dataset,
        health_timeout=args.health_timeout,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
