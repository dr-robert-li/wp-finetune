"""Phase 4 Eval Triage Orchestrator.

Full pipeline for base-model profiling + sequential adapter eval + triage decision.

Steps:
    0. Setup (output dirs, wp-bench clone)
    1. Base-model E_eff profiling (gradient-free, all 5 ratios)
    2. Sequential adapter eval (vLLM LoRA serving per ratio + full eval suite + wp-bench)
    3. Triage decision (load_eval_results -> triage_ratios -> write_triage_decision)

Idempotency: every major step writes a completion marker (.complete file).
On re-run, steps with existing markers are skipped unless --force is passed.

Usage:
    python scripts/run_eval_triage.py
    python scripts/run_eval_triage.py --skip-wpbench
    python scripts/run_eval_triage.py --ratios 30_70,50_50
    python scripts/run_eval_triage.py --force
    python scripts/run_eval_triage.py --skip-profiling
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
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

ALL_EVAL_RATIOS = ["30_70", "40_60", "50_50"]
ALL_PROFILE_RATIOS = ["30_70", "40_60", "50_50", "60_40", "70_30"]
DATASET_BASE = "data/final_dataset"
OUTPUT_DIR = PROJECT_ROOT / "output"
PROFILING_DIR = OUTPUT_DIR / "profiling"
EVAL_TRIAGE_DIR = OUTPUT_DIR / "eval_triage"
WP_BENCH_DIR = PROJECT_ROOT / "wp-bench"

VLLM_HEALTH_TIMEOUT_S = 300   # 5 minutes
VLLM_HEALTH_POLL_S = 5
EVAL_RETRY_DELAY_S = 30

# Completion markers
COMPLETION_MARKERS = {
    "profiling": str(PROFILING_DIR / ".complete"),
    "eval_ratio": str(EVAL_TRIAGE_DIR / "ratio_{ratio}" / ".complete"),
    "wpbench_ratio": str(EVAL_TRIAGE_DIR / "ratio_{ratio}" / ".wpbench_complete"),
    "triage": str(OUTPUT_DIR / ".triage_complete"),
}


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


# ---------------------------------------------------------------------------
# Step 0: Setup
# ---------------------------------------------------------------------------


def setup_output_dirs(eval_ratios: list[str]) -> None:
    """Create all output directories."""
    PROFILING_DIR.mkdir(parents=True, exist_ok=True)
    for ratio in eval_ratios:
        (EVAL_TRIAGE_DIR / f"ratio_{ratio}").mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directories ready: {OUTPUT_DIR}")


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
        RATIO_ORDER,
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

    # Build ratio data paths
    ratio_data_paths = {}
    for ratio in RATIO_ORDER:
        data_path = PROJECT_ROOT / DATASET_BASE / f"ratio_{ratio}" / "openai_train.jsonl"
        if data_path.exists():
            ratio_data_paths[ratio] = str(data_path)
        else:
            logger.warning(f"Ratio {ratio} data not found at {data_path} -- skipping")

    if not ratio_data_paths:
        raise RuntimeError("No ratio data found. Cannot profile.")

    logger.info(f"Loading extended tokenizer from {abs_tokenizer_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(abs_tokenizer_path))

    logger.info(f"Loading model from {abs_model_path} (bfloat16) ...")
    # Log GPU memory before loading
    if torch.cuda.is_available():
        free_before = torch.cuda.mem_get_info()[0] / (1024**3)
        logger.info(f"GPU free memory before model load: {free_before:.1f} GB")

    model = AutoModelForCausalLM.from_pretrained(
        str(abs_model_path),
        torch_dtype=torch.bfloat16,
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
        for ratio in RATIO_ORDER:
            if ratio in all_ratio_eeffs:
                vals = [
                    v for v in all_ratio_eeffs[ratio]["eeff_total"]
                    if not (isinstance(v, float) and math.isnan(v))
                ]
                means_total.append(float(np.nanmean(vals)) if vals else float("nan"))
            else:
                means_total.append(float("nan"))

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


def _wait_for_vllm(endpoint: str, timeout_s: int = VLLM_HEALTH_TIMEOUT_S) -> bool:
    """Poll vLLM health endpoint until ready or timeout.

    Returns True if vLLM is ready, False on timeout.
    """
    deadline = time.time() + timeout_s
    attempt = 0
    while time.time() < deadline:
        if _vllm_health_check(endpoint):
            logger.info(f"vLLM is healthy at {endpoint} (attempt {attempt + 1})")
            return True
        attempt += 1
        logger.debug(f"Waiting for vLLM at {endpoint} (attempt {attempt}) ...")
        time.sleep(VLLM_HEALTH_POLL_S)

    logger.error(
        f"vLLM did not become healthy within {timeout_s}s. "
        f"Check: docker logs vllm | tail -50"
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


def _start_vllm_with_lora(ratio: str) -> subprocess.Popen:
    """Start vLLM with LoRA adapter for the given ratio.

    Returns the process handle. Caller is responsible for stopping it via _stop_vllm.

    Tries DGX Toolbox first; falls back to direct docker run if unavailable.
    """
    adapter_path = f"/workspace/wp-finetune/adapters/qwen3-30b-wp-{ratio}"
    model_path = "/workspace/wp-finetune/models/Qwen3-30B-A3B"

    vllm_extra_args = [
        "--enable-lora",
        f"--lora-modules=qwen3-wp={adapter_path}",
        "--max-lora-rank=64",
        "--max-model-len=4096",
        "--gpu-memory-utilization=0.92",
    ]

    # Try DGX Toolbox first
    try:
        from scripts.dgx_toolbox import get_toolbox
        dgx = get_toolbox()
        vllm_script = dgx.resolve("vllm")
        cmd = [str(vllm_script), model_path] + vllm_extra_args
        logger.info(f"Starting vLLM via DGX Toolbox for ratio {ratio}: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return proc
    except Exception as e:
        logger.info(f"DGX Toolbox unavailable ({e}), falling back to direct docker run")

    # Fallback: docker run directly
    # Clean up any existing vllm container
    subprocess.run(["docker", "rm", "-f", "vllm"], capture_output=True, timeout=10)

    home = Path.home()
    cmd = [
        "docker", "run", "--rm", "--name", "vllm",
        "--gpus", "all", "--ipc=host",
        "--user", f"{subprocess.check_output(['id', '-u']).decode().strip()}:{subprocess.check_output(['id', '-g']).decode().strip()}",
        "-p", "0.0.0.0:8020:8000",
        "-v", f"{PROJECT_ROOT}:/workspace/wp-finetune",
        "-v", f"{home}/.cache/huggingface:/root/.cache/huggingface",
        "vllm/vllm-openai:latest",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", "8000",
    ] + vllm_extra_args

    logger.info(f"Starting vLLM via docker run for ratio {ratio}")
    logger.debug(f"Docker cmd: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
    subprocess.run(["docker", "rm", "-f", "vllm"], capture_output=True, timeout=15)
    logger.info("vLLM process stopped")


def _fallback_merge_and_serve(ratio: str) -> Optional[subprocess.Popen]:
    """Fallback: merge adapter and serve merged model without LoRA.

    Used when vLLM fails to load adapter (e.g., modules_to_save tensors).
    Returns vLLM process handle for merged model, or None on failure.
    """
    logger.info(f"LoRA fallback: merging adapter for ratio {ratio} ...")
    merged_model_path = PROJECT_ROOT / "models" / f"merged-{ratio}"

    if not merged_model_path.exists():
        # Run merge_adapter.py
        merge_script = PROJECT_ROOT / "scripts" / "merge_adapter.py"
        if not merge_script.exists():
            logger.error("merge_adapter.py not found -- cannot fall back to merged serving")
            return None

        adapter_path = PROJECT_ROOT / "adapters" / f"qwen3-30b-wp-{ratio}"
        result = subprocess.run(
            [
                sys.executable,
                str(merge_script),
                "--adapter-path", str(adapter_path),
                "--output-path", str(merged_model_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            logger.error(f"merge_adapter.py failed: {result.stderr[:1000]}")
            return None

        # Warn about disk usage
        logger.warning(
            f"Merged checkpoint created at {merged_model_path}. "
            f"Note: merged checkpoints are ~60GB each. Check disk usage."
        )

    # Serve merged model as full model (no --lora-modules)
    container_merged_path = f"/workspace/wp-finetune/models/merged-{ratio}"
    extra_args = [
        "--max-model-len=4096",
        "--gpu-memory-utilization=0.92",
    ]

    # Try DGX Toolbox first
    try:
        from scripts.dgx_toolbox import get_toolbox
        dgx = get_toolbox()
        vllm_script = dgx.resolve("vllm")
        cmd = [str(vllm_script), container_merged_path] + extra_args
        logger.info(f"Serving merged model for ratio {ratio} via DGX Toolbox: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return proc
    except Exception as e:
        logger.info(f"DGX Toolbox unavailable ({e}), using direct docker run for merged model")

    # Fallback: docker run directly
    subprocess.run(["docker", "rm", "-f", "vllm"], capture_output=True, timeout=10)

    home = Path.home()
    cmd = [
        "docker", "run", "--rm", "--name", "vllm",
        "--gpus", "all", "--ipc=host",
        "-p", "0.0.0.0:8020:8000",
        "-v", f"{PROJECT_ROOT}:/workspace/wp-finetune",
        "-v", f"{home}/.cache/huggingface:/root/.cache/huggingface",
        "vllm/vllm-openai:latest",
        "--model", container_merged_path,
        "--host", "0.0.0.0",
        "--port", "8000",
    ] + extra_args

    logger.info(f"Serving merged model for ratio {ratio} via docker run")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc


# ---------------------------------------------------------------------------
# Step 2a: Eval for a single ratio
# ---------------------------------------------------------------------------


def run_eval_for_ratio(
    ratio: str,
    dataset_path: str = "data/final_dataset/openai_test.jsonl",
    force: bool = False,
) -> bool:
    """Start vLLM, run eval_gen + eval_judge + eval_gate, stop vLLM.

    Returns True if eval completed successfully (or was skipped due to marker).
    Returns False if eval failed -- pipeline continues with next ratio.
    """
    marker = COMPLETION_MARKERS["eval_ratio"].format(ratio=ratio)

    if not force and step_complete(marker):
        logger.info(f"Skipping eval for ratio {ratio} (already complete, marker: {marker})")
        return True

    logger.info("=" * 60)
    logger.info(f"STEP 2: Eval for ratio {ratio}")
    logger.info("=" * 60)

    from eval import eval_gen, eval_judge, eval_gate

    out_dir = EVAL_TRIAGE_DIR / f"ratio_{ratio}"
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoint = _get_vllm_endpoint()
    vllm_proc: Optional[subprocess.Popen] = None
    eval_succeeded = False

    try:
        # Start vLLM with LoRA adapter
        try:
            vllm_proc = _start_vllm_with_lora(ratio)
        except Exception as e:
            logger.error(f"Failed to start vLLM for ratio {ratio}: {e}")
            return False

        # Wait for vLLM health
        if not _wait_for_vllm(endpoint, timeout_s=VLLM_HEALTH_TIMEOUT_S):
            # Check if it's a LoRA loading error and attempt fallback
            logger.warning(
                f"vLLM failed to start for ratio {ratio}. "
                f"Attempting merge-and-serve fallback ..."
            )
            _stop_vllm(vllm_proc)
            vllm_proc = _fallback_merge_and_serve(ratio)
            if vllm_proc is None:
                logger.error(f"Fallback also failed for ratio {ratio}. Skipping eval.")
                return False
            if not _wait_for_vllm(endpoint, timeout_s=VLLM_HEALTH_TIMEOUT_S):
                raise RuntimeError(
                    f"vLLM (merged fallback) failed to become healthy for ratio {ratio} "
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
            logger.error(f"eval_gen failed after 2 attempts for ratio {ratio}. Marking as eval_failed.")
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
            logger.error(f"eval_judge failed after 2 attempts for ratio {ratio}. Marking as eval_failed.")
            (out_dir / ".eval_judge_failed").write_text(f"failed: {datetime.now().isoformat()}\n")
            return False

        # Run eval_gate (informational -- does not block pipeline)
        try:
            logger.info("Running eval_gate ...")
            passed, gate_rows = eval_gate.run_gate(
                results_dir=str(out_dir),
            )
            gate_status = "PASS" if passed else "FAIL"
            logger.info(f"Eval gate for ratio {ratio}: {gate_status}")
            # Write gate summary
            gate_summary = {"ratio": ratio, "passed": passed, "gate_rows": gate_rows}
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


def run_wpbench_for_ratio(
    ratio: str,
    force: bool = False,
) -> bool:
    """Run wp-bench against the live vLLM endpoint for the given ratio.

    IMPORTANT: This function assumes vLLM is already running.
    It must be called while vLLM is up (before _stop_vllm).

    Returns True if wp-bench completed (or was skipped), False on fatal error.
    """
    marker = COMPLETION_MARKERS["wpbench_ratio"].format(ratio=ratio)

    if not force and step_complete(marker):
        logger.info(f"Skipping wp-bench for ratio {ratio} (already complete, marker: {marker})")
        return True

    if not WP_BENCH_DIR.exists():
        logger.warning(f"wp-bench not found at {WP_BENCH_DIR}. Skipping wp-bench for ratio {ratio}.")
        return False

    logger.info(f"Running wp-bench for ratio {ratio} ...")

    out_dir = EVAL_TRIAGE_DIR / f"ratio_{ratio}"
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

        # Write temporary config for this ratio
        tmp_config = out_dir / "wp_bench_config_tmp.yaml"
        with open(tmp_config, "w") as f:
            _yaml.dump(config, f)

        # Run wp-bench
        result = subprocess.run(
            ["python", "-m", "wp_bench.run", "--config", str(tmp_config)],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
            cwd=str(WP_BENCH_DIR),
        )

        if result.returncode != 0:
            logger.warning(f"wp-bench returned non-zero for ratio {ratio}: {result.stderr[:500]}")
            logger.warning(f"Continuing without wp-bench for ratio {ratio}.")
            # Write a None score placeholder
            wp_bench_output.write_text(json.dumps({"wpbench_score": None, "error": "non-zero exit"}))
            return False

        logger.info(f"wp-bench completed for ratio {ratio}: {wp_bench_output}")
        mark_complete(marker)
        return True

    except subprocess.TimeoutExpired:
        logger.warning(f"wp-bench timed out for ratio {ratio}. Skipping.")
        return False
    except Exception as e:
        logger.warning(f"wp-bench error for ratio {ratio} (non-fatal): {e}")
        return False


# ---------------------------------------------------------------------------
# Combined eval + wpbench loop
# ---------------------------------------------------------------------------


def run_eval_and_wpbench_for_ratio(
    ratio: str,
    dataset_path: str = "data/final_dataset/openai_test.jsonl",
    skip_wpbench: bool = False,
    force: bool = False,
) -> bool:
    """Run eval + wp-bench for a ratio, keeping vLLM alive between them.

    This is the correct integration: vLLM starts, eval runs, wp-bench runs
    against the live endpoint, then vLLM stops.
    """
    eval_marker = COMPLETION_MARKERS["eval_ratio"].format(ratio=ratio)
    wpbench_marker = COMPLETION_MARKERS["wpbench_ratio"].format(ratio=ratio)

    eval_already_done = not force and step_complete(eval_marker)
    wpbench_already_done = skip_wpbench or (not force and step_complete(wpbench_marker))

    if eval_already_done and wpbench_already_done:
        logger.info(f"Skipping ratio {ratio} (both eval and wp-bench already complete)")
        return True

    from eval import eval_gen, eval_judge, eval_gate

    out_dir = EVAL_TRIAGE_DIR / f"ratio_{ratio}"
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoint = _get_vllm_endpoint()
    vllm_proc: Optional[subprocess.Popen] = None
    eval_succeeded = False

    try:
        # Start vLLM with LoRA adapter (unless eval is already done and we only need wp-bench)
        if not eval_already_done:
            try:
                vllm_proc = _start_vllm_with_lora(ratio)
            except Exception as e:
                logger.error(f"Failed to start vLLM for ratio {ratio}: {e}")
                return False

            # Wait for vLLM
            if not _wait_for_vllm(endpoint, timeout_s=VLLM_HEALTH_TIMEOUT_S):
                logger.warning(
                    f"vLLM failed to start for ratio {ratio}. "
                    f"Attempting merge-and-serve fallback ..."
                )
                _stop_vllm(vllm_proc)
                vllm_proc = _fallback_merge_and_serve(ratio)
                if vllm_proc is None:
                    logger.error(f"Fallback also failed for ratio {ratio}. Skipping.")
                    return False
                if not _wait_for_vllm(endpoint, timeout_s=VLLM_HEALTH_TIMEOUT_S):
                    raise RuntimeError(
                        f"vLLM (merged fallback) failed for ratio {ratio} within "
                        f"{VLLM_HEALTH_TIMEOUT_S}s. Check: docker logs vllm | tail -100"
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
                    logger.info(f"Running eval_gen for ratio {ratio} (attempt {attempt + 1}) ...")
                    eval_gen.run_eval(
                        dataset_path=abs_dataset,
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
                logger.error(f"eval_gen failed after 2 attempts for ratio {ratio}.")
                (out_dir / ".eval_gen_failed").write_text(f"failed: {datetime.now().isoformat()}\n")
                return False

            # eval_judge with retry
            judge_output_path = str(out_dir / "eval_judge_results.json")
            judge_success = False
            for attempt in range(2):
                try:
                    logger.info(f"Running eval_judge for ratio {ratio} (attempt {attempt + 1}) ...")
                    eval_judge.run_eval(
                        dataset_path=abs_dataset,
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
                logger.error(f"eval_judge failed after 2 attempts for ratio {ratio}.")
                (out_dir / ".eval_judge_failed").write_text(f"failed: {datetime.now().isoformat()}\n")
                return False

            # eval_gate (informational)
            try:
                passed, gate_rows = eval_gate.run_gate(results_dir=str(out_dir))
                gate_status = "PASS" if passed else "FAIL"
                logger.info(f"Eval gate for ratio {ratio}: {gate_status}")
                gate_summary = {"ratio": ratio, "passed": passed, "gate_rows": gate_rows}
                (out_dir / "eval_gate_results.json").write_text(
                    json.dumps(gate_summary, indent=2)
                )
            except Exception as e:
                logger.warning(f"eval_gate error (non-fatal): {e}")

            eval_succeeded = True
            mark_complete(eval_marker)

        else:
            eval_succeeded = True
            logger.info(f"Eval already done for ratio {ratio}, starting vLLM for wp-bench only ...")
            # Still need vLLM for wp-bench
            if not skip_wpbench and not wpbench_already_done:
                try:
                    vllm_proc = _start_vllm_with_lora(ratio)
                    if not _wait_for_vllm(endpoint, timeout_s=VLLM_HEALTH_TIMEOUT_S):
                        logger.warning(f"vLLM failed to start for wp-bench on ratio {ratio}. Skipping wp-bench.")
                        return eval_succeeded
                except Exception as e:
                    logger.warning(f"Could not start vLLM for wp-bench: {e}")
                    return eval_succeeded

        # Run wp-bench while vLLM is still running
        if not skip_wpbench and not wpbench_already_done and vllm_proc is not None:
            run_wpbench_for_ratio(ratio=ratio, force=force)

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
        for ratio in ALL_EVAL_RATIOS:
            eval_marker = COMPLETION_MARKERS["eval_ratio"].format(ratio=ratio)
            if Path(eval_marker).exists():
                eval_mtime = Path(eval_marker).stat().st_mtime
                if eval_mtime > triage_mtime:
                    logger.info(
                        f"Eval marker for ratio {ratio} is newer than triage marker. "
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
    eval_ratios: list[str] = None,
    skip_profiling: bool = False,
    skip_wpbench: bool = False,
    force: bool = False,
    model_path: str = "models/Qwen3-30B-A3B",
    tokenizer_path: str = "adapters/tokenizer",
    dataset_path: str = "data/final_dataset/openai_test.jsonl",
) -> None:
    """Run the full Phase 4 pipeline: profiling -> eval -> triage.

    Args:
        eval_ratios: Ratios to evaluate (default: ["30_70", "40_60", "50_50"]).
        skip_profiling: Skip Step 1 (profiling). Still checks for existing results.
        skip_wpbench: Skip wp-bench entirely.
        force: Bypass all completion markers and re-run everything.
        model_path: Path to base model (relative to project root).
        tokenizer_path: Path to extended tokenizer (relative to project root).
        dataset_path: Path to test dataset (relative to project root).
    """
    if eval_ratios is None:
        eval_ratios = ALL_EVAL_RATIOS

    logger.info("Phase 4 Eval Triage pipeline starting ...")
    logger.info(f"Eval ratios: {eval_ratios}")
    logger.info(f"Skip profiling: {skip_profiling}")
    logger.info(f"Skip wp-bench: {skip_wpbench}")
    logger.info(f"Force re-run: {force}")

    start_time = time.time()

    # Step 0: Setup
    setup_output_dirs(eval_ratios)
    wpbench_available = False
    if not skip_wpbench:
        wpbench_available = setup_wpbench()
        if not wpbench_available:
            logger.warning("wp-bench setup failed -- will skip wp-bench for all ratios")
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

    # Step 2: Sequential adapter eval + wp-bench
    eval_failures = []
    for ratio in eval_ratios:
        logger.info(f"--- Evaluating ratio {ratio} ---")
        success = run_eval_and_wpbench_for_ratio(
            ratio=ratio,
            dataset_path=dataset_path,
            skip_wpbench=skip_wpbench,
            force=force,
        )
        if not success:
            eval_failures.append(ratio)
            logger.warning(f"Eval failed for ratio {ratio} -- continuing with remaining ratios")

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
    python scripts/run_eval_triage.py                    # Full pipeline
    python scripts/run_eval_triage.py --skip-wpbench    # Skip wp-bench
    python scripts/run_eval_triage.py --skip-profiling  # Skip profiling
    python scripts/run_eval_triage.py --ratios 30_70    # Eval only 30_70
    python scripts/run_eval_triage.py --force           # Force full re-run
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
        help="Skip wp-bench for all ratios. Triage will use static eval gates only.",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default=",".join(ALL_EVAL_RATIOS),
        help=f"Comma-separated list of ratios to evaluate (default: {','.join(ALL_EVAL_RATIOS)})",
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
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    eval_ratios = [r.strip() for r in args.ratios.split(",") if r.strip()]
    if not eval_ratios:
        parser.error("--ratios cannot be empty")

    run_full_triage(
        eval_ratios=eval_ratios,
        skip_profiling=args.skip_profiling,
        skip_wpbench=args.skip_wpbench,
        force=args.force,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        dataset_path=args.dataset,
    )


if __name__ == "__main__":
    main()
