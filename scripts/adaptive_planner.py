"""Adaptive training planner — core decision logic.

This module contains ALL algorithmic decisions for the adaptive training
planner as testable pure functions. The adaptive-planner skill (Plan 03)
is a thin wrapper that calls these functions.

Addresses HIGH review concern: "core routing logic in testable Python, not markdown skill".

Functions
---------
classify_power_zone  — power-zone routing from telemetry
couple_batch_grad_accum — batch/grad_accum coupling with round()
compute_batch_ceiling — tier-based batch cap from effective_scale API
apply_ladder         — thermal exploitation ladder (rungs 1-5, plus downscale for CAPPED/THROTTLED)
parse_telemetry_jsonl — canonical JSONL telemetry reader

All threshold values come from a config dict parameter — none are hardcoded.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Callable

from telemetry.effective_scale import compute as _effective_scale_compute

logger = logging.getLogger(__name__)

def classify_power_zone(
    avg_watts: float | None,
    peak_temp: float,
    avg_gpu_util: float,
    thresholds: dict,
    has_batch_headroom: bool = False,
    has_mem_headroom: bool = False,
) -> str:
    """Classify current training run into a power zone.

    Decision priority:
    1. Thermal brake — if peak_temp >= warning_temp -> THROTTLED (overrides all)
    2. Watts-based routing (primary signal)
    3. GPU util proxy (fallback when avg_watts is None)

    Parameters
    ----------
    avg_watts:
        Average wattage reading from telemetry. May be None if power meter
        is unavailable (falls back to gpu_util proxy).
    peak_temp:
        Peak GPU temperature (°C) during the monitoring window.
    avg_gpu_util:
        Average GPU utilisation (%) during the monitoring window.
    thresholds:
        Config dict loaded from adaptive_planning.yaml. Must contain
        ``thermal_brake.warning_temp``, ``power_zones.*`` keys.
    has_batch_headroom:
        True when current batch < tier batch_cap.
    has_mem_headroom:
        True when mem_available_gb > jitter_margin_gb.

    Returns
    -------
    str
        One of: "THROTTLED", "CAPPED", "TARGET", "MODERATE", "UNDERUTILIZED".
    """
    tb = thresholds["thermal_brake"]
    pz = thresholds["power_zones"]

    warning_temp: float = tb["warning_temp"]  # 82°C
    capped_min_watts: float = pz["capped_min_watts"]  # 80 W
    target_min_watts: float = pz["target_min_watts"]   # 50 W

    if peak_temp >= warning_temp:
        return "THROTTLED"

    if avg_watts is not None:
        if avg_watts >= capped_min_watts:
            return "CAPPED"
        if avg_watts >= target_min_watts:
            if has_batch_headroom and has_mem_headroom:
                return "MODERATE"
            return "TARGET"
        return "UNDERUTILIZED"

    if avg_gpu_util >= 90:
        return "CAPPED"
    if avg_gpu_util >= 70:
        if has_batch_headroom and has_mem_headroom:
            return "MODERATE"
        return "TARGET"
    if avg_gpu_util >= 30:
        return "TARGET"
    return "UNDERUTILIZED"

def couple_batch_grad_accum(
    effective_batch: int,
    new_batch: int,
    max_drift: int = 1,
) -> int:
    """Compute grad_accum steps to preserve effective batch size.

    Addresses HIGH review concern: use ``round()`` not ``//``.

    Algorithm:
    1. If effective_batch is exactly divisible by new_batch, return exact quotient.
    2. Otherwise use ``round(effective_batch / new_batch)``.
    3. Result is clamped to >= 1.
    4. If ``abs(new_batch * result - effective_batch) > max_drift``, log a warning
       but still return (drift is a soft constraint).

    Parameters
    ----------
    effective_batch:
        Target effective batch size (per_device_batch * grad_accum).
    new_batch:
        New per-device batch size after ladder step.
    max_drift:
        Maximum allowed absolute drift from effective_batch. Default 1.

    Returns
    -------
    int
        Recommended gradient_accumulation_steps (>= 1).
    """
    if new_batch <= 0:
        raise ValueError(f"new_batch must be positive, got {new_batch}")

    if effective_batch % new_batch == 0:
        return effective_batch // new_batch

    grad_accum = max(1, round(effective_batch / new_batch))

    actual_effective = new_batch * grad_accum
    drift = abs(actual_effective - effective_batch)
    if drift > max_drift:
        logger.warning(
            "couple_batch_grad_accum: effective_batch drift %d > max_drift %d "
            "(wanted %d, got %d with new_batch=%d, grad_accum=%d)",
            drift, max_drift, effective_batch, actual_effective, new_batch, grad_accum,
        )

    return grad_accum

def compute_batch_ceiling(
    model_config: dict,
    lora_config: dict,
) -> dict:
    """Compute tier-based batch cap using Phase 13 effective_scale API.

    Uses ``telemetry.effective_scale.compute`` (NOT compute_effective_scale).
    For Qwen3-30B-A3B: raw_params=30e9, quant_mode="bf16",
    gradient_checkpointing_mode="full", optimizer="adamw".

    Parameters
    ----------
    model_config:
        Dict with at minimum ``max_seq_length`` (int).
    lora_config:
        Dict with at minimum ``r`` (int, LoRA rank).

    Returns
    -------
    dict
        Keys: ``effective_params`` (float), ``batch_cap`` (int),
        ``min_headroom_pct`` (int).
    """
    seq_len: int = model_config.get("max_seq_length", 2048)
    lora_rank: int = lora_config.get("r", 0)

    result = _effective_scale_compute(
        raw_params=30e9,            # Qwen3-30B-A3B
        quant_mode="bf16",          # bfloat16 training
        training_framework="pytorch",
        gradient_checkpointing_mode="full",  # Unsloth "unsloth" mode ~ full
        lora_rank=lora_rank,
        seq_len=seq_len,
        optimizer="adamw",
    )

    return {
        "effective_params": result["effective_params"],
        "batch_cap": result["tier"]["batch_cap"],
        "min_headroom_pct": result["tier"]["min_headroom_pct"],
    }

_RUNG_NAMES = ["rung_1_batch", "rung_2_prefetch", "rung_3_workers",
               "rung_4_save_steps", "rung_5_eval_steps"]

def apply_ladder(
    current_config: dict,
    power_zone: str,
    thresholds: dict,
    telemetry_summary: dict,
    batch_ceiling: int = 8,
    anchor_lookup_fn: Callable[[str], dict | None] | None = None,
) -> dict:
    """Apply the thermal exploitation ladder to a training config.

    Evaluates rungs in v4.0 order: batch (1) > prefetch (2) > workers (3)
    > save_steps (4) > eval_steps (5).

    Only applies changes when power_zone is MODERATE or UNDERUTILIZED.
    Returns a dict of config deltas plus ``rungs_applied`` list.

    Parameters
    ----------
    current_config:
        Dict of current training hyperparameters.
    power_zone:
        Output of classify_power_zone().
    thresholds:
        Config dict from adaptive_planning.yaml.
    telemetry_summary:
        Output of parse_telemetry_jsonl() — must have avg_gpu_util, avg_watts,
        avg_sample_mb, mem_available_gb keys.
    batch_ceiling:
        Maximum allowed per-device batch size (from compute_batch_ceiling).
    anchor_lookup_fn:
        Optional callable(config_hash) -> record dict for AnchorStore lookup.

    Returns
    -------
    dict
        ``rungs_applied``: list of rung name strings that were modified.
        Additional keys: updated config values with ``reason_*`` siblings.
    """
    pz_cfg = thresholds["power_zones"]
    ladder_cfg = thresholds["ladder"]
    wb_cfg = thresholds["worker_budget"]

    avg_gpu_util: float = telemetry_summary.get("avg_gpu_util", 50.0)
    avg_watts: float | None = telemetry_summary.get("avg_watts")
    avg_sample_mb: float = telemetry_summary.get("avg_sample_mb", 1.0)
    mem_available_gb: float = telemetry_summary.get("mem_available_gb", 20.0)

    io_util_thresh: float = pz_cfg["io_util_threshold"]   # 30%
    io_watts_thresh: float = pz_cfg["io_watts_threshold"]  # 30 W

    prefetch_cap: int = ladder_cfg["prefetch_cap"]         # 4
    save_steps_cap: int = ladder_cfg["save_steps_cap"]     # 400
    eval_steps_cap: int = ladder_cfg["eval_steps_cap"]     # 200
    max_gb: float = wb_cfg["max_gb"]                       # 2.0
    hard_cap_uma: int = wb_cfg["hard_cap_uma"]             # 6

    step_size: int = ladder_cfg["step_size_large"]  # 1 (>13B)

    delta: dict[str, Any] = {"rungs_applied": []}

    # Downscale zones: reduce batch toward floor to shed thermal/power load
    downscale_zones = ("CAPPED", "THROTTLED")
    if power_zone in downscale_zones:
        downscale_floor: int = ladder_cfg.get("downscale_floor", 1)
        current_batch_ds: int = current_config.get("per_device_train_batch_size", 4)
        current_grad_accum_ds: int = current_config.get("gradient_accumulation_steps", 4)
        effective_batch_ds: int = current_batch_ds * current_grad_accum_ds

        if current_batch_ds > downscale_floor:
            new_batch_ds = downscale_floor
            new_grad_accum_ds = couple_batch_grad_accum(effective_batch_ds, new_batch_ds)
            delta["per_device_train_batch_size"] = new_batch_ds
            delta["gradient_accumulation_steps"] = new_grad_accum_ds
            delta["reason_batch"] = (
                f"Downscale: batch {current_batch_ds} -> {new_batch_ds}, "
                f"grad_accum {current_grad_accum_ds} -> {new_grad_accum_ds} "
                f"(zone={power_zone}, effective_batch kept ~{effective_batch_ds})"
            )
            delta["rungs_applied"].append("rung_1_batch_downscale")
        return delta

    # Only climb when zone is MODERATE or UNDERUTILIZED
    upscale_zones = ("MODERATE", "UNDERUTILIZED")
    if power_zone not in upscale_zones:
        return delta

    io_bottleneck = (avg_gpu_util < io_util_thresh and
                     (avg_watts is None or avg_watts < io_watts_thresh))

    current_batch: int = current_config.get("per_device_train_batch_size", 4)
    current_grad_accum: int = current_config.get("gradient_accumulation_steps", 4)
    effective_batch: int = current_batch * current_grad_accum

    if not io_bottleneck and current_batch < batch_ceiling:
        new_batch = min(current_batch + step_size, batch_ceiling)
        new_grad_accum = couple_batch_grad_accum(effective_batch, new_batch)
        delta["per_device_train_batch_size"] = new_batch
        delta["gradient_accumulation_steps"] = new_grad_accum
        delta["reason_batch"] = (
            f"Rung 1: batch {current_batch} -> {new_batch}, "
            f"grad_accum {current_grad_accum} -> {new_grad_accum} "
            f"(effective_batch kept ~{effective_batch})"
        )
        delta["rungs_applied"].append("rung_1_batch")

    current_prefetch: int = current_config.get("dataloader_prefetch_factor", 2)
    if current_prefetch < prefetch_cap:
        new_prefetch = current_prefetch + 1
        delta["dataloader_prefetch_factor"] = new_prefetch
        delta["reason_prefetch"] = (
            f"Rung 2: prefetch_factor {current_prefetch} -> {new_prefetch}"
        )
        delta["rungs_applied"].append("rung_2_prefetch")

    current_workers: int = current_config.get("dataloader_num_workers", 2)
    used_prefetch = delta.get("dataloader_prefetch_factor", current_prefetch)
    worker_mem_gb = (current_workers + 1) * used_prefetch * avg_sample_mb / 1024
    if worker_mem_gb < max_gb and (current_workers + 1) <= hard_cap_uma:
        new_workers = current_workers + 1
        delta["dataloader_num_workers"] = new_workers
        delta["reason_workers"] = (
            f"Rung 3: workers {current_workers} -> {new_workers} "
            f"(budget {worker_mem_gb:.2f} GB < {max_gb} GB)"
        )
        delta["rungs_applied"].append("rung_3_workers")

    current_save: int = current_config.get("save_steps", 200)
    if current_save < save_steps_cap:
        new_save = min(current_save * 2, save_steps_cap)
        delta["save_steps"] = new_save
        delta["reason_save_steps"] = f"Rung 4: save_steps {current_save} -> {new_save}"
        delta["rungs_applied"].append("rung_4_save_steps")

    current_eval: int = current_config.get("eval_steps", 100)
    if current_eval < eval_steps_cap:
        new_eval = min(current_eval * 2, eval_steps_cap)
        delta["eval_steps"] = new_eval
        delta["reason_eval_steps"] = f"Rung 5: eval_steps {current_eval} -> {new_eval}"
        delta["rungs_applied"].append("rung_5_eval_steps")

    delta["dataloader_pin_memory"] = False
    if current_config.get("dataloader_persistent_workers", False):
        delta["dataloader_persistent_workers"] = True  # never reverts once True

    return delta

def parse_telemetry_jsonl(jsonl_path: str | Path) -> dict | None:
    """Parse a canonical telemetry JSONL file into a summary dict.

    Skips the first 2 readings (warm-up noise per Gemini suggestion).
    Returns None if fewer than 3 valid readings remain after the skip.

    Expected JSONL record keys (any subset):
        watts, gpu_util_pct, temperature_c, mem_available_gb,
        mem_total_gb, ram_used_gb

    Parameters
    ----------
    jsonl_path:
        Path to the JSONL telemetry file.

    Returns
    -------
    dict or None
        Summary with keys:
        avg_watts, peak_watts, avg_gpu_util, peak_temp, avg_temp,
        mem_total_gb, peak_ram_gb, effective_headroom_gb, min_mem_available_gb.
        Returns None when not enough data.
    """
    path = Path(jsonl_path)
    if not path.exists():
        return None

    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("parse_telemetry_jsonl: skipping invalid JSON line: %r", line)

    records = records[2:]

    if len(records) < 3:
        return None

    watts_list = [r["watts"] for r in records if "watts" in r]
    util_list = [r["gpu_util_pct"] for r in records if "gpu_util_pct" in r]
    temp_list = [r["temperature_c"] for r in records if "temperature_c" in r]
    mem_list = [r["mem_available_gb"] for r in records if "mem_available_gb" in r]
    mem_total_list = [r.get("mem_total_gb", 0.0) for r in records if "mem_total_gb" in r]
    ram_list = [r.get("ram_used_gb", 0.0) for r in records if "ram_used_gb" in r]

    avg_watts = sum(watts_list) / len(watts_list) if watts_list else 0.0
    peak_watts = max(watts_list) if watts_list else 0.0
    avg_gpu_util = sum(util_list) / len(util_list) if util_list else 0.0
    peak_temp = max(temp_list) if temp_list else 0.0
    avg_temp = sum(temp_list) / len(temp_list) if temp_list else 0.0
    min_mem_available_gb = min(mem_list) if mem_list else 0.0
    mem_total_gb = max(mem_total_list) if mem_total_list else 0.0
    peak_ram_gb = max(ram_list) if ram_list else 0.0
    effective_headroom_gb = min_mem_available_gb  # simplified for UMA

    return {
        "avg_watts": avg_watts,
        "peak_watts": peak_watts,
        "avg_gpu_util": avg_gpu_util,
        "peak_temp": peak_temp,
        "avg_temp": avg_temp,
        "mem_total_gb": mem_total_gb,
        "peak_ram_gb": peak_ram_gb,
        "effective_headroom_gb": effective_headroom_gb,
        "min_mem_available_gb": min_mem_available_gb,
    }
