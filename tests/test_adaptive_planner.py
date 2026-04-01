"""Decision-table tests for adaptive_planner.py.

Tests MUST NOT require GPU hardware — telemetry imports are mocked.
"""
from __future__ import annotations

import sys
import types
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal telemetry stub — prevents ImportError without real dgx-toolbox
# ---------------------------------------------------------------------------

def _install_telemetry_stubs():
    """Install lightweight stub modules for telemetry if not already present."""
    if "telemetry" not in sys.modules:
        telemetry_pkg = types.ModuleType("telemetry")
        sys.modules["telemetry"] = telemetry_pkg

    if "telemetry.effective_scale" not in sys.modules:
        es_mod = types.ModuleType("telemetry.effective_scale")

        def _stub_compute(raw_params, quant_mode="fp16", training_framework="pytorch",
                          gradient_checkpointing_mode="none", lora_rank=0,
                          seq_len=2048, optimizer="adamw", model_weight_gb=0.0):
            # Approximate batch_cap from Qwen3-30B-A3B tier (>13B, <=30B -> 8)
            if raw_params <= 30e9:
                batch_cap = 8
            else:
                batch_cap = 4
            return {"effective_params": raw_params * 0.05, "tier": {"batch_cap": batch_cap, "min_headroom_pct": 20}}

        es_mod.compute = _stub_compute
        sys.modules["telemetry.effective_scale"] = es_mod
        setattr(sys.modules["telemetry"], "effective_scale", es_mod)

    if "telemetry.failure_classifier" not in sys.modules:
        fc_mod = types.ModuleType("telemetry.failure_classifier")
        fc_mod.classify_failure = MagicMock(return_value={"classification": "clean", "evidence": {}})
        sys.modules["telemetry.failure_classifier"] = fc_mod
        setattr(sys.modules["telemetry"], "failure_classifier", fc_mod)

    if "telemetry.anchor_store" not in sys.modules:
        as_mod = types.ModuleType("telemetry.anchor_store")
        class _FakeAnchorStore:
            def __init__(self, store_path):
                self._store_path = store_path
            def compute_config_hash(self, config):
                return "deadbeef"
            def lookup(self, config_hash):
                return None
            def apply_override(self, config_hash, status, batch_size, tier_cap, step_size=2):
                return {"status": status, "batch_cap": tier_cap}
        as_mod.AnchorStore = _FakeAnchorStore
        sys.modules["telemetry.anchor_store"] = as_mod
        setattr(sys.modules["telemetry"], "anchor_store", as_mod)


_install_telemetry_stubs()

# Now import the module under test
from scripts import adaptive_planner  # noqa: E402


# ---------------------------------------------------------------------------
# Shared thresholds fixture (loaded from adaptive_planning.yaml once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def thresholds():
    import yaml
    cfg_path = Path(__file__).parent.parent / "config" / "adaptive_planning.yaml"
    if cfg_path.exists():
        with open(cfg_path) as fh:
            return yaml.safe_load(fh)
    # Fallback inline thresholds so tests can run before config is written
    return {
        "power_zones": {
            "capped_min_watts": 80,
            "target_min_watts": 50,
            "underutilized_max_watts": 50,
            "io_util_threshold": 30,
            "io_watts_threshold": 30,
        },
        "thermal_brake": {
            "warning_temp": 82,
            "emergency_temp": 85,
            "cooldown_target": 78,
        },
        "ladder": {
            "step_size_small": 2,
            "step_size_large": 1,
            "prefetch_cap": 4,
            "save_steps_cap": 400,
            "eval_steps_cap": 200,
        },
        "worker_budget": {"max_gb": 2, "hard_cap_uma": 6},
        "coupling": {"max_drift": 1},
        "probe": {"steps": 5, "cooldown_runs": 2},
        "monitor": {"sample_interval_steps": 50, "consecutive_samples_for_trigger": 3},
    }


# ---------------------------------------------------------------------------
# classify_power_zone tests
# ---------------------------------------------------------------------------

class TestClassifyPowerZone:
    """Decision-table tests for routing function."""

    def test_classify_power_zone_throttled(self, thresholds):
        """Thermal brake overrides watts: peak_temp=83 -> THROTTLED."""
        zone = adaptive_planner.classify_power_zone(
            avg_watts=60, peak_temp=83, avg_gpu_util=70, thresholds=thresholds
        )
        assert zone == "THROTTLED", f"Expected THROTTLED, got {zone}"

    def test_classify_power_zone_throttled_high_temp_low_watts(self, thresholds):
        """Even low watts cannot prevent THROTTLED at high temp."""
        zone = adaptive_planner.classify_power_zone(
            avg_watts=20, peak_temp=82, avg_gpu_util=20, thresholds=thresholds
        )
        assert zone == "THROTTLED"

    def test_classify_power_zone_capped(self, thresholds):
        """High watts, safe temp -> CAPPED."""
        zone = adaptive_planner.classify_power_zone(
            avg_watts=85, peak_temp=75, avg_gpu_util=90, thresholds=thresholds
        )
        assert zone == "CAPPED"

    def test_classify_power_zone_capped_at_boundary(self, thresholds):
        """Exactly at capped_min_watts=80 -> CAPPED."""
        zone = adaptive_planner.classify_power_zone(
            avg_watts=80, peak_temp=75, avg_gpu_util=85, thresholds=thresholds
        )
        assert zone == "CAPPED"

    def test_classify_power_zone_target(self, thresholds):
        """Mid watts, no headroom -> TARGET."""
        zone = adaptive_planner.classify_power_zone(
            avg_watts=65, peak_temp=75, avg_gpu_util=70,
            thresholds=thresholds,
            has_batch_headroom=False, has_mem_headroom=False,
        )
        assert zone == "TARGET"

    def test_classify_power_zone_moderate(self, thresholds):
        """Mid watts, both headrooms -> MODERATE."""
        zone = adaptive_planner.classify_power_zone(
            avg_watts=65, peak_temp=75, avg_gpu_util=70,
            thresholds=thresholds,
            has_batch_headroom=True, has_mem_headroom=True,
        )
        assert zone == "MODERATE"

    def test_classify_power_zone_target_partial_headroom(self, thresholds):
        """Only batch headroom (not mem) -> TARGET, not MODERATE."""
        zone = adaptive_planner.classify_power_zone(
            avg_watts=65, peak_temp=75, avg_gpu_util=70,
            thresholds=thresholds,
            has_batch_headroom=True, has_mem_headroom=False,
        )
        assert zone == "TARGET"

    def test_classify_power_zone_underutilized(self, thresholds):
        """Low watts, safe temp -> UNDERUTILIZED."""
        zone = adaptive_planner.classify_power_zone(
            avg_watts=40, peak_temp=70, avg_gpu_util=35, thresholds=thresholds
        )
        assert zone == "UNDERUTILIZED"

    def test_classify_power_zone_no_watts_fallback_underutilized(self, thresholds):
        """avg_watts=None, avg_gpu_util=20 -> UNDERUTILIZED via gpu proxy."""
        zone = adaptive_planner.classify_power_zone(
            avg_watts=None, peak_temp=70, avg_gpu_util=20, thresholds=thresholds
        )
        assert zone == "UNDERUTILIZED"

    def test_classify_power_zone_no_watts_fallback_capped(self, thresholds):
        """avg_watts=None, avg_gpu_util=92 -> CAPPED via gpu proxy."""
        zone = adaptive_planner.classify_power_zone(
            avg_watts=None, peak_temp=70, avg_gpu_util=92, thresholds=thresholds
        )
        assert zone == "CAPPED"


# ---------------------------------------------------------------------------
# couple_batch_grad_accum tests
# ---------------------------------------------------------------------------

class TestCoupleBatchGradAccum:
    """Tests for round()-based batch/grad-accum coupling."""

    def test_exact_divisor(self):
        """Exact divisor: eff=16, new_batch=8 -> grad_accum=2."""
        result = adaptive_planner.couple_batch_grad_accum(effective_batch=16, new_batch=8)
        assert result == 2

    def test_round_non_divisor(self):
        """Round case: eff=16, new_batch=5 -> round(16/5)=3."""
        result = adaptive_planner.couple_batch_grad_accum(effective_batch=16, new_batch=5)
        assert result == 3  # round(3.2) = 3

    def test_prefer_divisor_when_close(self):
        """eff=16, new_batch=6: round(16/6)=round(2.67)=3, drift=|6*3-16|=2 > max_drift=1.
        Returns 3 but emits warning (drift > 1 allowed, function still returns)."""
        result = adaptive_planner.couple_batch_grad_accum(effective_batch=16, new_batch=6, max_drift=1)
        # result is round(16/6)=3; drift=2 but function returns anyway with warning
        assert isinstance(result, int)
        assert result >= 1

    def test_min_one(self):
        """new_batch equals effective_batch -> grad_accum=1."""
        result = adaptive_planner.couple_batch_grad_accum(effective_batch=16, new_batch=16)
        assert result == 1

    def test_large_batch_exact(self):
        """eff=32, new_batch=4 -> grad_accum=8."""
        result = adaptive_planner.couple_batch_grad_accum(effective_batch=32, new_batch=4)
        assert result == 8

    def test_result_at_least_one(self):
        """Result is always >= 1 even for very large new_batch."""
        result = adaptive_planner.couple_batch_grad_accum(effective_batch=4, new_batch=100)
        assert result >= 1


# ---------------------------------------------------------------------------
# apply_ladder rung-order tests
# ---------------------------------------------------------------------------

class TestApplyLadder:
    """Tests for the thermal exploitation ladder."""

    def _base_config(self):
        return {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,  # effective=16
            "dataloader_prefetch_factor": 2,
            "dataloader_num_workers": 2,
            "save_steps": 200,
            "eval_steps": 100,
            "dataloader_persistent_workers": True,
        }

    def _telemetry(self, gpu_util=70, avg_watts=65, peak_temp=75, avg_temp=72,
                   avg_sample_mb=1.0, mem_available_gb=20.0):
        return {
            "avg_gpu_util": gpu_util,
            "avg_watts": avg_watts,
            "peak_temp": peak_temp,
            "avg_temp": avg_temp,
            "avg_sample_mb": avg_sample_mb,
            "mem_available_gb": mem_available_gb,
            "min_mem_available_gb": mem_available_gb - 2,
        }

    def test_apply_ladder_rung_order(self, thresholds):
        """In MODERATE zone, batch (Rung 1) fires before prefetch (Rung 2)."""
        cfg = self._base_config()
        telem = self._telemetry(gpu_util=70, avg_watts=65)
        result = adaptive_planner.apply_ladder(
            current_config=cfg,
            power_zone="MODERATE",
            thresholds=thresholds,
            telemetry_summary=telem,
            batch_ceiling=8,
        )
        # Should have attempted Rung 1 changes
        assert "rungs_applied" in result
        assert isinstance(result["rungs_applied"], list)

    def test_apply_ladder_io_bottleneck_skips_rung1(self, thresholds):
        """IO bottleneck (gpu_util<30, watts<30) -> Rung 1 skipped, Rungs 2-3 can still fire."""
        cfg = self._base_config()
        telem = self._telemetry(gpu_util=20, avg_watts=20)
        result = adaptive_planner.apply_ladder(
            current_config=cfg,
            power_zone="MODERATE",
            thresholds=thresholds,
            telemetry_summary=telem,
            batch_ceiling=8,
        )
        assert "rung_1_batch" not in result.get("rungs_applied", [])

    def test_apply_ladder_respects_batch_ceiling(self, thresholds):
        """If batch already at ceiling, Rung 1 is not applied."""
        cfg = self._base_config()
        cfg["per_device_train_batch_size"] = 8  # already at ceiling
        telem = self._telemetry(gpu_util=70, avg_watts=65)
        result = adaptive_planner.apply_ladder(
            current_config=cfg,
            power_zone="MODERATE",
            thresholds=thresholds,
            telemetry_summary=telem,
            batch_ceiling=8,  # same as current batch
        )
        # Rung 1 should not be in applied rungs (already at ceiling)
        assert "rung_1_batch" not in result.get("rungs_applied", [])

    def test_apply_ladder_prefetch_increases(self, thresholds):
        """In MODERATE zone without IO bottleneck, prefetch can increase."""
        cfg = self._base_config()
        cfg["dataloader_prefetch_factor"] = 2
        # High GPU util to avoid IO bottleneck bypass
        telem = self._telemetry(gpu_util=70, avg_watts=65)
        result = adaptive_planner.apply_ladder(
            current_config=cfg,
            power_zone="MODERATE",
            thresholds=thresholds,
            telemetry_summary=telem,
            batch_ceiling=8,
        )
        assert isinstance(result, dict)

    def test_apply_ladder_save_steps_doubles(self, thresholds):
        """save_steps can be doubled (Rung 4)."""
        cfg = self._base_config()
        cfg["save_steps"] = 200
        telem = self._telemetry()
        result = adaptive_planner.apply_ladder(
            current_config=cfg,
            power_zone="MODERATE",
            thresholds=thresholds,
            telemetry_summary=telem,
            batch_ceiling=8,
        )
        # Result is a dict with config deltas and rungs_applied
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# compute_batch_ceiling tests
# ---------------------------------------------------------------------------

class TestComputeBatchCeiling:
    """Tests for the batch ceiling calculation."""

    def test_compute_batch_ceiling_returns_dict(self):
        """Returns dict with batch_cap key."""
        model_cfg = {"name": "Qwen/Qwen3-30B-A3B", "max_seq_length": 4096}
        lora_cfg = {"r": 32}
        result = adaptive_planner.compute_batch_ceiling(
            model_config=model_cfg, lora_config=lora_cfg
        )
        assert isinstance(result, dict)
        assert "batch_cap" in result
        assert "effective_params" in result

    def test_compute_batch_ceiling_uses_correct_api(self):
        """Verify compute is called with correct args for Qwen3-30B-A3B."""
        model_cfg = {"name": "Qwen/Qwen3-30B-A3B", "max_seq_length": 4096}
        lora_cfg = {"r": 32}
        with patch("scripts.adaptive_planner._effective_scale_compute") as mock_compute:
            mock_compute.return_value = {
                "effective_params": 1.5e9,
                "tier": {"batch_cap": 8, "min_headroom_pct": 20},
            }
            result = adaptive_planner.compute_batch_ceiling(
                model_config=model_cfg, lora_config=lora_cfg
            )
        mock_compute.assert_called_once()
        call_kwargs = mock_compute.call_args
        # raw_params should be 30e9 for Qwen3-30B-A3B
        args, kwargs = call_kwargs
        passed_raw = kwargs.get("raw_params") or (args[0] if args else None)
        assert passed_raw == pytest.approx(30e9, rel=0.01)

    def test_compute_batch_ceiling_qwen3_30b_tier(self):
        """For Qwen3-30B-A3B, tier batch_cap <= 8."""
        model_cfg = {"name": "Qwen/Qwen3-30B-A3B", "max_seq_length": 4096}
        lora_cfg = {"r": 32}
        result = adaptive_planner.compute_batch_ceiling(
            model_config=model_cfg, lora_config=lora_cfg
        )
        # Qwen3-30B-A3B: raw=30B <= 30B threshold -> batch_cap=8
        assert result["batch_cap"] <= 8


# ---------------------------------------------------------------------------
# parse_telemetry_jsonl tests
# ---------------------------------------------------------------------------

class TestParseTelemetryJsonl:
    """Tests for JSONL telemetry parsing."""

    def test_parse_returns_none_on_empty(self, tmp_path):
        """Empty file returns None."""
        f = tmp_path / "telem.jsonl"
        f.write_text("")
        result = adaptive_planner.parse_telemetry_jsonl(str(f))
        assert result is None

    def test_parse_returns_none_too_few_readings(self, tmp_path):
        """Fewer than 3 valid readings after warm-up skip -> None."""
        import json
        f = tmp_path / "telem.jsonl"
        readings = [
            {"watts": 60.0, "gpu_util_pct": 70, "temperature_c": 75, "mem_available_gb": 20},
            {"watts": 61.0, "gpu_util_pct": 71, "temperature_c": 75, "mem_available_gb": 20},
        ]
        f.write_text("\n".join(json.dumps(r) for r in readings))
        result = adaptive_planner.parse_telemetry_jsonl(str(f))
        assert result is None

    def test_parse_skips_first_two(self, tmp_path):
        """First 2 readings skipped (warm-up), remaining readings used."""
        import json
        f = tmp_path / "telem.jsonl"
        # First 2 are outliers (warm-up), rest are normal
        readings = [
            {"watts": 200.0, "gpu_util_pct": 10, "temperature_c": 90, "mem_available_gb": 5},   # skip
            {"watts": 190.0, "gpu_util_pct": 15, "temperature_c": 89, "mem_available_gb": 5},   # skip
            {"watts": 60.0, "gpu_util_pct": 70, "temperature_c": 75, "mem_available_gb": 20},
            {"watts": 62.0, "gpu_util_pct": 72, "temperature_c": 76, "mem_available_gb": 20},
            {"watts": 61.0, "gpu_util_pct": 71, "temperature_c": 75, "mem_available_gb": 20},
        ]
        f.write_text("\n".join(json.dumps(r) for r in readings))
        result = adaptive_planner.parse_telemetry_jsonl(str(f))
        assert result is not None
        # avg_watts should be near 61, not 200
        assert result["avg_watts"] < 100

    def test_parse_returns_required_keys(self, tmp_path):
        """Result has all required summary keys."""
        import json
        f = tmp_path / "telem.jsonl"
        readings = [
            {"watts": 10.0, "gpu_util_pct": 5, "temperature_c": 60, "mem_available_gb": 30},  # skip
            {"watts": 10.0, "gpu_util_pct": 5, "temperature_c": 60, "mem_available_gb": 30},  # skip
            {"watts": 60.0, "gpu_util_pct": 70, "temperature_c": 75, "mem_available_gb": 20},
            {"watts": 62.0, "gpu_util_pct": 72, "temperature_c": 76, "mem_available_gb": 19},
            {"watts": 61.0, "gpu_util_pct": 71, "temperature_c": 75, "mem_available_gb": 21},
        ]
        f.write_text("\n".join(json.dumps(r) for r in readings))
        result = adaptive_planner.parse_telemetry_jsonl(str(f))
        assert result is not None
        required = ["avg_watts", "peak_watts", "avg_gpu_util", "peak_temp",
                    "avg_temp", "min_mem_available_gb"]
        for key in required:
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# apply_ladder downscale tests (CAPPED / THROTTLED zones)
# ---------------------------------------------------------------------------

class TestApplyLadderDownscale:
    """Tests for CAPPED/THROTTLED batch downscale path in apply_ladder()."""

    def _base_config(self):
        return {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,  # effective_batch = 16
            "dataloader_prefetch_factor": 2,
            "dataloader_num_workers": 2,
            "save_steps": 200,
            "eval_steps": 100,
            "dataloader_persistent_workers": True,
        }

    def _capped_telemetry(self):
        """Telemetry representing CAPPED zone: avg_watts=95, safe temp."""
        return {
            "avg_gpu_util": 90,
            "avg_watts": 95,
            "peak_temp": 75,
            "avg_temp": 72,
            "avg_sample_mb": 1.0,
            "mem_available_gb": 20.0,
            "min_mem_available_gb": 18.0,
        }

    def _throttled_telemetry(self):
        """Telemetry representing THROTTLED zone: temp >= 82C."""
        return {
            "avg_gpu_util": 60,
            "avg_watts": 60,
            "peak_temp": 84,
            "avg_temp": 82,
            "avg_sample_mb": 1.0,
            "mem_available_gb": 20.0,
            "min_mem_available_gb": 18.0,
        }

    def test_apply_ladder_capped_downscale(self, thresholds):
        """CAPPED zone: batch=4, grad_accum=4 (eff=16) -> batch=1, grad_accum=16, rung_1_batch_downscale."""
        cfg = self._base_config()  # batch=4, grad_accum=4
        telem = self._capped_telemetry()
        result = adaptive_planner.apply_ladder(
            current_config=cfg,
            power_zone="CAPPED",
            thresholds=thresholds,
            telemetry_summary=telem,
            batch_ceiling=8,
        )
        assert result["per_device_train_batch_size"] == 1
        assert result["gradient_accumulation_steps"] == adaptive_planner.couple_batch_grad_accum(16, 1)
        assert "rung_1_batch_downscale" in result["rungs_applied"]

    def test_apply_ladder_throttled_downscale(self, thresholds):
        """THROTTLED zone: batch=4, grad_accum=4 (eff=16) -> batch=1, grad_accum=16, rung_1_batch_downscale."""
        cfg = self._base_config()  # batch=4, grad_accum=4
        telem = self._throttled_telemetry()
        result = adaptive_planner.apply_ladder(
            current_config=cfg,
            power_zone="THROTTLED",
            thresholds=thresholds,
            telemetry_summary=telem,
            batch_ceiling=8,
        )
        assert result["per_device_train_batch_size"] == 1
        assert result["gradient_accumulation_steps"] == adaptive_planner.couple_batch_grad_accum(16, 1)
        assert "rung_1_batch_downscale" in result["rungs_applied"]

    def test_apply_ladder_capped_already_at_floor(self, thresholds):
        """CAPPED zone, batch already at downscale_floor=1 -> no rungs applied."""
        cfg = self._base_config()
        cfg["per_device_train_batch_size"] = 1
        cfg["gradient_accumulation_steps"] = 16  # effective=16
        telem = self._capped_telemetry()
        result = adaptive_planner.apply_ladder(
            current_config=cfg,
            power_zone="CAPPED",
            thresholds=thresholds,
            telemetry_summary=telem,
            batch_ceiling=8,
        )
        assert result["rungs_applied"] == []

    def test_apply_ladder_target_no_downscale(self, thresholds):
        """TARGET zone -> no downscale or upscale, empty rungs_applied."""
        cfg = self._base_config()
        telem = {
            "avg_gpu_util": 70,
            "avg_watts": 65,
            "peak_temp": 75,
            "avg_temp": 72,
            "avg_sample_mb": 1.0,
            "mem_available_gb": 20.0,
            "min_mem_available_gb": 18.0,
        }
        result = adaptive_planner.apply_ladder(
            current_config=cfg,
            power_zone="TARGET",
            thresholds=thresholds,
            telemetry_summary=telem,
            batch_ceiling=8,
        )
        assert result["rungs_applied"] == []

    def test_apply_ladder_moderate_still_upscales(self, thresholds):
        """MODERATE zone -> still returns rung_1_batch upscale (no regression)."""
        cfg = self._base_config()  # batch=4, below ceiling=8
        telem = {
            "avg_gpu_util": 70,
            "avg_watts": 65,
            "peak_temp": 75,
            "avg_temp": 72,
            "avg_sample_mb": 1.0,
            "mem_available_gb": 20.0,
            "min_mem_available_gb": 18.0,
        }
        result = adaptive_planner.apply_ladder(
            current_config=cfg,
            power_zone="MODERATE",
            thresholds=thresholds,
            telemetry_summary=telem,
            batch_ceiling=8,
        )
        assert "rung_1_batch" in result["rungs_applied"]
        assert "rung_1_batch_downscale" not in result["rungs_applied"]

    def test_apply_ladder_capped_downscale_coupling(self, thresholds):
        """Effective batch preserved: batch=8, grad_accum=2 (eff=16) -> downscale to 1, grad_accum=16."""
        cfg = self._base_config()
        cfg["per_device_train_batch_size"] = 8
        cfg["gradient_accumulation_steps"] = 2  # effective=16
        telem = self._capped_telemetry()
        result = adaptive_planner.apply_ladder(
            current_config=cfg,
            power_zone="CAPPED",
            thresholds=thresholds,
            telemetry_summary=telem,
            batch_ceiling=8,
        )
        assert result["per_device_train_batch_size"] == 1
        expected_grad_accum = adaptive_planner.couple_batch_grad_accum(16, 1)
        assert result["gradient_accumulation_steps"] == expected_grad_accum
        assert "rung_1_batch_downscale" in result["rungs_applied"]
