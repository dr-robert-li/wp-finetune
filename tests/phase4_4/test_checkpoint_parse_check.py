"""GPU-free unit tests for scripts/checkpoint_parse_check.py 04.3-03 additions.

Covers (Task 1(d)): CLI plumbing of --include-streams / --max-new-tokens / --no-adapter
/ --out / --binding-dryrun; the capture-compatible structural histogram + terse identity;
and the runtime MoE-binding guard verdict logic (representation-agnostic structural fallback
— the forward-activation-delta PRIMARY path needs a real model and is validated live in the
--binding-dryrun, eyeballed in binding_dryrun.md). No Unsloth/vLLM boot here.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

cpc = pytest.importorskip("scripts.checkpoint_parse_check")


# --------------------------------------------------------------------------- #
# helpers / stubs
# --------------------------------------------------------------------------- #

class _StubModel:
    """Minimal model exposing named_modules/named_parameters for probe_moe_binding.
    No experts submodule and no disable_adapter -> the activation-delta path fails and
    the probe falls back to the representation-agnostic structural-name verdict."""

    def __init__(self, module_names, param_names=None):
        self._mods = list(module_names)
        self._pars = list(param_names or [])

    def named_modules(self):
        return [(n, object()) for n in self._mods]

    def named_parameters(self):
        return [(n, object()) for n in self._pars]


# --------------------------------------------------------------------------- #
# structural histogram + terse identity
# --------------------------------------------------------------------------- #

class TestStructuralHistogram:
    def test_keys_match_capture_and_terse_identity(self):
        responses = [
            "prose[/REASONING]<judge_output>{}</judge_output>",  # close tag + judge_output
            "<think></think>\n{bare json, no close tag}",         # terse
            "more prose[/REASONING] tail",                        # close tag, no judge_output
        ]
        parsed = [{"overall_score": 7}, None, {"verdict": "PASS"}]  # 2nd unparseable, 3rd no overall_score
        h = cpc._structural_histogram(responses, parsed)
        for k in ("n_total", "n_with_close_tag", "n_with_judge_output",
                  "n_parseable_scores", "parseable_rate"):
            assert k in h, f"missing capture-compatible key {k}"
        assert h["n_total"] == 3
        assert h["n_with_close_tag"] == 2
        # terse is computed identically to 04.3-02: n_total - n_with_close_tag
        assert h["n_total"] - h["n_with_close_tag"] == 1
        assert h["n_with_judge_output"] == 1
        # n_parseable requires parsed is not None AND has overall_score
        assert h["n_parseable_scores"] == 1
        assert h["parseable_rate"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# runtime MoE-binding guard verdict (structural fallback path)
# --------------------------------------------------------------------------- #

class TestBindingGuard:
    def test_raises_binding_failed_on_attention_only(self):
        """Attention-only bind (q/k/v/o LoRA, NO mlp.experts.* LoRA) -> BINDING_FAILED.
        This is the false-MERGE_ARTIFACT path the guard must block."""
        m = _StubModel([
            "model.layers.0.self_attn.q_proj.lora_A",
            "model.layers.0.self_attn.q_proj.lora_B",
            "model.layers.0.self_attn.v_proj.lora_A",
        ])
        rep = cpc.probe_moe_binding(m, tokenizer=object())
        assert rep["verdict"] == "BINDING_FAILED"
        assert rep["structural_expert_lora"] == []
        assert rep["structural_attn_lora"]  # attention LoRA WAS seen

    def test_bound_when_live_expert_lora_present(self):
        """A live expert-LoRA name on the model object -> BOUND (structural fallback,
        since the stub cannot run the activation-delta primary)."""
        m = _StubModel([
            "model.layers.0.self_attn.q_proj.lora_A",
            "model.layers.0.mlp.experts.lora_A",   # expert-targeted LoRA present
            "model.layers.0.mlp.experts.lora_B",
        ])
        rep = cpc.probe_moe_binding(m, tokenizer=object())
        assert rep["verdict"] == "BOUND"
        assert rep["structural_expert_lora"]

    def test_expert_lora_detected_via_parametrization_name(self):
        """target_parameters adapters may surface as parametrizations on the fused param
        rather than lora_A/lora_B child modules — the keyword set must still catch it."""
        m = _StubModel(
            module_names=["model.layers.0.mlp.experts"],
            param_names=["model.layers.0.mlp.experts.parametrizations.gate_up_proj.0.adapter"],
        )
        rep = cpc.probe_moe_binding(m, tokenizer=object())
        assert rep["verdict"] == "BOUND"
        assert rep["structural_expert_lora"]


# --------------------------------------------------------------------------- #
# CLI plumbing (monkeypatch the run_* funcs — no model boot)
# --------------------------------------------------------------------------- #

class TestCliPlumbing:
    def test_no_adapter_arm_plumbs_through(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(cpc, "run_checkpoint_parse_check",
                            lambda **kw: captured.update(kw) or True)
        monkeypatch.setattr(sys, "argv", [
            "checkpoint_parse_check.py", "--no-adapter",
            "--base", "models/qwen3-30b-wp-30_70-reasoning-merged",
            "--include-streams", "cot,ctf", "--max-new-tokens", "2048",
            "--n", "120", "--out", "/tmp/merged72_unsloth_histogram.json",
        ])
        cpc.main()
        assert captured["no_adapter"] is True
        assert captured["checkpoint_dir"] is None
        assert captured["include_streams"] == "cot,ctf"
        assert captured["max_new_tokens"] == 2048
        assert captured["out_path"] == "/tmp/merged72_unsloth_histogram.json"
        assert captured["n"] == 120

    def test_checkpoint_arm_plumbs_through(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(cpc, "run_checkpoint_parse_check",
                            lambda **kw: captured.update(kw) or True)
        monkeypatch.setattr(sys, "argv", [
            "checkpoint_parse_check.py",
            "--checkpoint-dir", "adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72",
            "--base", "models/qwen3-30b-wp-30_70-merged-v2",
            "--include-streams", "cot,ctf", "--max-new-tokens", "2048",
            "--n", "120", "--out", "/tmp/unmerged72_unsloth_histogram.json",
        ])
        cpc.main()
        assert captured["no_adapter"] is False
        assert captured["checkpoint_dir"].endswith("checkpoint-72")
        assert captured["out_path"].endswith("unmerged72_unsloth_histogram.json")

    def test_binding_dryrun_plumbs_through(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(cpc, "run_binding_dryrun",
                            lambda **kw: captured.update(kw) or {"verdict": "BOUND"})
        monkeypatch.setattr(sys, "argv", [
            "checkpoint_parse_check.py",
            "--checkpoint-dir", "adapters/qwen3-30b-wp-30_70-reasoning/checkpoint-72",
            "--base", "models/qwen3-30b-wp-30_70-merged-v2",
            "--binding-dryrun",
            "--binding-out", "/tmp/binding_dryrun.md",
        ])
        cpc.main()
        assert captured["checkpoint_dir"].endswith("checkpoint-72")
        assert captured["base"].endswith("merged-v2")
        assert captured["out_md"] == "/tmp/binding_dryrun.md"

    def test_binding_dryrun_requires_checkpoint(self, monkeypatch):
        # --no-adapter + --binding-dryrun: dry-run needs an adapter -> parser.error -> SystemExit
        monkeypatch.setattr(sys, "argv",
                            ["checkpoint_parse_check.py", "--no-adapter", "--binding-dryrun"])
        with pytest.raises(SystemExit):
            cpc.main()


class TestModuleSurface:
    def test_exposes_symbols(self):
        for sym in ("verify_dataset_readiness", "run_checkpoint_parse_check",
                    "run_binding_dryrun", "probe_moe_binding", "check_close_tag_survives",
                    "_structural_histogram", "_stream_of", "_user_messages", "main"):
            assert hasattr(cpc, sym), f"missing {sym}"
