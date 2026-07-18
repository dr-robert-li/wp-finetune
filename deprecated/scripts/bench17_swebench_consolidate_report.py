#!/usr/bin/env python3
"""Consolidate the two swebench harness run reports into the single BENCH-02
eval receipt (output/bench17/swebench_eval_report.json), cross-referencing the
committed pre-registration (output/bench17/swebench_scope_preregistration.md).

Primary resolved rates use the FULL pre-registered scope as denominator
(Lite n=300, PHP n=43): over-length, unparseable, patch-apply-failed and
harness-environment-failed instances all count as unresolved, per the locked
pre-registration ("counted against the model, disclosed"). Evaluated-subset
rates are reported as secondary numbers. FAIL_TO_PASS/PASS_TO_PASS bookkeeping
is the harness's own (per-instance report.json files under
logs/run_evaluation/); nothing is re-derived here.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "bench17" / "swebench_eval_report.json"
GEN_RECEIPT = json.loads((ROOT / "output/bench17/swebench_generation_receipt.json").read_text())

LEGS = {
    "lite300": {
        "harness_report": ROOT / "output/bench17/swebench_harness_report_lite300_v1.json",
        "predictions": ROOT / "output/bench17/swebench_predictions.jsonl",
        "run_id": "lite300_v1",
        "dataset": "SWE-bench/SWE-bench_Lite",
        "prompt_dataset": "princeton-nlp/SWE-bench_Lite_oracle",
        "scoped_n": 300,
    },
    "php43": {
        "harness_report": ROOT / "output/bench17/swebench_harness_report_php43_v1.json",
        "predictions": ROOT / "output/bench17/swebench_predictions_php.jsonl",
        "run_id": "php43_v1",
        "dataset": "SWE-bench/SWE-bench_Multilingual (PHP-repo subset: phpoffice/phpspreadsheet, laravel/framework, php-cs-fixer/php-cs-fixer, briannesbitt/carbon)",
        "prompt_dataset": "oracle-equivalent via swebench.inference.make_datasets (style-2, file_source=oracle)",
        "scoped_n": 43,
    },
}


def classify_errors(run_id: str, model: str, error_ids: list) -> dict:
    """Split harness error_ids into patch-apply failures (model's fault) vs
    harness environment/instance-image build failures (arm64/toolchain gaps)."""
    apply_failed, env_failed = [], []
    for iid in error_ids:
        log = ROOT / f"logs/run_evaluation/{run_id}/{model}/{iid}/run_instance.log"
        txt = log.read_text() if log.exists() else ""
        (apply_failed if "Patch Apply Failed" in txt else env_failed).append(iid)
    return {"patch_apply_failed": sorted(apply_failed), "harness_env_failed": sorted(env_failed)}


def leg_summary(name: str, cfg: dict) -> dict:
    r = json.loads(cfg["harness_report"].read_text())
    preds = [json.loads(l) for l in open(cfg["predictions"])]
    assert len(preds) == cfg["scoped_n"], (name, len(preds))
    over_length = sorted(p["instance_id"] for p in preds if p.get("_over_length"))
    gen_failed = sorted(
        p["instance_id"] for p in preds
        if not p["model_patch"] and not p.get("_over_length")
    )
    errors = classify_errors(cfg["run_id"], "qwen3-30b-wp-30_70-reasoning-merged-v4", r["error_ids"])
    resolved = len(r["resolved_ids"])
    evaluated = r["completed_instances"]  # ran to a test verdict inside a container
    return {
        "dataset": cfg["dataset"],
        "prompt_source": cfg["prompt_dataset"],
        "run_id": cfg["run_id"],
        "scoped_instances": cfg["scoped_n"],
        "resolved": resolved,
        "resolved_rate_full_scope": round(resolved / cfg["scoped_n"], 4),
        "evaluated_in_container": evaluated,
        "resolved_rate_evaluated_subset": round(resolved / evaluated, 4) if evaluated else None,
        "unresolved_in_container": r["unresolved_instances"],
        "disclosure": {
            "over_length_prompts_scored_unresolved": len(over_length),
            "generation_failed_or_unparseable_scored_unresolved": len(gen_failed),
            "patch_apply_failed_scored_unresolved": len(errors["patch_apply_failed"]),
            "harness_env_failed_scored_unresolved": len(errors["harness_env_failed"]),
            "over_length_ids": over_length,
            "generation_failed_ids": gen_failed,
            "patch_apply_failed_ids": errors["patch_apply_failed"],
            "harness_env_failed_ids": errors["harness_env_failed"],
        },
        "resolved_ids": sorted(r["resolved_ids"]),
        "harness_report_file": str(cfg["harness_report"].relative_to(ROOT)),
        "per_instance_fail_to_pass_pass_to_pass": f"logs/run_evaluation/{cfg['run_id']}/qwen3-30b-wp-30_70-reasoning-merged-v4/<instance_id>/report.json (harness-produced, not re-derived)",
    }


def main():
    legs = {name: leg_summary(name, cfg) for name, cfg in LEGS.items()}
    # sanity: every scoped instance is accounted for exactly once per leg
    for name, cfg in LEGS.items():
        s = legs[name]
        d = s["disclosure"]
        accounted = (
            s["resolved"] + s["unresolved_in_container"]
            + d["over_length_prompts_scored_unresolved"]
            + d["generation_failed_or_unparseable_scored_unresolved"]
            + d["patch_apply_failed_scored_unresolved"]
            + d["harness_env_failed_scored_unresolved"]
        )
        assert accounted == cfg["scoped_n"], (name, accounted, cfg["scoped_n"])

    report = {
        "requirement": "BENCH-02",
        "title": "SWE-bench generation-mode eval at the pre-registered scope, native arm64 local Docker",
        "generated_utc": "2026-07-11",
        "pre_registration": {
            "file": "output/bench17/swebench_scope_preregistration.md",
            "committed_before_any_eval_result": True,
            "commit": "65116ed",
            "scope": "SWE-bench Lite 300 (primary) + SWE-bench-Multilingual PHP 43 (secondary), oracle retrieval, generation-mode (non-agentic), native arm64 local Docker eval, <=20h budget",
        },
        "mode": "generation (non-agentic: one prompt in, one unified-diff patch out; no agent scaffold)",
        "retrieval": "oracle",
        "arch": "arm64 (native, local Docker; make_test_spec(arch='arm64', namespace=None) via scripts/swebench_arm64_eval.py; patch application ONLY inside per-instance harness containers)",
        "swebench_version": "4.1.0",
        "model": {
            "path": "models/qwen3-30b-wp-30_70-reasoning-merged-v4",
            "role": "wp_gen (v1.2)",
            "model_name_or_path": "qwen3-30b-wp-30_70-reasoning-merged-v4",
        },
        "serving_config": GEN_RECEIPT["serving_config"],
        "sampling_config": {**GEN_RECEIPT["sampling_config"], "warmup": "real-generation gate (not /health)"},
        "over_length_handling": "Prompts exceeding max_model_len-2048 were submitted anyway; server context-length rejections scored unresolved and disclosed below (pre-registered; no silent exclusion, no post-hoc re-scoping).",
        "results": legs,
        "wall_clock": {
            "generation_s": GEN_RECEIPT["total_wall_clock_s"],
            "generation_h": GEN_RECEIPT["total_wall_clock_h"],
        },
        "notes": [
            "Primary resolved rates use the full pre-registered scope as denominator (Lite n=300, PHP n=43); every non-resolved disposition (over-length, unparseable generation, patch-apply failure, harness env-build failure) counts against the model, per the pre-registration.",
            "harness_env_failed instances are arm64/2026-toolchain gaps in the benchmark's own environment specs (conda packages unavailable on linux-aarch64: cdms2, py3.6-era setuptools/scipy; pip>=24 removed --no-use-pep517; PEP-660 editable-install gaps; sympy branch '1.7' deleted upstream). The model's patches for these instances were never tested; scoring them unresolved is conservative-against-the-model and disclosed rather than excluded.",
            "A first Lite pass reused 7 stale env images + 17 stale instance images from a Nov-2024 swebench install on this host (old layout bakes the repo into /testbed, colliding with 4.1.0's instance build). All stale images were deleted and every affected instance re-run; one contaminated completed instance (pytest-dev__pytest-7490) was also re-run from scratch.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report["results"].items()}, indent=2)[:2000])
    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()
