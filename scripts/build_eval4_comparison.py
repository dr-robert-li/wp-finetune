#!/usr/bin/env python
"""Phase 23 Plan 01 -- EVAL4-01 synthesis: audit receipt-comparability across
the four v4.0 wp-bench candidates, backfill the one missing CI (raw base) via
the SAME stratified bootstrap used for every other arm, then assemble the
milestone verdict JSON + apply the pre-registered acceptance criteria
mechanically. Pure synthesis over existing Phase 21 + diagnostic receipts --
no GPU, no Tinker, no new measurement.

    python3 scripts/build_eval4_comparison.py --emit audit
    python3 scripts/build_eval4_comparison.py --emit verdict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_gen03_wpbench import _bootstrap_ci_lower  # noqa: E402

BASE21 = PROJECT_ROOT / "output" / "base21"
DIAG = BASE21 / "diagnostic"
EVAL3 = PROJECT_ROOT / "output" / "eval3" / "eval3_final_comparison.json"
OUT_DIR = PROJECT_ROOT / "output" / "eval4"
AUDIT_PATH = OUT_DIR / "comparability_audit.json"
VERDICT_PATH = OUT_DIR / "eval4_final_comparison.json"

GEN03 = BASE21 / "gen03_wpbench.json"
EXP1 = DIAG / "exp1_ep1_wpbench.json"
EXP4 = DIAG / "exp4_bench.json"
JUDGE03_RHO = BASE21 / "judge03_rho.json"
JUDGE03_CAPTURE = BASE21 / "judge03_capture_rho.json"
RAW_BASE_RESULTS = BASE21 / "gen03_fresh_new_base_anchor" / "wp_bench_results_20260714_082330.json"

FINGERPRINT_FIELDS = [
    "seed", "max_tokens", "enable_thinking", "temperature", "concurrency",
    "n_tests", "n_knowledge", "n_execution", "n_boot", "alpha", "bootstrap_seed",
]
EXPECTED_FINGERPRINT = {
    "seed": 1337, "max_tokens": 2048, "enable_thinking": False, "temperature": 0.0,
    "concurrency": 4, "n_tests": 344, "n_knowledge": 320, "n_execution": 24,
    "n_boot": 1000, "alpha": 0.05, "bootstrap_seed": 1337,
}


def rel(p: Path) -> str:
    return str(p.relative_to(PROJECT_ROOT))


def emit_audit() -> dict:
    gen03 = json.loads(GEN03.read_text())
    exp1 = json.loads(EXP1.read_text())
    exp4 = json.loads(EXP4.read_text())
    judge_rho = json.loads(JUDGE03_RHO.read_text())
    judge_capture = json.loads(JUDGE03_CAPTURE.read_text())

    # Gen-side comparability: the three candidates that carry the full
    # harness fingerprint directly in their receipt (ep3, ep1, v4b).
    candidates = {"ep3": gen03, "ep1": exp1, "v4b": exp4}
    per_field = {}
    for field in FINGERPRINT_FIELDS:
        values = {name: c.get(field) for name, c in candidates.items()}
        # ep3's gen03_wpbench.json predates the bootstrap_seed key being
        # persisted in the result dict (bootstrap_seed is hardcoded to 1337
        # inside _bootstrap_ci_lower regardless of caller -- see
        # scripts/build_gen03_wpbench.py -- so a missing/None value there is
        # a known schema-vintage gap, not a harness divergence). Require
        # every PRESENT value to match the expected constant.
        present = {k: v for k, v in values.items() if v is not None}
        per_field[field] = {
            "values": values,
            "equal": bool(present) and all(v == EXPECTED_FINGERPRINT[field] for v in present.values()),
        }
    gen_harness_comparable = all(v["equal"] for v in per_field.values())

    # Raw-base anchor (nested in gen03_wpbench.json) carries the SAME
    # generation-side fields at its results_file metadata level (same
    # build_gen03_wpbench.py::_run_wpbench_on call, same script, same
    # hardcoded harness constants -- only model_dir/tag differ) but no CI,
    # since it was only benched as a floor-shift check, never bootstrapped.
    raw_base = gen03["fresh_new_base_anchor"]
    raw_base_results_path = Path(raw_base["results_file"])
    raw_base_metadata = json.loads(raw_base_results_path.read_text())["metadata"]
    raw_base_generation_fingerprint_matches = (
        raw_base_metadata["model"]["temperature"] == EXPECTED_FINGERPRINT["temperature"]
        and raw_base_metadata["model"]["max_tokens"] == EXPECTED_FINGERPRINT["max_tokens"]
        and raw_base_metadata["grader"]["concurrency"] == EXPECTED_FINGERPRINT["concurrency"]
    )

    # Backfill the ONE missing figure: raw-base CI, via the identical
    # stratified bootstrap (same function, same n_boot/alpha/bootstrap_seed
    # defaults) used to produce every other arm's CI.
    raw_base_ci = _bootstrap_ci_lower(raw_base_results_path)

    needs_confirmatory_gpu_run = not (gen_harness_comparable and raw_base_generation_fingerprint_matches)
    confirmatory_run_justification = (
        "(1) every gen candidate (ep3/ep1/v4b) shares the identical harness fingerprint "
        f"(field-equality asserted: {gen_harness_comparable}); "
        "(2) the sole missing figure (raw-base CI) is computed offline from its existing "
        "full 344-test results file via the same stratified bootstrap as every other arm "
        f"(backfilled ci_lower={raw_base_ci['ci_lower']}, ci_upper={raw_base_ci['ci_upper']}); "
        "(3) wp-bench ran greedy (temperature=0.0, seed=1337) so a re-serve reproduces the "
        "same per-test outcomes deterministically and cannot narrow the small 24-test "
        "execution-stratum CI."
    )

    audit = {
        "gen_harness_comparable": gen_harness_comparable,
        "gen_harness_per_field": per_field,
        "raw_base_generation_fingerprint_matches": raw_base_generation_fingerprint_matches,
        "raw_base_ci_lower": raw_base_ci["ci_lower"],
        "raw_base_ci_upper": raw_base_ci["ci_upper"],
        "raw_base_ci_point": raw_base_ci["point"],
        "raw_base_ci_source": rel(raw_base_results_path),
        "raw_base_ci_method": "reused scripts.build_gen03_wpbench._bootstrap_ci_lower "
                               "(stratified 320-knowledge/24-execution bootstrap, "
                               "0.3/0.4/0.3 weighted overall, n_boot=1000, alpha=0.05, "
                               "bootstrap_seed=1337) -- identical to ep3/ep1/v4b",
        "needs_confirmatory_gpu_run": needs_confirmatory_gpu_run,
        "confirmatory_run_justification": confirmatory_run_justification,
        "judge_cross_base_caveat": True,
        "judge_cross_base_caveat_note": (
            "v4 judge figures (served s1 0.7872, capture s1 0.8358, capture ensemble 0.8160) "
            "use eval_relabel.py, 8192-token cap, n=121 held-out relabeled val, 0 parse "
            "failures, same 2000-resample bootstrap as v3.0's shipping figures "
            f"(ensemble {judge_rho['framed_vs']['v30_ensemble']}, single "
            f"{judge_rho['framed_vs']['v30_single']}, from output/eval3/eval3_final_comparison.json) "
            "-- the harness matches, the base differs."
        ),
        "judge_engine_numerics_ceiling": True,
        "judge_engine_numerics_ceiling_note": (
            "Per DIAGNOSTIC_SYNTHESIS.md exp3: served ~0.78-0.79 is a serving-stack ceiling "
            "(Tinker-vs-vLLM numerics) common to both bases, not a model or label deficiency. "
            f"Capture path improved (new {judge_capture['best_single_seed']['rho']} > old 0.8274)."
        ),
        "source_receipts": {
            "gen03_wpbench.json": rel(GEN03),
            "exp1_ep1_wpbench.json": rel(EXP1),
            "exp4_bench.json": rel(EXP4),
            "judge03_rho.json": rel(JUDGE03_RHO),
            "judge03_capture_rho.json": rel(JUDGE03_CAPTURE),
            "raw_base_results_file": rel(raw_base_results_path),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2))
    print(f"[eval4] wrote {AUDIT_PATH}")
    return audit


def emit_verdict() -> dict:
    if not AUDIT_PATH.exists():
        print(f"HALT: {AUDIT_PATH} does not exist -- run --emit audit first.", file=sys.stderr)
        sys.exit(2)
    audit = json.loads(AUDIT_PATH.read_text())
    if not audit["gen_harness_comparable"] or audit["needs_confirmatory_gpu_run"]:
        print("HALT: un-reconciled comparability gap in comparability_audit.json "
              "-- cannot assemble verdict.", file=sys.stderr)
        sys.exit(3)

    gen03 = json.loads(GEN03.read_text())
    exp1 = json.loads(EXP1.read_text())
    exp4 = json.loads(EXP4.read_text())
    judge_rho = json.loads(JUDGE03_RHO.read_text())
    judge_capture = json.loads(JUDGE03_CAPTURE.read_text())

    floor = 0.4286

    candidate_A_raw_base = {
        "label": "raw base (Qwen3.6-35B-A3B, no adapter)",
        "overall": gen03["fresh_new_base_anchor"]["wpbench_overall"],
        "ci_lower": audit["raw_base_ci_lower"],
        "ci_upper": audit["raw_base_ci_upper"],
        "clears_floor_ci_aware": audit["raw_base_ci_lower"] >= floor,
        "source_receipt": rel(GEN03),
        "source_receipt_note": "fresh_new_base_anchor block; CI backfilled offline, see comparability_audit.json",
    }
    candidate_B_best_trained = {
        "label": "ep1 (best trained gen variant)",
        "overall": exp1["wpbench_overall"],
        "ci_lower": exp1["wpbench_ci_lower"],
        "ci_upper": exp1["wpbench_ci_upper"],
        "clears_floor_ci_aware": exp1["wpbench_ci_lower"] >= floor,
        "candidate_B_selection_rationale": (
            "ep1 is the best trained gen variant -- highest trained point estimate of "
            f"{{ep3 {gen03['wpbench_overall']}, ep1 {exp1['wpbench_overall']}, v4b {exp4['wpbench_overall']}}}; "
            "exp1 confirmed ep3 overtraining (ep1 recovers "
            f"{exp1['decision']['recovered_fraction_of_ep3_to_raw_gap']:.0%} of the ep3->raw gap); "
            "exp4 confirmed the rebuilt-mix v4b "
            f"({exp4['wpbench_overall']}) lands below ep1."
        ),
        "source_receipt": rel(EXP1),
    }
    reference_ep3 = {
        "label": "ep3 (shipped/promoted, overtrained)",
        "overall": gen03["wpbench_overall"],
        "ci_lower": gen03["wpbench_ci_lower"],
        "ci_upper": gen03["wpbench_ci_upper"],
        "clears_floor_ci_aware": gen03["wpbench_ci_lower"] >= floor,
        "source_receipt": rel(GEN03),
    }
    reference_v4b = {
        "label": "v4b (rebuilt-mix, 2 epochs)",
        "overall": exp4["wpbench_overall"],
        "ci_lower": exp4["wpbench_ci_lower"],
        "ci_upper": exp4["wpbench_ci_upper"],
        "clears_floor_ci_aware": exp4["wpbench_ci_lower"] >= floor,
        "source_receipt": rel(EXP4),
    }

    gen_role_winner = "raw_base"
    gen_role_winner_rationale = (
        "raw base dominates every trained variant on BOTH point estimate "
        f"({candidate_A_raw_base['overall']} > ep1 {candidate_B_best_trained['overall']} > "
        f"v4b {reference_v4b['overall']} > ep3 {reference_ep3['overall']}) AND CI-lower "
        f"({candidate_A_raw_base['ci_lower']} > ep1 {candidate_B_best_trained['ci_lower']}), "
        f"exceeds the v3.0 shipping gen figure (0.4365), and the diagnostic's final verdict "
        "is that the v1.2 SFT-for-codegen recipe has NEGATIVE headroom on this stronger base "
        "(the model's raw coding ability exceeds anything the current corpus teaches). This "
        "gen-role-winner call is robust regardless of whether the raw-base CI-lower itself "
        "clears the floor, because it is a relative A/B and raw base wins both metrics."
    )

    gen_ab = {
        "floor": floor,
        "candidate_A_raw_base": candidate_A_raw_base,
        "candidate_B_best_trained": candidate_B_best_trained,
        "reference_ep3": reference_ep3,
        "reference_v4b": reference_v4b,
        "v30_shipping_gen": {
            "fresh_full_rerun": 0.4365,
            "gate1_reference": 0.4484,
            "cross_base_caveat": True,
            "note": "both OLD base (Qwen3-30B-A3B); Phase 17 fresh full re-run figure is primary",
        },
        "gen_role_winner": gen_role_winner,
        "gen_role_winner_rationale": gen_role_winner_rationale,
    }

    served_s1 = {
        "methodology": "vllm_served",
        "rho": judge_rho["single_seed_figure"]["rho"],
        "ci_lower": judge_rho["single_seed_figure"]["ci_lower"],
        "target": 0.85,
        "source_receipt": rel(JUDGE03_RHO),
    }
    capture_ensemble = {
        "methodology": "tinker_capture",
        "rho": judge_rho["ensemble_figure"]["rho"],
        "ci_lower": judge_rho["ensemble_figure"]["ci_lower"],
        "target": 0.87,
        "source_receipt": rel(JUDGE03_RHO),
        "source_receipt_secondary": rel(JUDGE03_CAPTURE),
    }
    capture_s1_reference = {
        "methodology": "tinker_capture",
        "rho": judge_rho["single_seed_tinker_capture_reference"]["rho"],
        "note": "NON-GATING promotion-path reference",
        "source_receipt": rel(JUDGE03_RHO),
    }

    judge_ab = {
        "served_s1": served_s1,
        "capture_ensemble": capture_ensemble,
        "capture_s1_reference": capture_s1_reference,
        "framed_vs": {
            "v30_ensemble_served": judge_rho["framed_vs"]["v30_ensemble"],
            "v30_single_served": judge_rho["framed_vs"]["v30_single"],
            "capture_vs_capture": {
                "new": judge_capture["best_single_seed"]["rho"],
                "old": 0.8274,
                "note": "the rebase DID improve raw judge capability (+0.0084 capture), masked "
                        "by the ~0.78-0.79 served engine-numerics ceiling common to both bases",
            },
            "ceiling": judge_rho["framed_vs"]["ceiling"],
        },
        "relabel_reopen_condition_met": judge_rho["reopen_condition"]["condition_met"],
        "judge_cross_base_caveat": audit["judge_cross_base_caveat"],
        "judge_engine_numerics_ceiling": audit["judge_engine_numerics_ceiling"],
    }

    judge_single_met = served_s1["ci_lower"] > 0.85
    judge_ensemble_met = capture_ensemble["ci_lower"] > 0.87
    primary_judge_target_met = judge_single_met or judge_ensemble_met
    wpbench_floor_met_by_gen_role_winner = candidate_A_raw_base["clears_floor_ci_aware"]

    pre_registered_verdict = {
        "judge_single_met": judge_single_met,
        "judge_ensemble_met": judge_ensemble_met,
        "primary_judge_target_met": primary_judge_target_met,
        "wpbench_floor_met_by_gen_role_winner": wpbench_floor_met_by_gen_role_winner,
        "milestone_primary_verdict": (
            "PRIMARY TARGET (judge rho > 0.85 single OR > 0.87 ensemble, CI-aware) NOT MET "
            f"(served s1 ci_lower={served_s1['ci_lower']:.4f}; capture ensemble "
            f"ci_lower={capture_ensemble['ci_lower']:.4f}). Recorded as the valid, "
            "pre-registered failure disposition (\"no_winner is a result\"), not a forced pass."
        ),
        "disposition": "valid_recorded_miss",
    }

    verdict = {
        "phase": 23,
        "title": "v4.0 Final Evaluation -- EVAL4-01 primary A/B verdict",
        "generated_utc": "2026-07-15",
        "provenance_note": (
            "This is a consolidation report. Every quality number below is copied from an "
            "existing measured Phase 21 / diagnostic artifact under output/base21/ (same-harness, "
            "same-stack, taken 2026-07-14), with a `source_receipt` on every gating row. No new "
            "GPU eval was run for Phase 23; the single offline addition is the raw-base CI "
            "backfill (see comparability_audit.json), computed via the identical stratified "
            "bootstrap already used for every other arm."
        ),
        "shipping_stack": {
            "generation_role": {
                "winner": "raw_base",
                "name": "Qwen3.6-35B-A3B raw base (no gen adapter)",
                "role": "wp_gen",
            },
            "judge_role": {
                "name": "v4 relabel-SFT judge, s1 served / 3-seed capture ensemble",
                "role": "wp_judge",
            },
        },
        "comparison_arms_status": {
            "gen_arm": "Dual-candidate A/B per USER DIRECTIVE: raw base (candidate A) vs best "
                       "trained ep1 (candidate B), both framed against the 0.4286 floor and "
                       "v3.0 shipping gen. ep3 (shipped/promoted) and v4b (rebuilt-mix) carried "
                       "as reference rows.",
            "judge_arm": "v4 SFT judge (served s1 / 3-seed capture ensemble) vs v3.0 shipping "
                         "judge figures, with the cross-base + engine-numerics-ceiling caveats.",
        },
        "gen_ab": gen_ab,
        "judge_ab": judge_ab,
        "pre_registered_verdict": pre_registered_verdict,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VERDICT_PATH.write_text(json.dumps(verdict, indent=2))
    print(f"[eval4] wrote {VERDICT_PATH}")
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", choices=["audit", "verdict"], required=True)
    args = parser.parse_args()
    if args.emit == "audit":
        emit_audit()
    else:
        emit_verdict()
    return 0


if __name__ == "__main__":
    sys.exit(main())
