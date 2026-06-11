"""Rank x replay grid eval driver (D-N5/D-N6/D-N7/D-N8).

Per-candidate orchestrator that enforces the eval economy:
  capture + judge gates (cheap, on Tinker sampler, no merge) ->
  only-if-pass merge (merge_tinker_v3.py, MoE-only-aware) ->
  REVL-04 wp-bench (run_eval_reasoning.py --wpbench-only, ~2.7h/candidate).

Acceptance bars (resolved 2026-06-11, VALIDATION.md authoritative):
  Judge accept bar (D-N7): POINT Spearman >= 0.263 (v3 floor / do-no-harm).
    CI-lower is a noise-guard diagnostic recorded in both modes -- NOT the threshold.
    Configurable: --judge-bar-mode point|ci_lower (default point), --judge-bar 0.263.
  Codegen accept bar (D-N8): REVL-04 wp-bench POINT >= 0.4537 (full 344-test) -- HARD.
    No candidate clearing -> escalation (exit 2), not auto-ship.

Pre-registered selection rule: max wp-bench s.t. judge POINT Spearman >= 0.263
  AND REVL-05 gates pass; tie-break higher judge Spearman.

Exit codes: 0 = winner selected; 1 = gate/usage failure; 2 = escalation (no candidate clears D-N8).

Pure importable decide() for unit testing (Task 3 synthetic fixture):
  decide(results, wpbench_baseline) -> (winner_or_None, exit_code)
  No Tinker / merge / wp-bench / subprocess / file-IO side effects.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Pure decision function — importable with zero side effects (T-04.3-08 / Task 3)
# ---------------------------------------------------------------------------

def decide(results: list[dict], wpbench_baseline: float) -> tuple:
    """Pre-registered selection rule + escalation (D-N8).

    Parameters
    ----------
    results : list[dict]
        Per-candidate summary dicts (same shape written to summary.json):
          {"candidate": str, "gates": {
              "judge_spearman": {"rho": float, "pass": bool, ...},
              "sentinel_0_policy": {"pass": bool, ...},
              "confusion_gate": {"pass": bool, ...},
              "fs_gate": {"pass": bool, ...}
          }, "wpbench_score": float|None, "selected": bool}
        A judge-failing candidate has wpbench_score=None (was never wp-benched).
    wpbench_baseline : float
        HARD gate threshold (D-N8). Default 0.4537 (full 344-test baseline).

    Returns
    -------
    (winner_or_None, exit_code)
        exit_code 0  -> winner selected (winner["selected"] set to True in-place)
        exit_code 2  -> escalation: no candidate cleared D-N8 (no auto-ship)
    """
    # A candidate is in "passing" only if:
    #   - wpbench_score is not None (was wp-benched, meaning it passed all pre-merge gates)
    #   - wpbench_score >= wpbench_baseline (HARD gate D-N8)
    #   - all four judge/REVL-05 gates pass
    passing = [
        c for c in results
        if c["wpbench_score"] is not None
        and c["wpbench_score"] >= wpbench_baseline
        and all(c["gates"][g]["pass"] for g in (
            "judge_spearman", "sentinel_0_policy", "confusion_gate", "fs_gate"
        ))
    ]

    if not passing:
        # D-N8 escalation: no winner auto-selected
        return (None, 2)

    # Pre-registered selection rule: max wp-bench; tie-break higher judge Spearman
    winner = max(
        passing,
        key=lambda c: (c["wpbench_score"], c["gates"]["judge_spearman"]["rho"])
    )
    winner["selected"] = True
    return (winner, 0)


# ---------------------------------------------------------------------------
# Sampler path resolution (PATTERNS lines 362-369)
# ---------------------------------------------------------------------------

def _resolve_tinker_path(tinker_path: str | None, manifest_path: str) -> str:
    """Resolve the Tinker sampler path from CLI override or manifest promoted entry."""
    if tinker_path:
        return tinker_path
    with open(manifest_path) as f:
        m = json.load(f)
    promoted = m.get("promoted")
    for c in m.get("checkpoints", []):
        if c.get("name") == promoted:
            return c["sampler_path"]
    raise SystemExit(
        f"could not resolve sampler_path for promoted={promoted!r} in {manifest_path}"
    )


# ---------------------------------------------------------------------------
# Per-candidate result loading / terminal-verdict check (resumability)
# ---------------------------------------------------------------------------

def _load_terminal(summary_path: Path) -> dict | None:
    """Return the loaded summary dict if it records a terminal verdict, else None.

    Terminal = pre_merge gates recorded (all four gate keys present) AND
    either wpbench_score is not None OR pre_merge_pass was False (gates dict
    shows at least one gate did not pass).
    """
    if not summary_path.is_file():
        return None
    try:
        s = json.loads(summary_path.read_text())
    except Exception:
        return None
    gates = s.get("gates", {})
    required_keys = {"judge_spearman", "sentinel_0_policy", "confusion_gate", "fs_gate"}
    if not required_keys.issubset(gates.keys()):
        # Incomplete summary — not terminal yet
        return None
    # Terminal if: wpbench_score is set (post-merge), OR pre_merge_pass was False (skipped merge)
    wpbench_score = s.get("wpbench_score")
    pre_merge_pass = all(gates[k]["pass"] for k in required_keys)
    if wpbench_score is not None or not pre_merge_pass:
        return s
    return None


# ---------------------------------------------------------------------------
# Per-candidate pre-merge judge gate execution
# ---------------------------------------------------------------------------

def _run_candidate_judge_gates(
    candidate_tag: str,
    adapter_tar: str,
    sampler_path: str,
    candidate_out: Path,
    dataset: str,
    args,
) -> dict:
    """Run the four cheap pre-merge gates and return the gate dict.

    Runs (in order):
      1. capture_judge_responses_tinker.py  (Tinker sampler offline capture)
      2. eval/eval_judge.py --responses-jsonl  (Spearman point + ci_lower)
      3. check_invalid_php_sentinel.py  (0/24 policy false-pass)
      4. check_verdict_confusion.py  (Pareto vs v3)
      5. tinker_fs_gate.py  (Wilson-upper)
    """
    candidate_out.mkdir(parents=True, exist_ok=True)
    responses_jsonl = candidate_out / "captured_responses.jsonl"
    judge_results_json = candidate_out / "judge_results.json"
    sentinel_json = candidate_out / "sentinel_result.json"
    confusion_json = candidate_out / "confusion_result.json"
    fs_json = candidate_out / "fs_result.json"

    sentinel_responses = candidate_out / "sentinel_responses.jsonl"

    def _capture(dataset_path: str, out_path: Path, label: str):
        # capture_judge_responses_tinker.py imports `tinker` -> .venv-tinker (args.tinker_python),
        # not the project venv. RESUMABLE: reuse a non-empty capture so a mid-grid crash never
        # re-spends Tinker sampling (--force overrides).
        if out_path.exists() and out_path.stat().st_size > 0 and not args.force:
            print(f"[{candidate_tag}] reuse existing {label} capture ({out_path})", flush=True)
            return
        print(f"[{candidate_tag}] capture {label} responses from Tinker sampler...", flush=True)
        rc = subprocess.call([
            args.tinker_python,
            str(PROJECT_ROOT / "scripts" / "capture_judge_responses_tinker.py"),
            "--tinker-path", sampler_path,
            "--dataset", dataset_path,
            "--out", str(out_path),
        ])
        if rc != 0:
            raise RuntimeError(f"[{candidate_tag}] capture ({label}) failed (rc={rc})")

    # Step 1: capture judge responses on the val set (judge Spearman + confusion both reuse this)
    _capture(dataset, responses_jsonl, "judge-val")

    # Step 2: eval_judge --responses-jsonl -> REVL-01A POINT Spearman (calibrated_canonical GT).
    # Flags: --gt-mode calibrated_canonical (required for offline mode), --output (NOT --out:
    # ambiguous with --output-format), --output-format auto (captured judge output is prose-or-json).
    # Reuse a prior judge_results.json so a driver re-run for merge+wp-bench trusts the
    # authoritative judge verdict from the filter pass. eval_judge's calibrated_canonical GT is
    # mildly nondeterministic (rho wobbles ~0.02-0.05 on the SAME capture); re-running risks an
    # unlucky draw flipping a recorded pass. --force re-evaluates.
    if judge_results_json.exists() and judge_results_json.stat().st_size > 0 and not args.force:
        print(f"[{candidate_tag}] reuse existing judge_results ({judge_results_json})", flush=True)
    else:
        print(f"[{candidate_tag}] eval_judge --responses-jsonl ...", flush=True)
        rc = subprocess.call([
            sys.executable, "-m", "eval.eval_judge",
            "--responses-jsonl", str(responses_jsonl),
            "--gt-mode", "calibrated_canonical",
            "--output-format", "auto",
            "--output", str(judge_results_json),
            "--dataset", dataset,
        ], cwd=str(PROJECT_ROOT))
        if rc != 0:
            raise RuntimeError(f"[{candidate_tag}] eval_judge failed (rc={rc})")
    judge_data = json.loads(judge_results_json.read_text())
    revl = judge_data.get("revl01a_overall_spearman_HARD", {}) or {}
    spearman_point = revl.get("corr", 0.0) or 0.0
    ci_lower = revl.get("ci_lower", 0.0) or 0.0   # absent for point mode -> diagnostic only

    # Step 3: invalid-PHP sentinel gate. The sentinel prompts (24 held-out invalid-PHP cases) are
    # NOT in the judge-val capture -> a SEPARATE capture of the sentinel dataset is required.
    # Output key is policy_false_pass; flag is --output; exits 0 (all pass) / 1 (false-passes found).
    _capture(args.sentinel_dataset, sentinel_responses, "sentinel")
    print(f"[{candidate_tag}] check_invalid_php_sentinel ...", flush=True)
    rc = subprocess.call([
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "check_invalid_php_sentinel.py"),
        "--dataset", args.sentinel_dataset,
        "--responses-jsonl", str(sentinel_responses),
        "--output", str(sentinel_json),
    ])
    if rc not in (0, 2):  # check_invalid_php_sentinel: 0 = 0 policy false-pass (PASS), 2 = FAIL
        raise RuntimeError(f"[{candidate_tag}] check_invalid_php_sentinel error (rc={rc})")
    sentinel_data = json.loads(sentinel_json.read_text())
    sentinel_false_passes = sentinel_data.get("policy_false_pass", 1)

    # Step 4: verdict confusion gate (reuses the judge-val capture). Output rates are
    # policy.{false_FAIL_on_teacherPASS, recall_on_teacherFAIL} as [hit, tot, rate] lists.
    print(f"[{candidate_tag}] check_verdict_confusion ...", flush=True)
    rc = subprocess.call([
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "check_verdict_confusion.py"),
        "--dataset", dataset,
        "--responses-jsonl", str(responses_jsonl),
        "--output", str(confusion_json),
    ])
    if rc not in (0, 1):
        raise RuntimeError(f"[{candidate_tag}] check_verdict_confusion error (rc={rc})")
    confusion_data = json.loads(confusion_json.read_text())

    def _rate(v):
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            return float(v[2])
        return float(v) if isinstance(v, (int, float)) else float("nan")

    _pol = confusion_data.get("policy", {})
    conf_false_fail = _rate(_pol.get("false_FAIL_on_teacherPASS"))
    conf_recall = _rate(_pol.get("recall_on_teacherFAIL"))
    # Pareto (VERDICT-POLICY.md): recall on teacher-FAIL rises vs P4 (>= 0.638) AND false-FAIL on
    # teacher-PASS not materially worse than P4 (<= 0.403). Both axes vs the P4 policy baseline.
    pareto_ok = (conf_recall >= 0.638) and (conf_false_fail <= 0.403)

    # Build the three cheap gates first so we can short-circuit the slow FS sampling.
    gates: dict[str, dict] = {}
    judge_bar_mode = args.judge_bar_mode
    judge_bar = args.judge_bar
    judge_pass = (spearman_point >= judge_bar) if judge_bar_mode == "point" else (ci_lower >= judge_bar)
    gates["judge_spearman"] = {
        "rho": spearman_point, "ci_lower": ci_lower, "bar": judge_bar,
        "mode": judge_bar_mode, "pass": judge_pass,
    }
    gates["sentinel_0_policy"] = {
        "policy_false_pass": sentinel_false_passes, "pass": sentinel_false_passes == 0,
    }
    gates["confusion_gate"] = {
        "false_FAIL_on_teacherPASS": conf_false_fail, "recall_on_teacherFAIL": conf_recall,
        "p4_false_fail_baseline": 0.403, "p4_recall_baseline": 0.638,
        "pareto_ok": pareto_ok, "pass": pareto_ok,
    }

    # Step 5: FS terse Wilson-upper gate (RTRN-05). tinker_fs_gate.py samples FRESH on the sampler
    # (cot+ctf scope, not the judge capture; the slowest gate) and imports tinker_cookbook ->
    # .venv-tinker. Top-level "pass" = (rate <= 0.10 AND wilson_upper <= 0.15) across arms;
    # exits 0=PASS / 2=FAIL. SHORT-CIRCUIT: pre_merge_pass needs all four gates, so if any cheaper
    # gate already failed, the FS sampling cannot change the outcome -> skip it (saves Tinker time
    # on doomed candidates). Correctness-neutral: a skipped candidate is already filtered.
    cheap_pass = judge_pass and (sentinel_false_passes == 0) and pareto_ok
    if cheap_pass:
        # Reuse a prior fs_result (the slowest, priciest gate: fresh 300-sample temp>0 arm) so a
        # driver re-run for merge+wp-bench does not re-pay ~40 min of Tinker sampling (--force overrides).
        if fs_json.exists() and fs_json.stat().st_size > 0 and not args.force:
            print(f"[{candidate_tag}] reuse existing fs_result ({fs_json})", flush=True)
            fs_data = json.loads(fs_json.read_text())
        else:
            print(f"[{candidate_tag}] tinker_fs_gate ...", flush=True)
            rc = subprocess.call([
                args.tinker_python,
                str(PROJECT_ROOT / "scripts" / "tinker_fs_gate.py"),
                "--tinker-path", sampler_path,
                "--dataset", dataset,
                "--gate-n", str(args.fs_gate_n),
                "--out", str(fs_json),
            ])
            if rc not in (0, 2):
                raise RuntimeError(f"[{candidate_tag}] tinker_fs_gate error (rc={rc})")
            fs_data = json.loads(fs_json.read_text())
        fs_pass = bool(fs_data.get("pass", False))
        wilson_upper = max((a.get("wilson_upper", 1.0) for a in fs_data.get("arms", [])), default=1.0)
        gates["fs_gate"] = {"wilson_upper": wilson_upper, "fs_pass": fs_pass, "pass": fs_pass}
    else:
        print(f"[{candidate_tag}] tinker_fs_gate SKIPPED (cheaper gate already failed; "
              f"judge={judge_pass} sentinel0={sentinel_false_passes == 0} pareto={pareto_ok})", flush=True)
        gates["fs_gate"] = {"wilson_upper": None, "fs_pass": None,
                            "pass": False, "skipped": "cheaper gate already failed"}

    return gates


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="rank×replay grid eval driver (D-N5)")
    ap.add_argument("--grid-manifest", required=True,
                    help="JSON file listing candidates: rank, replay_pct, adapter_tar, sampler_path")
    ap.add_argument("--out-dir", required=True,
                    help="Root output dir; per-candidate results go in <out-dir>/<candidate_tag>/")
    ap.add_argument("--judge-bar-mode", default="point", choices=["point", "ci_lower"],
                    help="D-N7 bar semantics: 'point' = rho >= bar (resolved intent); "
                         "'ci_lower' = CI-lower >= bar (requires n expansion). Default: point.")
    ap.add_argument("--judge-bar", type=float, default=0.263,
                    help="Spearman accept threshold (D-N7: point >= 0.263). Default: 0.263.")
    ap.add_argument("--wpbench-baseline", type=float, default=0.4537,
                    help="HARD gate baseline (D-N8; full 344-test). Default: 0.4537.")
    ap.add_argument("--skip-merge", action="store_true",
                    help="Run pre-merge judge gates only (no merge + wp-bench). "
                         "Useful for a cheap dry pass over all candidates.")
    ap.add_argument("--dataset",
                    default="data/reasoning_dataset/openai_val.jsonl",
                    help="Val dataset JSONL for judge capture. Default: openai_val.jsonl.")
    ap.add_argument("--base-model",
                    default="Qwen/Qwen3-30B-A3B",
                    help="Pinned stock base model for merge (D-N1). Default: Qwen/Qwen3-30B-A3B.")
    ap.add_argument("--gpu-mem-util", type=float, default=0.55,
                    help="vLLM GPU memory utilization for wp-bench. Default: 0.55.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run candidates even if summary.json already records a terminal verdict.")
    ap.add_argument("--tinker-python",
                    default=str(PROJECT_ROOT / ".venv-tinker" / "bin" / "python"),
                    help="Interpreter for the Tinker-dependent steps (capture + fs_gate import "
                         "tinker/tinker_cookbook, absent from the project venv). "
                         "Default: <root>/.venv-tinker/bin/python.")
    ap.add_argument("--sentinel-dataset",
                    default="data/reasoning_dataset/invalid_php_sentinel.jsonl",
                    help="Held-out invalid-PHP sentinel dataset (REVL-05); captured separately "
                         "from the judge-val set. Default: invalid_php_sentinel.jsonl.")
    ap.add_argument("--fs-gate-n", type=int, default=300,
                    help="tinker_fs_gate target sample count for the temp>0 Wilson-sizing arm. "
                         "Default: 300.")
    args = ap.parse_args()

    if not os.path.exists(args.tinker_python):
        print(f"ERROR: --tinker-python not found: {args.tinker_python} "
              f"(capture + fs_gate need the .venv-tinker interpreter)", file=sys.stderr, flush=True)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load grid manifest
    manifest_path = Path(args.grid_manifest)
    if not manifest_path.is_file():
        print(f"ERROR: grid manifest not found: {manifest_path}", file=sys.stderr, flush=True)
        return 1
    grid_manifest = json.loads(manifest_path.read_text())
    # Manifest may be a bare list (Plan 04 schema) or a dict with a "candidates" list.
    candidates = grid_manifest.get("candidates", []) if isinstance(grid_manifest, dict) else grid_manifest
    if not isinstance(candidates, list):
        print("ERROR: grid manifest must be a JSON object with 'candidates' list or a bare list.",
              file=sys.stderr, flush=True)
        return 1

    dataset = str(PROJECT_ROOT / args.dataset) if not os.path.isabs(args.dataset) else args.dataset

    results: list[dict] = []

    for cand in candidates:
        candidate_tag = cand.get("candidate_tag") or (
            f"r{cand.get('rank', '?')}_rp{cand.get('replay_pct', '?')}"
        )
        adapter_tar = cand.get("adapter_tar", "")
        sampler_path = cand.get("sampler_path", "")
        rank = cand.get("rank")
        replay_pct = cand.get("replay_pct")

        candidate_out = out_dir / candidate_tag
        summary_path = candidate_out / "summary.json"

        # Resumability: skip terminal candidates unless --force
        if not args.force:
            existing = _load_terminal(summary_path)
            if existing is not None:
                print(f"[{candidate_tag}] SKIP — terminal summary.json found (use --force to re-run)",
                      flush=True)
                results.append(existing)
                continue

        print(f"\n{'=' * 60}", flush=True)
        print(f"[{candidate_tag}] BEGIN (rank={rank}, replay_pct={replay_pct})", flush=True)

        # --- Pre-merge judge gates (cheap) ---
        try:
            gates = _run_candidate_judge_gates(
                candidate_tag=candidate_tag,
                adapter_tar=adapter_tar,
                sampler_path=sampler_path,
                candidate_out=candidate_out,
                dataset=dataset,
                args=args,
            )
        except RuntimeError as e:
            print(f"[{candidate_tag}] GATE ERROR: {e}", file=sys.stderr, flush=True)
            return 1

        pre_merge_pass = all(v["pass"] for v in gates.values())
        print(f"[{candidate_tag}] pre_merge_pass={pre_merge_pass}", flush=True)
        for k, v in gates.items():
            print(f"  {k}: {v}", flush=True)

        # Write partial summary with wpbench_score=None before merge decision
        wpbench_score: float | None = None
        summary = {
            "candidate": candidate_tag,
            "rank": rank,
            "replay_pct": replay_pct,
            "adapter_tar": adapter_tar,
            "gates": gates,
            "wpbench_score": wpbench_score,
            "selected": False,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))

        # --- Only-if-pass merge + REVL-04 wp-bench (expensive) ---
        if pre_merge_pass and not args.skip_merge:
            # Under _staging/ so merge_tinker_v3's canonical-overwrite guard passes without
            # --force-canonical (the guard checks for "_staging/" in the output path).
            staging_dir = out_dir / "_staging" / candidate_tag
            staging_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{candidate_tag}] merging adapter (MoE-only-aware)...", flush=True)
            rc = subprocess.call([
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "merge_tinker_v3.py"),
                "--adapter-tar", adapter_tar,
                "--base", args.base_model,
                "--output-dir", str(staging_dir),
            ])
            if rc != 0:
                print(f"[{candidate_tag}] merge_tinker_v3 FAILED (rc={rc})", file=sys.stderr, flush=True)
                return 1

            # REVL-04 wp-bench via run_eval_reasoning.py --wpbench-only
            # RC-A fix (enable_thinking=False, commit b88faa3) is active in that script.
            print(f"[{candidate_tag}] REVL-04 wp-bench (--wpbench-only)...", flush=True)
            wpbench_out = candidate_out / "wpbench"
            wpbench_out.mkdir(parents=True, exist_ok=True)
            wp_result_path = wpbench_out / "04.4_wp_bench_results.json"
            # Resumable: reuse a prior wp-bench result (the ~2.7h-per-model GPU step) so a driver
            # re-run does not re-bench (--force overrides).
            if wp_result_path.is_file() and not args.force:
                print(f"[{candidate_tag}] reuse existing wp-bench result ({wp_result_path})", flush=True)
                rc = 0
            else:
                # run_eval_reasoning --wpbench-only re-benches reasoning + baseline against an existing
                # summary.json (prior eval numbers). The grid runs the judge gate separately, so seed a
                # minimal stub summary.json to satisfy that precondition; the baseline re-bench yields a
                # faithful same-harness baseline (informational — the HARD gate uses --wpbench-baseline).
                summ = wpbench_out / "summary.json"
                if not summ.is_file():
                    summ.write_text(json.dumps({"baseline": {}, "reasoning": {}}, indent=2))
                rc = subprocess.call([
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_eval_reasoning.py"),
                    "--wpbench-only",
                    "--reasoning-model", str(staging_dir),
                    "--out-dir", str(wpbench_out),
                    "--gpu-mem-util", str(args.gpu_mem_util),
                ], cwd=str(PROJECT_ROOT))
            if rc != 0:
                print(f"[{candidate_tag}] run_eval_reasoning --wpbench-only FAILED (rc={rc})",
                      file=sys.stderr, flush=True)
                return 1

            # Parse the 344-test wp-bench point score from the result artifact
            wp_result_path = wpbench_out / "04.4_wp_bench_results.json"
            if wp_result_path.is_file():
                wp_data = json.loads(wp_result_path.read_text())
                wpbench_score = wp_data.get("reasoning_score")
            else:
                # Fallback: try summary.json wpbench_reasoning score
                wp_summary_path = wpbench_out / "summary.json"
                if wp_summary_path.is_file():
                    wp_summary = json.loads(wp_summary_path.read_text())
                    wpbench_score = (wp_summary.get("wpbench_reasoning") or {}).get("wpbench_score")

            print(f"[{candidate_tag}] wpbench_score={wpbench_score}", flush=True)

            # Update summary with wpbench_score (terminal state)
            summary["wpbench_score"] = wpbench_score
            summary_path.write_text(json.dumps(summary, indent=2))

        elif pre_merge_pass and args.skip_merge:
            print(f"[{candidate_tag}] --skip-merge: skipping merge + wp-bench", flush=True)
        else:
            print(f"[{candidate_tag}] SKIP merge+wpbench (pre_merge_pass=False)", flush=True)

        results.append(summary)

    # --- Pre-registered selection + escalation via pure decide() ---
    winner, exit_code = decide(results, args.wpbench_baseline)

    # Write grid_results.json regardless of outcome
    grid_results = {
        "candidates": results,
        "wpbench_baseline": args.wpbench_baseline,
        "winner": winner,
        "escalation": exit_code == 2,
    }
    grid_results_path = out_dir / "grid_results.json"
    grid_results_path.write_text(json.dumps(grid_results, indent=2))
    print(f"\ngrid_results.json -> {grid_results_path}", flush=True)

    if exit_code == 2:
        print(
            "ESCALATION: no grid candidate cleared wpbench >= 0.4537 AND all judge gates. "
            "Do NOT auto-promote. Human decision required.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    print(f"\nWINNER: {winner['candidate']} "
          f"(wpbench={winner['wpbench_score']}, "
          f"rho={winner['gates']['judge_spearman']['rho']})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
