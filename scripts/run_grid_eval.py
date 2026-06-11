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

    # Step 1: capture judge responses on the Tinker sampler
    print(f"[{candidate_tag}] capture judge responses from Tinker sampler...", flush=True)
    rc = subprocess.call([
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "capture_judge_responses_tinker.py"),
        "--tinker-path", sampler_path,
        "--dataset", dataset,
        "--out", str(responses_jsonl),
    ])
    if rc != 0:
        raise RuntimeError(f"[{candidate_tag}] capture_judge_responses_tinker failed (rc={rc})")

    # Step 2: eval_judge --responses-jsonl -> Spearman point + ci_lower
    print(f"[{candidate_tag}] eval_judge --responses-jsonl ...", flush=True)
    rc = subprocess.call([
        sys.executable, "-m", "eval.eval_judge",
        "--responses-jsonl", str(responses_jsonl),
        "--out", str(judge_results_json),
        "--dataset", dataset,
    ], cwd=str(PROJECT_ROOT))
    if rc != 0:
        raise RuntimeError(f"[{candidate_tag}] eval_judge failed (rc={rc})")

    judge_data = json.loads(judge_results_json.read_text())
    spearman_point = judge_data.get("revl01a_overall_spearman_HARD", {}).get("corr", 0.0)
    ci_lower = judge_data.get("revl01a_overall_spearman_HARD", {}).get("ci_lower", 0.0)

    # Step 3: invalid-PHP sentinel gate
    print(f"[{candidate_tag}] check_invalid_php_sentinel ...", flush=True)
    rc = subprocess.call([
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "check_invalid_php_sentinel.py"),
        "--responses-jsonl", str(responses_jsonl),
        "--out", str(sentinel_json),
    ])
    if rc not in (0, 1):  # 0 = all pass, 1 = some false-passes found (both are valid exits)
        raise RuntimeError(
            f"[{candidate_tag}] check_invalid_php_sentinel error (rc={rc})"
        )
    sentinel_data = json.loads(sentinel_json.read_text())
    sentinel_false_passes = sentinel_data.get("false_passes", 1)

    # Step 4: verdict confusion gate
    print(f"[{candidate_tag}] check_verdict_confusion ...", flush=True)
    rc = subprocess.call([
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "check_verdict_confusion.py"),
        "--responses-jsonl", str(responses_jsonl),
        "--out", str(confusion_json),
    ])
    if rc not in (0, 1):
        raise RuntimeError(
            f"[{candidate_tag}] check_verdict_confusion error (rc={rc})"
        )
    confusion_data = json.loads(confusion_json.read_text())
    pareto_ok = confusion_data.get("pareto_ok", False)

    # Step 5: FS terse Wilson-upper gate (RTRN-05)
    print(f"[{candidate_tag}] tinker_fs_gate ...", flush=True)
    rc = subprocess.call([
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "tinker_fs_gate.py"),
        "--responses-jsonl", str(responses_jsonl),
        "--out", str(fs_json),
    ])
    if rc not in (0, 1):
        raise RuntimeError(f"[{candidate_tag}] tinker_fs_gate error (rc={rc})")
    fs_data = json.loads(fs_json.read_text())
    wilson_upper = fs_data.get("wilson_upper", 1.0)

    # Build gate dict (PATTERNS lines 300-321)
    gates: dict[str, dict] = {}

    # D-N7: judge_spearman gate — point estimate vs bar (default mode); ci_lower recorded as diagnostic
    judge_bar_mode = args.judge_bar_mode
    judge_bar = args.judge_bar
    if judge_bar_mode == "point":
        judge_pass = spearman_point >= judge_bar
    else:  # ci_lower mode
        judge_pass = ci_lower >= judge_bar

    gates["judge_spearman"] = {
        "rho": spearman_point,
        "ci_lower": ci_lower,
        "bar": judge_bar,
        "mode": judge_bar_mode,
        "pass": judge_pass,
    }
    gates["sentinel_0_policy"] = {
        "false_passes": sentinel_false_passes,
        "pass": sentinel_false_passes == 0,
    }
    gates["confusion_gate"] = {
        "pareto_ok": pareto_ok,
        "pass": pareto_ok,
    }
    gates["fs_gate"] = {
        "wilson_upper": wilson_upper,
        "pass": wilson_upper <= 0.15,
    }

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
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load grid manifest
    manifest_path = Path(args.grid_manifest)
    if not manifest_path.is_file():
        print(f"ERROR: grid manifest not found: {manifest_path}", file=sys.stderr, flush=True)
        return 1
    grid_manifest = json.loads(manifest_path.read_text())
    candidates = grid_manifest.get("candidates", grid_manifest)
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
            staging_dir = out_dir / candidate_tag / "staging"
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
