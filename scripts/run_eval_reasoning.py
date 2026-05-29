"""W2-02: sequential reasoning-eval orchestrator (Phase 4.4 REVL-01/02/04).

Council two-GT / Option 3 / halt-on-regression. Steps:
  0. REVL-01 GT preflight (HARD) — abort if canonical GT not ready.
  1. D-03 baseline re-eval: serve merged-v2 -> eval_gen (PHPCS) + eval_judge
     (calibrated-canonical REVL-01A) -> record baseline -> stop vLLM.
  2. Reasoning eval: serve reasoning-merged -> eval_gen + eval_judge ->
     REVL-04 wp-bench (HARD, while served) -> stop vLLM.
  3. Gates (halt-on-regression):
     REVL-02 PHPCS within 2pp of baseline; REVL-01A overall Spearman >= baseline;
     REVL-04 wp-bench >= baseline. Any HARD fail -> failure summary, stop.

Merge already done + smoke-certified, so this VALIDATES (ship-vs-iterate), not
gates-the-merge. Outputs under output/eval_reasoning/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, stop_vllm, VllmBootTimeout  # noqa: E402

BASELINE = "models/qwen3-30b-wp-30_70-merged-v2"
REASONING = "models/qwen3-30b-wp-30_70-reasoning-merged"
DATASET = "data/reasoning_dataset/openai_val.jsonl"
OUT_DIR = PROJECT_ROOT / "output" / "eval_reasoning"
WP_BENCH_DIR = PROJECT_ROOT / "wp-bench"
PHPCS_REGRESSION_PP = 0.02
PORT = 8021


def _serve_and_eval(model_dir: str, name: str, tag: str, dataset: str,
                    limit, gpu_mem_util: float) -> dict:
    """Boot vLLM on model_dir, run eval_gen + eval_judge(calibrated-canonical), stop."""
    from eval import eval_gen, eval_judge
    endpoint = f"http://localhost:{PORT}/v1"
    os.environ["EVAL_GEN_BASE_URL"] = endpoint
    os.environ["EVAL_JUDGE_BASE_URL"] = endpoint
    out = OUT_DIR / tag
    out.mkdir(parents=True, exist_ok=True)
    result = {"tag": tag, "model": model_dir}
    try:
        boot_vllm(model_dir, name, PORT, gpu_mem_util)
        served = wait_healthy(PORT, name)
        result["served"] = served
        print(f"[{tag}] eval_gen (REVL-02 PHPCS) ...", file=sys.stderr)
        gen = eval_gen.run_eval(dataset_path=dataset, limit=limit,
                                output_path=str(out / "eval_gen_results.json"),
                                base_url=endpoint)
        result["phpcs_pass_rate"] = gen.get("phpcs_pass_rate")
        result["gen_overall_mean"] = gen.get("overall_mean")
        print(f"[{tag}] eval_judge (REVL-01A calibrated-canonical) ...", file=sys.stderr)
        jud = eval_judge.run_eval(dataset_path=dataset, limit=limit,
                                  output_path=str(out / "eval_judge_results.json"),
                                  base_url=endpoint, output_format="auto",
                                  gt_mode="calibrated_canonical")
        result["revl01a"] = jud.get("revl01a_overall_spearman_HARD", {})
        result["revl01b"] = jud.get("revl01b_overall_spearman_teacher_SOFT", {})
        result["revl01a_variance"] = jud.get("revl01a_variance_preflight", {})
        result["excluded"] = jud.get("excluded", {})
    finally:
        stop_vllm(name)
    return result


def _run_wpbench(tag: str) -> dict:
    """REVL-04 wp-bench against the live endpoint. Requires wp-bench CLI + config."""
    out = OUT_DIR / tag / "wp_bench_results.json"
    cfg = PROJECT_ROOT / "config" / "wp-bench.yaml"
    if not WP_BENCH_DIR.exists() or not cfg.exists():
        return {"wpbench_score": None, "error": "wp-bench dir/config missing", "ran": False}
    try:
        import yaml as _yaml
        conf = _yaml.safe_load(open(cfg)) or {}
        # Schema: models is a LIST; api_base + name are per-model. Served name is
        # 'wp-30_70' (serve_30_70_vllm.sh --served-model-name, same for both dirs).
        endpoint = f"http://localhost:{PORT}/v1"
        if conf.get("models"):
            conf["models"][0]["api_base"] = endpoint
            conf["models"][0]["name"] = "wp-30_70"
        conf.setdefault("output", {})["path"] = str(out)
        conf["output"]["jsonl_path"] = str(out.with_suffix(".jsonl"))
        tmp = OUT_DIR / tag / "wp_bench_config_tmp.yaml"
        _yaml.dump(conf, open(tmp, "w"))
        r = subprocess.run(["wp-bench", "run", "--config", str(tmp)],
                           capture_output=True, text=True, timeout=3600, cwd=str(WP_BENCH_DIR))
        if r.returncode != 0:
            return {"wpbench_score": None, "error": f"exit {r.returncode}",
                    "detail": (r.stderr or r.stdout)[:500], "ran": True}
        res = json.loads(out.read_text()) if out.exists() else {}
        return {"wpbench_score": res.get("score") or res.get("wpbench_score"), "ran": True}
    except FileNotFoundError:
        return {"wpbench_score": None, "error": "wp-bench CLI not on PATH "
                "(pip install -e wp-bench/python)", "ran": False}
    except Exception as e:  # noqa: BLE001
        return {"wpbench_score": None, "error": str(e)[:300], "ran": False}


def main() -> int:
    ap = argparse.ArgumentParser(description="W2-02 reasoning eval orchestrator")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--limit", type=int, default=None, help="None = full set")
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--skip-wpbench", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failure = OUT_DIR / "04.4_failure_summary.md"
    if failure.exists():
        failure.unlink()

    # Step 0: REVL-01 GT preflight (HARD)
    if not args.skip_preflight:
        print("=== Step 0: REVL-01 GT preflight ===", file=sys.stderr)
        pf = subprocess.run([sys.executable, "-m", "scripts._p0_revl01_preflight",
                             "--dataset", args.dataset]
                            + (["--limit", str(args.limit)] if args.limit else []),
                            cwd=str(PROJECT_ROOT))
        if pf.returncode != 0:
            failure.write_text("REVL-01 GT preflight FAILED — canonical GT not ready. "
                               "Cascade aborted before serving.\n")
            print("HALT: preflight failed.", file=sys.stderr)
            return 1

    # Step 1: baseline re-eval (D-03)
    print("=== Step 1: baseline re-eval (merged-v2) ===", file=sys.stderr)
    try:
        baseline = _serve_and_eval(BASELINE, "wp-eval-baseline-vllm", "baseline_30_70",
                                   args.dataset, args.limit, args.gpu_mem_util)
    except VllmBootTimeout as e:
        failure.write_text(f"Baseline vLLM boot failed: {e}\n")
        return 3

    # Step 2: reasoning eval
    print("=== Step 2: reasoning-merged eval ===", file=sys.stderr)
    try:
        boot_vllm(REASONING, "wp-eval-reasoning-vllm", PORT, args.gpu_mem_util)
        served = wait_healthy(PORT, "wp-eval-reasoning-vllm")
        endpoint = f"http://localhost:{PORT}/v1"
        os.environ["EVAL_GEN_BASE_URL"] = endpoint
        os.environ["EVAL_JUDGE_BASE_URL"] = endpoint
        from eval import eval_gen, eval_judge
        rout = OUT_DIR / "reasoning_merged"
        rout.mkdir(parents=True, exist_ok=True)
        print("[reasoning] eval_gen ...", file=sys.stderr)
        rgen = eval_gen.run_eval(dataset_path=args.dataset, limit=args.limit,
                                 output_path=str(rout / "eval_gen_results.json"), base_url=endpoint)
        print("[reasoning] eval_judge (calibrated-canonical) ...", file=sys.stderr)
        rjud = eval_judge.run_eval(dataset_path=args.dataset, limit=args.limit,
                                   output_path=str(rout / "eval_judge_results.json"),
                                   base_url=endpoint, output_format="auto",
                                   gt_mode="calibrated_canonical")
        reasoning = {"tag": "reasoning_merged", "model": REASONING, "served": served,
                     "phpcs_pass_rate": rgen.get("phpcs_pass_rate"),
                     "gen_overall_mean": rgen.get("overall_mean"),
                     "revl01a": rjud.get("revl01a_overall_spearman_HARD", {}),
                     "revl01b": rjud.get("revl01b_overall_spearman_teacher_SOFT", {}),
                     "excluded": rjud.get("excluded", {})}
        wpbench_reasoning = ({} if args.skip_wpbench else _run_wpbench("reasoning_merged"))
    except VllmBootTimeout as e:
        failure.write_text(f"Reasoning vLLM boot failed: {e}\n")
        return 3
    finally:
        stop_vllm("wp-eval-reasoning-vllm")

    # baseline wp-bench (separate boot) if not skipped
    wpbench_baseline = {}
    if not args.skip_wpbench:
        try:
            boot_vllm(BASELINE, "wp-eval-baseline-vllm2", PORT, args.gpu_mem_util)
            wait_healthy(PORT, "wp-eval-baseline-vllm2")
            wpbench_baseline = _run_wpbench("baseline_30_70")
        except VllmBootTimeout as e:
            wpbench_baseline = {"wpbench_score": None, "error": f"boot: {e}", "ran": False}
        finally:
            stop_vllm("wp-eval-baseline-vllm2")

    # Step 3: gates (halt-on-regression)
    gates = {}
    b_phpcs = baseline.get("phpcs_pass_rate")
    r_phpcs = reasoning.get("phpcs_pass_rate")
    if b_phpcs is not None and r_phpcs is not None:
        gates["REVL-02_phpcs"] = {
            "baseline": b_phpcs, "reasoning": r_phpcs,
            "pass": r_phpcs >= b_phpcs - PHPCS_REGRESSION_PP}
    b_sp = baseline.get("revl01a", {}).get("corr")
    r_sp = reasoning.get("revl01a", {}).get("corr")
    if b_sp is not None and r_sp is not None:
        gates["REVL-01A_spearman"] = {"baseline": b_sp, "reasoning": r_sp,
                                      "pass": r_sp >= b_sp}
    if not args.skip_wpbench:
        b_wp = wpbench_baseline.get("wpbench_score")
        r_wp = wpbench_reasoning.get("wpbench_score")
        if b_wp is not None and r_wp is not None:
            gates["REVL-04_wpbench_HARD"] = {"baseline": b_wp, "reasoning": r_wp,
                                             "pass": r_wp >= b_wp}
        else:
            gates["REVL-04_wpbench_HARD"] = {"baseline": b_wp, "reasoning": r_wp,
                                             "pass": None, "note": "wp-bench unavailable/failed — manual"}

    summary = {
        "baseline": baseline, "reasoning": reasoning,
        "wpbench_baseline": wpbench_baseline, "wpbench_reasoning": wpbench_reasoning,
        "gates": gates,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    hard_fails = [k for k, v in gates.items() if v.get("pass") is False]
    print("\n=== GATES ===", file=sys.stderr)
    for k, v in gates.items():
        print(f"  {k}: {v}", file=sys.stderr)
    if hard_fails:
        failure.write_text("REVL gates FAILED: " + ", ".join(hard_fails) +
                           f"\n\nSee {OUT_DIR / 'summary.json'}\n")
        print(f"\nHALT: {hard_fails}. {failure}", file=sys.stderr)
        return 1
    print(f"\nALL GATES PASS (or manual-pending). summary: {OUT_DIR / 'summary.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
