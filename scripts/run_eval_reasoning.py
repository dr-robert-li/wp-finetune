"""W2-02: sequential reasoning-eval orchestrator (Phase 4.4 REVL-01/02/04).

Council two-GT / Option 3 / halt-on-regression. Steps:
  0. REVL-01 GT preflight (HARD) — abort if canonical GT not ready.
  1. D-03 baseline re-eval: serve merged-v2 -> eval_gen (PHPCS) + eval_judge
     (calibrated-canonical REVL-01A) -> record baseline -> stop vLLM.
  2. Reasoning eval: serve reasoning-merged -> eval_gen + eval_judge ->
     assert_served_identity (v3 fingerprint check) ->
     REVL-04 wp-bench (HARD, while served) -> stop vLLM.
  3. Gates (halt-on-regression):
     REVL-02 PHPCS within 2pp of baseline; REVL-01A overall Spearman >= baseline;
     REVL-04 wp-bench >= baseline. Any HARD fail -> failure summary, stop.

Merge already done + smoke-certified, so this VALIDATES (ship-vs-iterate), not
gates-the-merge. Outputs under output/eval_reasoning/.

Parameterized (plan 04.4-03): --reasoning-model, --baseline-model, --out-dir override
the module-level defaults so v3 results land in output/eval_reasoning_v3/ without
touching the stale ckpt-72 artifacts in output/eval_reasoning/.
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
from scripts.fidelity_gate_v3 import assert_served_identity, PreconditionError  # noqa: E402

# Module-level defaults (backward-compatible; overridden by CLI --reasoning-model /
# --baseline-model / --out-dir for the v3 run so stale ckpt-72 paths are never used).
BASELINE = "models/qwen3-30b-wp-30_70-merged-v2"
REASONING = "models/qwen3-30b-wp-30_70-reasoning-merged"
DATASET = "data/reasoning_dataset/openai_val.jsonl"
OUT_DIR = PROJECT_ROOT / "output" / "eval_reasoning"
WP_BENCH_DIR = PROJECT_ROOT / "wp-bench"
PHPCS_REGRESSION_PP = 0.02
PORT = 8021

# merge_report for the v3 staging model (used by assert_served_identity)
MERGE_REPORT_V3 = str(PROJECT_ROOT / "output" / "merge_v3" / "merge_report.json")


def _serve_and_eval(model_dir: str, name: str, tag: str, dataset: str,
                    limit, gpu_mem_util: float, out_dir: Path) -> dict:
    """Boot vLLM on model_dir, run eval_gen + eval_judge(calibrated-canonical), stop."""
    from eval import eval_gen, eval_judge
    endpoint = f"http://localhost:{PORT}/v1"
    os.environ["EVAL_GEN_BASE_URL"] = endpoint
    os.environ["EVAL_JUDGE_BASE_URL"] = endpoint
    out = out_dir / tag
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


def _run_wpbench(tag: str, out_dir: Path) -> dict:
    """REVL-04 wp-bench against the live endpoint. Requires wp-bench CLI + config."""
    out = out_dir / tag / "wp_bench_results.json"
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
            # wp-bench routes via litellm, which needs an explicit provider prefix.
            # vLLM serves an OpenAI-compatible API as model `wp-30_70`; `openai/` tells
            # litellm to use the OpenAI provider and send bare `wp-30_70` to api_base.
            # (Bare `wp-30_70` => "LLM Provider NOT provided" BadRequestError.)
            conf["models"][0]["name"] = "openai/wp-30_70"
        # wp_env_dir / dataset cache_dir in the base config are relative to PROJECT_ROOT,
        # but HarnessConfig.from_file() resolves relatives against the *config file's dir*.
        # We dump the tmp config into output/.../<tag>/, so rewrite relevant paths to
        # absolute before dumping or wp-env start fails with FileNotFoundError on the runtime.
        grader = conf.get("grader")
        if isinstance(grader, dict) and grader.get("wp_env_dir"):
            wed = Path(grader["wp_env_dir"])
            if not wed.is_absolute():
                wed = (PROJECT_ROOT / wed).resolve()
            grader["wp_env_dir"] = str(wed)
        conf.setdefault("output", {})["path"] = str(out)
        conf["output"]["jsonl_path"] = str(out.with_suffix(".jsonl"))
        tmp = out_dir / tag / "wp_bench_config_tmp.yaml"
        _yaml.dump(conf, open(tmp, "w"))
        # wp-bench's environment.py shells out via `npx wp-env ...`. Two host
        # quirks break that: (1) `wp-env` is a global bin, not a registry package,
        # so bare `npx wp-env` 404s/hangs — shadow npx with a repo shim that execs
        # the command directly; (2) node's IPv6-first happy-eyeballs intermittently
        # ETIMEDOUTs wp-env's config-read GitHub call — force IPv4-first.
        env = os.environ.copy()
        shim = str(PROJECT_ROOT / "scripts" / "_wpbench_shim")
        env["PATH"] = shim + os.pathsep + env.get("PATH", "")
        node_opts = env.get("NODE_OPTIONS", "").strip()
        env["NODE_OPTIONS"] = (node_opts + " --dns-result-order=ipv4first "
                               "--network-family-autoselection-attempt-timeout=2000").strip()
        # wp-bench's models.py calls litellm.completion() WITHOUT passing the
        # config's api_base/api_key, so route via litellm's OpenAI-provider env:
        #   OPENAI_API_BASE -> the live vLLM endpoint  (else hits real OpenAI)
        #   OPENAI_API_KEY  -> any non-empty value     (vLLM ignores it)
        env.setdefault("OPENAI_API_KEY", "EMPTY")
        env["OPENAI_API_BASE"] = endpoint
        env["OPENAI_BASE_URL"] = endpoint
        # Strip Qwen3 <think> scaffold from model output before scoring (mirrors
        # eval_gen.py:60). usercustomize.py auto-loads when its dir is on PYTHONPATH.
        pth = str(PROJECT_ROOT / "scripts" / "_wpbench_pth")
        env["PYTHONPATH"] = pth + os.pathsep + env.get("PYTHONPATH", "")
        r = subprocess.run(["wp-bench", "run", "--config", str(tmp)], env=env,
                           capture_output=True, text=True, timeout=7200, cwd=str(WP_BENCH_DIR))
        full_log = out_dir / tag / "wp_bench_run.log"
        full_log.write_text(f"=== STDOUT ===\n{r.stdout}\n=== STDERR ===\n{r.stderr}\n")
        if r.returncode != 0:
            return {"wpbench_score": None, "error": f"exit {r.returncode}",
                    "detail": (r.stderr or r.stdout)[-1500:], "log": str(full_log), "ran": True}
        # wp-bench writes a TIMESTAMPED file (wp_bench_results_<ts>.json), not the
        # configured output.path. Pick the newest match; score is metadata.scores.overall.
        cands = sorted(out.parent.glob(out.stem + "_*.json"), key=lambda p: p.stat().st_mtime)
        score_file = cands[-1] if cands else (out if out.exists() else None)
        res = json.loads(score_file.read_text()) if score_file else {}
        scores = res.get("metadata", {}).get("scores", {})
        return {"wpbench_score": scores.get("overall"), "scores": scores,
                "results_file": str(score_file) if score_file else None, "ran": True}
    except FileNotFoundError:
        return {"wpbench_score": None, "error": "wp-bench CLI not on PATH "
                "(pip install -e wp-bench/python)", "ran": False}
    except Exception as e:  # noqa: BLE001
        return {"wpbench_score": None, "error": str(e)[:300], "ran": False}


def _wpbench_with_boot(model_dir: str, name: str, tag: str, gpu_mem_util: float,
                       out_dir: Path) -> dict:
    """Boot vLLM on model_dir, run REVL-04 wp-bench against the live endpoint, stop."""
    try:
        boot_vllm(model_dir, name, PORT, gpu_mem_util)
        wait_healthy(PORT, name)
        return _run_wpbench(tag, out_dir)
    except VllmBootTimeout as e:
        return {"wpbench_score": None, "error": f"boot: {e}", "ran": False}
    finally:
        stop_vllm(name)


def main() -> int:
    ap = argparse.ArgumentParser(description="W2-02 reasoning eval orchestrator")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--limit", type=int, default=None, help="None = full set")
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--skip-wpbench", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--wpbench-only", action="store_true",
                    help="Skip preflight + eval_gen/judge; only (re)run REVL-04 wp-bench on "
                         "both models, reusing eval numbers from the existing summary.json.")
    # Plan 04.4-03: parameterized model + output paths so v3 run is namespaced under
    # output/eval_reasoning_v3/ and never touches the stale ckpt-72 artifacts.
    ap.add_argument("--reasoning-model", default=REASONING,
                    help="Path (relative to PROJECT_ROOT or absolute) to the reasoning model. "
                         "Default: %(default)s (stale ckpt-72 canonical; override for v3).")
    ap.add_argument("--baseline-model", default=BASELINE,
                    help="Path to the baseline model. Default: %(default)s.")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory. Default: output/eval_reasoning (legacy canonical). "
                         "Override to e.g. output/eval_reasoning_v3 for the v3 run.")
    args = ap.parse_args()

    # Resolve effective paths from CLI overrides (no code path closes over module globals).
    reasoning_model = args.reasoning_model
    baseline_model = args.baseline_model
    out_dir: Path = (Path(args.out_dir) if args.out_dir else OUT_DIR)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    failure = out_dir / "04.4_failure_summary.md"
    if failure.exists():
        failure.unlink()

    # --wpbench-only: reuse prior eval numbers, re-run just REVL-04 on each model.
    if args.wpbench_only:
        prior_path = out_dir / "summary.json"
        if not prior_path.exists():
            failure.write_text("--wpbench-only requires an existing summary.json "
                               "(prior eval numbers). None found.\n")
            print("HALT: no prior summary.json for --wpbench-only.", file=sys.stderr)
            return 1
        prior = json.loads(prior_path.read_text())
        baseline_res = prior.get("baseline", {})
        reasoning_res = prior.get("reasoning", {})
        print("=== --wpbench-only: REVL-04 wp-bench on reasoning-merged + baseline ===",
              file=sys.stderr)
        wpbench_reasoning = ({} if args.skip_wpbench
                             else _wpbench_with_boot(reasoning_model, "wp-eval-reasoning-vllm",
                                                     "reasoning_merged", args.gpu_mem_util,
                                                     out_dir))
        wpbench_baseline = ({} if args.skip_wpbench
                            else _wpbench_with_boot(baseline_model, "wp-eval-baseline-vllm2",
                                                    "baseline_30_70", args.gpu_mem_util,
                                                    out_dir))
        return _finalize(baseline_res, reasoning_res, wpbench_baseline, wpbench_reasoning,
                         args, out_dir)

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
        baseline_res = _serve_and_eval(baseline_model, "wp-eval-baseline-vllm", "baseline_30_70",
                                       args.dataset, args.limit, args.gpu_mem_util, out_dir)
    except VllmBootTimeout as e:
        failure.write_text(f"Baseline vLLM boot failed: {e}\n")
        return 3

    # Step 2: reasoning eval
    print("=== Step 2: reasoning-merged eval ===", file=sys.stderr)
    try:
        boot_vllm(reasoning_model, "wp-eval-reasoning-vllm", PORT, args.gpu_mem_util)
        served = wait_healthy(PORT, "wp-eval-reasoning-vllm")
        endpoint = f"http://localhost:{PORT}/v1"
        os.environ["EVAL_GEN_BASE_URL"] = endpoint
        os.environ["EVAL_JUDGE_BASE_URL"] = endpoint
        from eval import eval_gen, eval_judge
        rout = out_dir / "reasoning_merged"
        rout.mkdir(parents=True, exist_ok=True)
        print("[reasoning] eval_gen ...", file=sys.stderr)
        rgen = eval_gen.run_eval(dataset_path=args.dataset, limit=args.limit,
                                 output_path=str(rout / "eval_gen_results.json"), base_url=endpoint)
        print("[reasoning] eval_judge (calibrated-canonical) ...", file=sys.stderr)
        rjud = eval_judge.run_eval(dataset_path=args.dataset, limit=args.limit,
                                   output_path=str(rout / "eval_judge_results.json"),
                                   base_url=endpoint, output_format="auto",
                                   gt_mode="calibrated_canonical")
        reasoning_res = {"tag": "reasoning_merged", "model": reasoning_model, "served": served,
                         "phpcs_pass_rate": rgen.get("phpcs_pass_rate"),
                         "gen_overall_mean": rgen.get("overall_mean"),
                         "revl01a": rjud.get("revl01a_overall_spearman_HARD", {}),
                         "revl01b": rjud.get("revl01b_overall_spearman_teacher_SOFT", {}),
                         "excluded": rjud.get("excluded", {})}
        # Assert served-model identity before wp-bench scoring (T-0443-01 mitigation).
        # serve_30_70_vllm.sh (used by boot_vllm) serves with --served-model-name wp-30_70;
        # the merge_report + on-disk shard count carry the real v3 fingerprint.
        if not args.skip_wpbench:
            print("[reasoning] asserting served-model identity before wp-bench ...",
                  file=sys.stderr)
            try:
                assert_served_identity(endpoint, merge_report_path=MERGE_REPORT_V3,
                                       staging_dir=str(PROJECT_ROOT / reasoning_model)
                                       if not reasoning_model.startswith("/")
                                       else reasoning_model,
                                       served_model_name="wp-30_70")
                print("[reasoning] served-identity OK — v3 fingerprint verified.", file=sys.stderr)
            except PreconditionError as exc:
                failure.write_text(
                    f"REVL-04 ABORTED: served-model identity check failed (stale model?):\n{exc}\n"
                )
                print(f"HALT: served-identity check failed: {exc}", file=sys.stderr)
                return 4
        wpbench_reasoning = ({} if args.skip_wpbench
                             else _run_wpbench("reasoning_merged", out_dir))
    except VllmBootTimeout as e:
        failure.write_text(f"Reasoning vLLM boot failed: {e}\n")
        return 3
    finally:
        stop_vllm("wp-eval-reasoning-vllm")

    # baseline wp-bench (separate boot) if not skipped
    wpbench_baseline = ({} if args.skip_wpbench
                        else _wpbench_with_boot(baseline_model, "wp-eval-baseline-vllm2",
                                                "baseline_30_70", args.gpu_mem_util, out_dir))

    return _finalize(baseline_res, reasoning_res, wpbench_baseline, wpbench_reasoning,
                     args, out_dir)


def _finalize(baseline: dict, reasoning: dict, wpbench_baseline: dict,
              wpbench_reasoning: dict, args, out_dir: Path) -> int:
    """Compute gates (halt-on-regression), write summary.json + the REVL-04 artifact."""
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
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # REVL-04 standalone artifact: written into the parameterized out_dir so the v3 run
    # lands in output/eval_reasoning_v3/04.4_wp_bench_results.json (not the legacy shared
    # output/04.4_wp_bench_results.json — T-0443-02 namespace isolation).
    wp_gate = gates.get("REVL-04_wpbench_HARD")
    if wp_gate is not None:
        b_wp = wp_gate.get("baseline")
        r_wp = wp_gate.get("reasoning")
        meets = bool(b_wp is not None and r_wp is not None and r_wp >= b_wp)
        artifact = {
            "gate": "REVL-04",
            "baseline_score": b_wp,
            "reasoning_score": r_wp,
            "meets_baseline": meets,
            "pass": wp_gate.get("pass"),
        }
        if wp_gate.get("note"):
            artifact["note"] = wp_gate["note"]
        (out_dir / "04.4_wp_bench_results.json").write_text(
            json.dumps(artifact, indent=2))

    hard_fails = [k for k, v in gates.items() if v.get("pass") is False]
    print("\n=== GATES ===", file=sys.stderr)
    for k, v in gates.items():
        print(f"  {k}: {v}", file=sys.stderr)
    if hard_fails:
        (out_dir / "04.4_failure_summary.md").write_text(
            "REVL gates FAILED: " + ", ".join(hard_fails) +
            f"\n\nSee {out_dir / 'summary.json'}\n")
        print(f"\nHALT: {hard_fails}.", file=sys.stderr)
        return 1
    print(f"\nALL GATES PASS (or manual-pending). summary: {out_dir / 'summary.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
