"""Throwaway staged-validation for REVL-04 wp-bench wiring (advisor step 2).

Boots vLLM on the reasoning-merged model, runs wp-bench with --limit 2 against the
live endpoint + wp-env grader, confirms a score comes back, then stops vLLM.
Confirms endpoint+grader wiring is sound before the full two-model comparison.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, stop_vllm  # noqa: E402

REASONING = "models/qwen3-30b-wp-30_70-reasoning-merged"
WP_BENCH_DIR = PROJECT_ROOT / "wp-bench"
PORT = 8021
NAME = "wp-revl04-wiring-vllm"
OUT = PROJECT_ROOT / "output" / "eval_reasoning" / "_wiring_check"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    conf = yaml.safe_load((PROJECT_ROOT / "config" / "wp-bench.yaml").read_text()) or {}
    endpoint = f"http://localhost:{PORT}/v1"
    conf["models"][0]["api_base"] = endpoint
    conf["models"][0]["name"] = "openai/wp-30_70"  # litellm provider prefix
    grader = conf.get("grader", {})
    if grader.get("wp_env_dir"):
        wed = Path(grader["wp_env_dir"])
        grader["wp_env_dir"] = str(wed if wed.is_absolute() else (PROJECT_ROOT / wed).resolve())
    out_json = OUT / "wp_bench_results.json"
    conf.setdefault("output", {})["path"] = str(out_json)
    conf["output"]["jsonl_path"] = str(out_json.with_suffix(".jsonl"))
    tmp = OUT / "wp_bench_config_tmp.yaml"
    tmp.write_text(yaml.dump(conf))

    try:
        print(f"[wiring] booting vLLM on {REASONING} ...", file=sys.stderr)
        boot_vllm(REASONING, NAME, PORT, 0.55)
        served = wait_healthy(PORT, NAME)
        print(f"[wiring] vLLM healthy, served={served}; wp-bench --limit 2 ...", file=sys.stderr)
        import os
        env = os.environ.copy()
        env["PATH"] = str(PROJECT_ROOT / "scripts" / "_wpbench_shim") + os.pathsep + env.get("PATH", "")
        env["NODE_OPTIONS"] = (env.get("NODE_OPTIONS", "").strip()
                               + " --dns-result-order=ipv4first"
                               " --network-family-autoselection-attempt-timeout=2000").strip()
        env.setdefault("OPENAI_API_KEY", "EMPTY")  # litellm openai provider needs a key
        r = subprocess.run(
            ["wp-bench", "run", "--config", str(tmp), "--limit", "2"], env=env,
            capture_output=True, text=True, timeout=1800, cwd=str(WP_BENCH_DIR))
        print("--- wp-bench stdout (tail) ---", file=sys.stderr)
        print((r.stdout or "")[-2000:], file=sys.stderr)
        if r.returncode != 0:
            print("--- wp-bench stderr (tail) ---", file=sys.stderr)
            print((r.stderr or "")[-3000:], file=sys.stderr)
            print(f"[wiring] FAIL: wp-bench exit {r.returncode}", file=sys.stderr)
            return 1
        res = json.loads(out_json.read_text()) if out_json.exists() else {}
        score = res.get("score") or res.get("wpbench_score") or res.get("metadata", {}).get("scores")
        print(f"[wiring] PASS: wp-bench score = {score}", file=sys.stderr)
        return 0
    finally:
        stop_vllm(NAME)


if __name__ == "__main__":
    sys.exit(main())
