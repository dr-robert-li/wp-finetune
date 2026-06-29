#!/usr/bin/env python
"""RLEV-01 fixed-set judge-Spearman eval across the RL checkpoints (post step-250 stop).

Pipeline-CONSISTENT (advisor-mandated): warmstart + every RL checkpoint go through the
IDENTICAL path — Tinker-sample on openai_val (capture_judge_responses_tinker, temp 0.0,
filter wp_judge_startswith) -> eval_judge.run_eval offline calibrated_canonical. No vLLM,
no merge. Verdict keys off the 50->250 TREND + bootstrap CI (improved_beyond_noise = CI
lower > 0), NOT a point comparison to the stale merged-vLLM 0.1534.

Run in .venv-tinker (has tinker + scipy):
  set -a; . ./.env; set +a
  .venv-tinker/bin/python scripts/_rlev01_batch.py
"""
import json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))

VAL = "data/reasoning_dataset/openai_val.jsonl"
OUT = REPO / "output/rl_eval"
# seedA live run (post-KL-fix). Old preKLfix run was 03c69b7b — DO NOT use.
# Verified vs output/rl_checkpoints/metrics/manifest.seedA.json sampler paths (2026-06-28).
RL_RUN = "tinker://9cb14129-f302-5c84-adf2-cc9ab92128a4:train:0/sampler_weights"
# warmstart = the v4 savestate promoted ep3 SAMPLER (same weights the RL run init-from'd),
# captured through the identical pipeline (NOT the stale 0.1534 merged-vLLM number).
# NOTE(2026-06-28): manifest init_from is .../weights/...-final-state; this uses the ep3
# SAMPLER export. Treated as the documented baseline per original design — VERIFY ep3==
# the init weights before trusting the warmstart-relative bootstrap at step-500.
CKPTS = [
    ("warmstart", "tinker://80c93d7c-2044-5dae-8e45-12dc1574d8f3:train:0/sampler_weights/wp-reasoning-v4-r32-rp30-savestate-ep3"),
    ("step-50",  f"{RL_RUN}/step-50"),
    ("step-100", f"{RL_RUN}/step-100"),
    ("step-150", f"{RL_RUN}/step-150"),
    ("step-200", f"{RL_RUN}/step-200"),
    ("step-250", f"{RL_RUN}/step-250"),
    ("step-300", f"{RL_RUN}/step-300"),
    ("step-350", f"{RL_RUN}/step-350"),
    ("step-400", f"{RL_RUN}/step-400"),
    ("step-450", f"{RL_RUN}/step-450"),
    ("step-500", f"{RL_RUN}/step-500"),
]


def capture(name, path):
    out = OUT / name / "judge_responses.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    # idempotent: skip if already captured with a sane line count (>100)
    if out.exists() and sum(1 for _ in out.open()) > 100:
        print(f"[{name}] capture cached ({out})", flush=True)
        return str(out)
    print(f"[{name}] capturing -> {out}", flush=True)
    r = subprocess.run(
        [".venv-tinker/bin/python", "scripts/capture_judge_responses_tinker.py",
         "--tinker-path", path, "--dataset", VAL, "--out", str(out),
         "--temperature", "0.0", "--filter", "wp_judge_startswith", "--max-tokens", "1024"],
        capture_output=True, text=True, timeout=3600)
    sys.stdout.write(r.stdout[-2000:] if r.stdout else "")
    if r.returncode != 0:
        print(f"[{name}] CAPTURE FAILED: {r.stderr[-1500:]}", flush=True)
        return None
    return str(out)


def evaluate(name, responses):
    from eval.eval_judge import run_eval
    op = str(OUT / name / "eval_judge_results.json")
    print(f"[{name}] eval offline -> {op}", flush=True)
    res = run_eval(dataset_path=VAL, output_path=op, gt_mode="calibrated_canonical",
                   responses_jsonl=responses, output_format="auto")
    return op, res


def load_pairs(name):
    """Return {index: (model_overall, gt_overall)} for rows with BOTH non-null."""
    p = OUT / name / "eval_judge_results.pairs.jsonl"
    d = {}
    if not p.exists():
        return d
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        mo, gt = r.get("model_overall"), r.get("gt_overall")
        if isinstance(r.get("index"), int) and mo is not None and gt is not None:
            d[r["index"]] = (float(mo), float(gt))
    return d


def main():
    results = {}
    pairs = {}
    for name, path in CKPTS:
        resp = capture(name, path)
        if not resp:
            results[name] = {"error": "capture_failed"}
            continue
        try:
            _, res = evaluate(name, resp)
            ov = res.get("overall_spearman", {})
            corr = ov.get("corr") if isinstance(ov, dict) else ov
            n = ov.get("n_pairs") if isinstance(ov, dict) else None
            results[name] = {"spearman": corr, "n_pairs": n}
            pairs[name] = load_pairs(name)
            print(f"[{name}] spearman={corr} n_pairs={n} pairs_loaded={len(pairs[name])}", flush=True)
        except Exception as e:  # noqa: BLE001
            results[name] = {"error": repr(e)}
            print(f"[{name}] EVAL ERROR: {e!r}", flush=True)

    # Inner-join on common indices where ALL captured checkpoints have a valid pair
    have = [n for n, _ in CKPTS if pairs.get(n)]
    common = None
    for n in have:
        ks = set(pairs[n].keys())
        common = ks if common is None else (common & ks)
    common = sorted(common or [])
    print(f"\ncommon aligned indices across {have}: n={len(common)}", flush=True)

    import importlib
    bg = importlib.import_module("scripts.bootstrap_gate")
    boot = {}
    if "warmstart" in pairs and len(common) >= 5:
        gt = [pairs["warmstart"][i][1] for i in common]
        base = [pairs["warmstart"][i][0] for i in common]
        for n in have:
            if n == "warmstart":
                continue
            cand = [pairs[n][i][0] for i in common]
            try:
                boot[n] = bg.bootstrap_spearman_improvement(cand, gt, base, n_boot=2000)
            except Exception as e:  # noqa: BLE001
                boot[n] = {"error": repr(e)}

    # Recompute aligned per-checkpoint Spearman on the COMMON set (apples-to-apples)
    from scipy.stats import spearmanr
    aligned_sp = {}
    if common:
        gt = [pairs[have[0]][i][1] for i in common]
        for n in have:
            pred = [pairs[n][i][0] for i in common]
            aligned_sp[n] = float(spearmanr(pred, gt).statistic)

    summary = {
        "n_common_aligned": len(common),
        "raw_spearman_per_ckpt": results,
        "aligned_spearman_common_set": aligned_sp,
        "bootstrap_vs_warmstart": boot,
        "checkpoints_order": [n for n, _ in CKPTS],
    }
    sp = OUT / "rlev01_summary.json"
    sp.write_text(json.dumps(summary, indent=2))
    print("\n=== RLEV-01 SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nwritten: {sp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
