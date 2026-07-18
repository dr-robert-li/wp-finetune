#!/usr/bin/env python
"""Phase 21 Plan 06 Task 1 -- JUDGE-03 cheap-path: Tinker-capture rho per seed
(8192-token cap, v4 base+renderer) + 3-seed median ensemble.

Orchestrates three ALREADY-EXISTING, UNCHANGED scorers (per plan: eval_relabel.py
and eval_relabel_ensemble.py are base-agnostic and need no change) around the
one real diff this plan makes -- capture_judge_responses_tinker.py's new
--base-model/--renderer flags (v4-aware, back-compat default v3):

  1. capture_judge_responses_tinker.py (--max-tokens 8192, --base-model v4)
     per seed -> output/base21/judge_capture_s<seed>.jsonl (runs in .venv-tinker)
  2. scripts/relabel/eval_relabel.py <capture> per seed -> per-seed rho + CI +
     parse_fail (runs in the project/conda env; scipy + eval/ package). Its
     eval_summary.json output path is shared/overwritten across runs, so each
     seed's summary is copied to a sidecar file immediately after scoring.
  3. scripts/relabel/eval_relabel_ensemble.py on all 3 captures -> 3-seed
     MEDIAN ensemble rho + CI (already implements exactly this: median overall
     per item across seeds, then Spearman + the same 2000-resample bootstrap).

Writes output/base21/judge03_capture_rho.json.

Usage (project/conda python -- shells out to .venv-tinker itself for step 1):
    python scripts/build_judge03_capture_rho.py
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "output" / "base21"
SEEDS = [1, 0, 2]
MAX_TOKENS = 8192
DATASET = "data/reasoning_dataset/openai_val.jsonl"
V4_BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
TINKER_PY = str(PROJECT_ROOT / ".venv-tinker" / "bin" / "python")
PROJECT_PY = sys.executable


def _capture(seed: int) -> str:
    manifest = f"output/tinker/wp-judge-v4-s{seed}-manifest.json"
    out = OUT_DIR / f"judge_capture_s{seed}.jsonl"
    cmd = [TINKER_PY, "scripts/capture_judge_responses_tinker.py",
           "--manifest", manifest, "--dataset", DATASET,
           "--max-tokens", str(MAX_TOKENS), "--temperature", "0.0",
           "--base-model", V4_BASE_MODEL, "--out", str(out)]
    print(f"[judge03-capture] seed {seed}: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600)
    print(r.stdout[-6000:])
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"capture failed for seed {seed} (exit {r.returncode})")
    return str(out)


def _eval_seed(cap_path: str, seed: int) -> dict:
    cmd = [PROJECT_PY, "scripts/relabel/eval_relabel.py", cap_path]
    print(f"[judge03-capture] scoring seed {seed}: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=600)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"eval_relabel failed for seed {seed} (exit {r.returncode})")

    # eval_relabel.py writes to a shared os.path.dirname(CAP)/eval_summary.json
    # path -- preserve THIS seed's summary before the next seed's run overwrites it.
    shared_summary = OUT_DIR / "eval_summary.json"
    sidecar = OUT_DIR / f"judge_capture_s{seed}_eval_summary.json"
    shutil.copy(shared_summary, sidecar)
    summary = json.loads(sidecar.read_text())

    m = re.search(r"parse_fail=(\d+)", r.stdout)
    parse_fail = int(m.group(1)) if m else summary.get("parse_fail")
    return {
        "seed": seed,
        "rho": summary["rho_new"],
        "ci_lower": summary["ci"][0],
        "ci_upper": summary["ci"][1],
        "n": summary["n"],
        "parse_fail": parse_fail,
    }


def _ensemble(cap_paths: list[str]) -> dict:
    out = OUT_DIR / "judge03_capture_ensemble.json"
    cmd = [PROJECT_PY, "scripts/relabel/eval_relabel_ensemble.py", str(out), *cap_paths]
    print(f"[judge03-capture] ensemble: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=600)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"eval_relabel_ensemble failed (exit {r.returncode})")
    d = json.loads(out.read_text())
    return {"rho": d["ensemble_rho"], "ci_lower": d["ci"][0], "ci_upper": d["ci"][1], "n": d["n"]}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cap_paths = []
    per_seed = []
    for seed in SEEDS:
        cap = _capture(seed)
        cap_paths.append(cap)
        per_seed.append(_eval_seed(cap, seed))

    ensemble = _ensemble(cap_paths)
    per_seed_by_rho = sorted(per_seed, key=lambda s: s["rho"], reverse=True)
    best = per_seed_by_rho[0]
    runner_up = per_seed_by_rho[1] if len(per_seed_by_rho) > 1 else None
    # WR-05: best_single_seed is picked by raw point-estimate rho alone (a
    # winner's-curse-prone choice with only 3 seeds); at minimum, log whether
    # the top seed's CI overlaps the runner-up's, so a human reviewing this
    # receipt before the costly JUDGE-03 merge+serve spend can see if the
    # "best" seed's advantage is inside the noise band.
    ci_overlaps_runner_up = (
        max(best["ci_lower"], runner_up["ci_lower"]) <= min(best["ci_upper"], runner_up["ci_upper"])
        if runner_up is not None else None
    )

    result = {
        "per_seed": per_seed,
        "best_single_seed": {"seed": best["seed"], "rho": best["rho"],
                              "ci_lower": best["ci_lower"], "ci_upper": best["ci_upper"]},
        "runner_up_seed": (
            {"seed": runner_up["seed"], "rho": runner_up["rho"],
             "ci_lower": runner_up["ci_lower"], "ci_upper": runner_up["ci_upper"]}
            if runner_up is not None else None
        ),
        "ci_overlaps_runner_up": ci_overlaps_runner_up,
        "ensemble_median": ensemble,
        "max_tokens": MAX_TOKENS,
        "method": "tinker_capture",
        "base_model": V4_BASE_MODEL,
    }
    out_path = OUT_DIR / "judge03_capture_rho.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[judge03-capture] wrote {out_path}", flush=True)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
