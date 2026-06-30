#!/usr/bin/env python
"""Generic RLEV-01 wp-bench (REVL-04) on a merged seedA checkpoint vs cached v1.2 0.4616.
Usage: _rlev01_wpbench_ckpt.py --model-dir <merged dir> --tag <name>
"""
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.run_eval_reasoning import _wpbench_with_boot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--tag", required=True)
cli = ap.parse_args()

OUT = REPO / "output/rl_eval" / f"wpbench_{cli.tag}"
OUT.mkdir(parents=True, exist_ok=True)
BASELINE_V12 = 0.4616  # D-10-03 v1.2 SFT bar; sub-floors knowledge>=0.45 exec>=0.375

res = _wpbench_with_boot(cli.model_dir, f"wp-eval-{cli.tag}-vllm", cli.tag, 0.55, OUT)
score = res.get("wpbench_score")
res["baseline_v12_0p4616"] = BASELINE_V12
res["passes_hard_gate"] = (score is not None and score >= BASELINE_V12)
(OUT / "wpbench_result.json").write_text(json.dumps(res, indent=2))
print("WPBENCH_RESULT:", json.dumps(res, indent=2))
print(f"\n{cli.tag} wp-bench={score} vs v1.2 0.4616 -> "
      f"{'PASS' if res['passes_hard_gate'] else 'FAIL/REGRESSION'}")
