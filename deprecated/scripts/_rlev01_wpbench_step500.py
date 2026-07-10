#!/usr/bin/env python
"""RLEV-01 wp-bench (REVL-04) on merged seedA step-500. Compares to cached v1.2 SFT 0.4616.

Surgical: reuses run_eval_reasoning._wpbench_with_boot (boot vLLM -> wp-bench -> stop).
Baseline (v1.2) wp-bench is cached on disk (0.4616) — only the candidate is benched here.
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.run_eval_reasoning import _wpbench_with_boot  # noqa: E402

MODEL = str(REPO / "models/_staging/qwen3-30b-wp-seedA-step500-merged")
OUT = REPO / "output/rl_eval/wpbench_seedA_step500"
OUT.mkdir(parents=True, exist_ok=True)
BASELINE_V12 = 0.4616  # D-10-03 v1.2 SFT bar (HARD); sub-floors knowledge>=0.45 exec>=0.375

res = _wpbench_with_boot(MODEL, "wp-eval-seedA-step500-vllm", "seedA_step500", 0.55, OUT)
res["baseline_v12_0p4616"] = BASELINE_V12
score = res.get("wpbench_score")
res["passes_hard_gate"] = (score is not None and score >= BASELINE_V12)
(OUT / "wpbench_result.json").write_text(json.dumps(res, indent=2))
print("RLEV01_WPBENCH_RESULT:", json.dumps(res, indent=2))
print(f"\nstep-500 wp-bench={score} vs v1.2 0.4616 -> "
      f"{'PASS' if res['passes_hard_gate'] else 'FAIL/CHECK'}")
