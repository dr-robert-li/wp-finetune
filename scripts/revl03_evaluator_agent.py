"""REVL-03 — opaque Claude-evaluator agent PLAN EMITTER (no LLM API anywhere).

This script DOES NOT spawn agents and imports NO LLM client (no anthropic, no openai).
It reads the captured reasoning responses and emits one opaque agent-prompt per
judge-task sample to a JSONL plan file. The ORCHESTRATING Claude Code session then
dispatches the agents.

=== Orchestrating-session contract (executed by the Claude session, NOT Python) ===
After running this script:
  1. Read output/eval_reasoning/revl03_agent_plan.jsonl.
  2. For each row, dispatch Agent(model='sonnet', description="REVL-03 eval sample
     {sample_id}", prompt=row['agent_prompt'], run_in_background=true). Batch in
     parallel groups of <=50 (~120 rows => 3 batches).
  3. Each agent returns ONLY the strict JSON object
     {dimension_coverage, score_reasoning_consistency, coherence}. Append it (with its
     sample_id) as one line to output/eval_reasoning/revl03_claude_eval.jsonl.
     Validate each line parses + has all three keys; re-dispatch malformed results.
  4. When all results collected, run `python -m scripts.aggregate_revl03`.
The agents receive OPAQUE inputs — they are NOT told which model produced the output.
This is the Claude-Code-agents-only execution rule (ROADMAP §4.4); NO Anthropic API.

Usage:
  python -m scripts.revl03_evaluator_agent \
      --captured-jsonl output/eval_reasoning/reasoning_merged/captured_responses.jsonl \
      --plan-out output/eval_reasoning/revl03_agent_plan.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.capture_reasoning_responses import extract_reasoning  # noqa: E402

DEFAULT_CAPTURED = "output/eval_reasoning/reasoning_merged/captured_responses.jsonl"
DEFAULT_PLAN_OUT = "output/eval_reasoning/revl03_agent_plan.jsonl"
EVAL_OUT_PATH = "output/eval_reasoning/revl03_claude_eval.jsonl"


def build_agent_prompt(extracted_reasoning: str, model_scores, prompt: str) -> str:
    """Verbatim opaque prompt (04.4-RESEARCH.md Example 3)."""
    return (
        "You are evaluating WordPress code-review reasoning quality.\n"
        "You are NOT told which model produced this output.\n\n"
        "GENERATED REASONING:\n"
        f"{extracted_reasoning}\n\n"
        "MODEL'S NUMERIC SCORES (per the 9-dimension rubric):\n"
        f"{json.dumps(model_scores, indent=2)}\n\n"
        "CODE UNDER REVIEW:\n"
        f"{prompt}\n\n"
        "Answer three questions in strict JSON:\n"
        "1. dimension_coverage (object): for each of the 9 dimensions (D1..D9), does\n"
        "   the reasoning text explicitly address that dimension? true/false per key.\n"
        "2. score_reasoning_consistency (object): for each dimension where the\n"
        "   reasoning makes a claim, does the numeric score align with the claim?\n"
        "   true/false per key.\n"
        "3. coherence (1-5 integer): is the reasoning logically structured,\n"
        "   issue-specific, and free of contradictions?\n\n"
        "Output ONLY the JSON object — no prose."
    )


def emit_plan(captured_jsonl: str, plan_out: str, include_streams, sample_size=None) -> int:
    cap_path = PROJECT_ROOT / captured_jsonl if not os.path.isabs(captured_jsonl) else Path(captured_jsonl)
    rows = [json.loads(l) for l in open(cap_path) if l.strip()]
    rows = [r for r in rows if r.get("task_type") in include_streams]
    if sample_size:
        rows = rows[:sample_size]
    out_path = PROJECT_ROOT / plan_out if not os.path.isabs(plan_out) else Path(plan_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in rows:
            reasoning = extract_reasoning(r.get("response", ""))
            agent_prompt = build_agent_prompt(reasoning, r.get("model_scores"), r.get("prompt", ""))
            fh.write(json.dumps({
                "sample_id": r["example_idx"],
                "agent_prompt": agent_prompt,
                "expected_output_path": EVAL_OUT_PATH,
            }) + "\n")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="REVL-03 agent-plan emitter (no LLM API)")
    ap.add_argument("--captured-jsonl", default=DEFAULT_CAPTURED)
    ap.add_argument("--plan-out", default=DEFAULT_PLAN_OUT)
    ap.add_argument("--include-streams", default="cot,ctf")
    ap.add_argument("--sample-size", type=int, default=None)
    args = ap.parse_args()
    streams = {s.strip() for s in args.include_streams.split(",") if s.strip()}
    n = emit_plan(args.captured_jsonl, args.plan_out, streams, args.sample_size)
    print(f"[revl03] emitted {n} agent-plan rows -> {args.plan_out}", file=sys.stderr)
    print("[revl03] Orchestrating session: dispatch one Agent(model='sonnet', "
          "run_in_background=true) per row; collect to revl03_claude_eval.jsonl; "
          "then run scripts.aggregate_revl03.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
