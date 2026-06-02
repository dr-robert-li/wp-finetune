"""REVL-03 capture pass: re-inference the reasoning-merged model and persist raw prose.

The W2-02 calibrated-canonical harness (run_eval_reasoning.py) persisted SCORES ONLY
(eval_judge pairs.jsonl has no response/reasoning text). REVL-03 judges the model's
reasoning prose, so a fresh capture pass is mandatory before REVL-03 can run on
anything non-vacuous.

One vLLM boot serves the whole pass. For each cot/ctf row (replay EXCLUDED — it is
code-gen, not judge-task), send the <wp_judge> user prompt, record the raw response
and parsed scores. A format-histogram self-assert HALTS (non-zero exit) if the
captured output diverges from the expected judge format, so a silent vacuous REVL-03
can never result from a model/format drift.

Lifecycle mirrors scripts/run_eval_reasoning.py: boot_vllm -> wait_healthy -> work ->
stop_vllm in finally. Client idiom copied from eval/eval_gen.py.

Usage:
  python -m scripts.capture_reasoning_responses \
      --dataset data/reasoning_dataset/openai_val.jsonl \
      --out output/eval_reasoning/reasoning_merged/captured_responses.jsonl \
      --include-streams cot,ctf --gpu-mem-util 0.55
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, stop_vllm, VllmBootTimeout  # noqa: E402

REASONING = "models/qwen3-30b-wp-30_70-reasoning-merged"  # served as wp-30_70
PORT = 8021
SERVED_MODEL = "wp-30_70"
DEFAULT_OUT = "output/eval_reasoning/reasoning_merged/captured_responses.jsonl"

# Open tag OPTIONAL — the real dataset/model format is close-only ([/REASONING] with
# NO [REASONING] open tag; verified across 478 cot+ctf rows).
REASONING_RE = re.compile(r"(?:\[REASONING\])?(.*?)\[/REASONING\]", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def classify_task_type(row: dict) -> str:
    """Task type is metadata.stream (cot/ctf/replay), NOT the prompt tag."""
    return row["metadata"]["stream"]


def extract_reasoning(response: str) -> str:
    """Strip <think>...</think>; return prose before [/REASONING], else whole text."""
    text = _THINK_RE.sub("", response).strip()
    m = REASONING_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return text


def _user_messages(row: dict) -> list[dict]:
    return [m for m in row["messages"] if m["role"] == "user"]


def main() -> int:
    ap = argparse.ArgumentParser(description="REVL-03 capture pass (one vLLM boot)")
    ap.add_argument("--dataset", default="data/reasoning_dataset/openai_val.jsonl")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--include-streams", default="cot,ctf",
                    help="Comma list; replay EXCLUDED by default (code-gen, not judge).")
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--min-parseable-rate", type=float, default=0.80)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import openai
    from eval.eval_judge import parse_judge_response

    streams = {s.strip() for s in args.include_streams.split(",") if s.strip()}
    ds_path = PROJECT_ROOT / args.dataset if not os.path.isabs(args.dataset) else Path(args.dataset)
    rows = [json.loads(l) for l in open(ds_path) if l.strip()]
    todo = [(i, r) for i, r in enumerate(rows) if classify_task_type(r) in streams]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("HALT: no rows matched include-streams.", file=sys.stderr)
        return 1

    out_path = PROJECT_ROOT / args.out if not os.path.isabs(args.out) else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path = out_path.parent / "capture_format_histogram.json"

    name = "wp-eval-reasoning-vllm"
    endpoint = f"http://localhost:{PORT}/v1"
    captured = []
    try:
        boot_vllm(REASONING, name, PORT, args.gpu_mem_util)
        wait_healthy(PORT, name)
        client = openai.OpenAI(base_url=endpoint, api_key="none")
        with open(out_path, "w") as fh:
            for idx, row in todo:
                msgs = _user_messages(row)
                try:
                    resp = client.chat.completions.create(
                        model=SERVED_MODEL, messages=msgs,
                        max_tokens=args.max_tokens, temperature=0.0)
                    response = resp.choices[0].message.content or ""
                except Exception as e:  # noqa: BLE001
                    print(f"[capture] gen error idx {idx}: {e}", file=sys.stderr)
                    response = ""
                rec = {
                    "example_idx": idx,
                    "task_type": classify_task_type(row),
                    "prompt": msgs[0]["content"] if msgs else "",
                    "response": response,
                    "model_scores": parse_judge_response(response) if response else None,
                }
                captured.append(rec)
                fh.write(json.dumps(rec) + "\n")
    except VllmBootTimeout as e:
        print(f"HALT: vLLM boot failed: {e}", file=sys.stderr)
        return 3
    finally:
        stop_vllm(name)

    # Format histogram + min-parseable self-assert.
    n = len(captured)
    hist = {
        "n_total": n,
        "n_with_close_tag": sum("[/REASONING]" in (r["response"] or "") for r in captured),
        "n_with_judge_output": sum("<judge_output>" in (r["response"] or "") for r in captured),
        "n_with_corrected_code": sum("<corrected_code>" in (r["response"] or "") for r in captured),
        "n_parseable_scores": sum(r["model_scores"] is not None for r in captured),
    }
    hist["parseable_rate"] = (hist["n_parseable_scores"] / n) if n else 0.0
    hist_path.write_text(json.dumps(hist, indent=2))
    print(f"[capture] histogram: {json.dumps(hist)}", file=sys.stderr)

    if hist["parseable_rate"] < args.min_parseable_rate:
        print(f"HALT: parseable_rate {hist['parseable_rate']:.3f} < "
              f"{args.min_parseable_rate}; captured format diverged from expected "
              "judge-format — downstream REVL-03 would be vacuous.", file=sys.stderr)
        return 2
    print(f"[capture] OK — {n} rows, parseable_rate {hist['parseable_rate']:.3f} -> {out_path}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
