"""REVL-08 — reasoning-length distribution (SOFT/flag).

Confirms the model's reasoning chains are neither truncated nor exploding. Extracts the
reasoning prose from each captured response (close-only [/REASONING] format, <think>
stripped — same REASONING_RE as capture_reasoning_responses), tokenizes it with the
model tokenizer, and records median / p95 / max / mean token counts. SOFT flags fire
when p95 > 6000 (exploding) OR median < 500 (truncated/too-terse). Never blocks merge.

Reads output/eval_reasoning/reasoning_merged/captured_responses.jsonl.

Usage:
  python -m scripts.revl08_reasoning_length \
      --captured-jsonl output/eval_reasoning/reasoning_merged/captured_responses.jsonl \
      --out output/04.4_reasoning_length.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.capture_reasoning_responses import extract_reasoning  # noqa: E402

DEFAULT_CAPTURED = "output/eval_reasoning/reasoning_merged/captured_responses.jsonl"
DEFAULT_OUT = "output/04.4_reasoning_length.json"
TOKENIZER_DIR = "models/qwen3-30b-wp-30_70-reasoning-merged"
P95_EXPLODE = 6000
MEDIAN_TRUNCATE = 500


def _percentile(sorted_vals, q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def revl08(captured_jsonl: str, out_path: str, tokenizer_dir: str = TOKENIZER_DIR) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(PROJECT_ROOT / tokenizer_dir), trust_remote_code=True)
    cp = PROJECT_ROOT / captured_jsonl if not os.path.isabs(captured_jsonl) else Path(captured_jsonl)
    rows = [json.loads(l) for l in open(cp) if l.strip()]

    lengths, empty = [], 0
    for r in rows:
        reasoning = extract_reasoning(r.get("response", "") or "")
        if not reasoning.strip():
            empty += 1
            continue
        lengths.append(len(tok.encode(reasoning, add_special_tokens=False)))

    lengths_sorted = sorted(lengths)
    median = st.median(lengths_sorted) if lengths_sorted else 0
    p95 = _percentile(lengths_sorted, 0.95)
    mx = lengths_sorted[-1] if lengths_sorted else 0
    mn = lengths_sorted[0] if lengths_sorted else 0
    mean = st.mean(lengths_sorted) if lengths_sorted else 0

    flags = []
    if p95 > P95_EXPLODE:
        flags.append(f"exploding: p95 {p95:.0f} > {P95_EXPLODE}")
    if median < MEDIAN_TRUNCATE:
        flags.append(f"truncated/terse: median {median:.0f} < {MEDIAN_TRUNCATE}")

    result = {
        "gate": "REVL-08",
        "gate_class": "soft",
        "n_total": len(rows),
        "n_measured": len(lengths),
        "n_empty_reasoning": empty,
        "tokenizer": tokenizer_dir,
        "median_tokens": median,
        "p95_tokens": p95,
        "max_tokens": mx,
        "min_tokens": mn,
        "mean_tokens": mean,
        "p95_explode_threshold": P95_EXPLODE,
        "median_truncate_threshold": MEDIAN_TRUNCATE,
        "flags": flags,
        "flagged": bool(flags),
    }
    op = PROJECT_ROOT / out_path if not os.path.isabs(out_path) else Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="REVL-08 reasoning-length distribution (SOFT)")
    ap.add_argument("--captured-jsonl", default=DEFAULT_CAPTURED)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--tokenizer-dir", default=TOKENIZER_DIR)
    args = ap.parse_args()
    res = revl08(args.captured_jsonl, args.out, args.tokenizer_dir)
    print(f"[revl08] n_measured={res['n_measured']} median={res['median_tokens']:.0f} "
          f"p95={res['p95_tokens']:.0f} max={res['max_tokens']} flags={res['flags']}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
