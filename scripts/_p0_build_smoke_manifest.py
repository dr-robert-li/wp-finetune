"""PR2a: build committed smoke-prompt manifest from openai_val.jsonl.

Deterministic selection (fixed val indices, no random) per council F4. Produces
data/phase4_4/smoke_prompts.json: 10 prompts = 5 judge-format + 5 gen-format,
including >=1 CtF-format row for critique-then-fix surface coverage.

NOTE (data-vs-spec correction 2026-05-29): val assistant targets use INLINE
dimensional reasoning ("WPCS Compliance: score 9/10 — ...") with NO literal
`<REASONING>` tags. Council's proposed mode-C "reasoning-trace present" check
(literal tag match) does not match the trained output format. The smoke gate
therefore measures the reasoning EFFECT via divergence-from-baseline, not tag
presence. Manifest records `expected_format` so Stage 2 can pick the right
coherence check per prompt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

VAL = "data/reasoning_dataset/openai_val.jsonl"
OUT = "data/phase4_4/smoke_prompts.json"

# Deterministic val indices. judge: first 4 standard + 1 CtF-format (idx 85).
# gen: 5 generation-token rows. Verified against val 2026-05-29.
JUDGE_IDXS = [0, 1, 2, 3, 85]
GEN_IDXS = [120, 121, 123, 124, 125]


def task_token(user: str) -> str:
    return user.split()[0] if user[:1] == "<" else "none"


def main() -> int:
    rows = [json.loads(l) for l in open(VAL)]
    manifest = []
    for kind, idxs in [("judge", JUDGE_IDXS), ("gen", GEN_IDXS)]:
        for i in idxs:
            r = rows[i]
            user = r["messages"][0]["content"]
            meta = r["metadata"]
            tok = task_token(user)
            entry = {
                "source_val_idx": i,
                "kind": kind,                       # judge | gen (coherence-check selector)
                "task_token": tok,                  # <wp_judge> | <wp_gen>
                "expected_format": meta.get("format"),  # cot | ctf | replay
                "function_name": meta.get("function_name"),
                "instruction": user,                # full prompt fed to model
                "reference_assistant": r["messages"][1]["content"],  # val target (audit only)
            }
            manifest.append(entry)

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(OUT, "w"), indent=2)
    # Summary
    from collections import Counter
    kinds = Counter(e["kind"] for e in manifest)
    fmts = Counter(e["expected_format"] for e in manifest)
    print(f"Wrote {len(manifest)} prompts -> {OUT}")
    print(f"  kinds: {dict(kinds)}")
    print(f"  formats: {dict(fmts)}")
    ctf = [e["source_val_idx"] for e in manifest if e["expected_format"] == "ctf"]
    print(f"  CtF coverage (idx): {ctf}")
    assert kinds["judge"] == 5 and kinds["gen"] == 5, "expected 5 judge + 5 gen"
    assert ctf, "manifest must include >=1 CtF-format prompt (F4)"
    return 0


if __name__ == "__main__":
    sys.exit(main())
