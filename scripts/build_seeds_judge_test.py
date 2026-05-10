"""Phase 0 step 0.7: Build seeds_as_judge_test.jsonl from human + UGC + boundary seeds.

Converts each seed's human-annotated dimension scores into a synthetic
``messages[user wp_judge, assistant scored-JSON]`` example so that ``eval_judge.py``
can compute model Spearman against human ground truth.

Score scale conversion: seed dim scores are 0-10; judge output fields are 0-100.
Multiply by 10. Seed overall_score (DJC schema) is already 0-100.

Output: output/diagnostic/seeds_as_judge_test.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Seed dim name -> judge field name (model output schema)
SEED_TO_JUDGE_FIELD = {
    "wpcs_compliance": "wpcs_compliance",
    "security": "security_score",
    "sql_safety": "sql_safety",
    "performance": "performance_score",
    "wp_api_usage": "wp_api_usage",
    "i18n": "i18n_score",
    "i18n_l10n": "i18n_score",
    "accessibility": "accessibility_score",
    "dependency_integrity": "wp_api_usage",
    "error_handling": "error_handling",
    "code_quality": "code_structure",
    "code_structure": "code_structure",
}

SEED_FILES = [
    "deps/wp-finetune-data/human_seeds/human_annotated_seeds.json",
    "deps/wp-finetune-data/ugc_seeds.json",
    "deps/wp-finetune-data/ugc_boundary_seeds.json",
]


def _ensure_php_tag(code: str) -> str:
    if re.search(r"<\?(php|=)", code[:200]):
        return code
    return "<?php\n" + code


def seed_to_record(seed: dict) -> dict | None:
    code = seed.get("defective_code") or seed.get("code")
    if not code:
        return None
    code = _ensure_php_tag(code)

    if "human_critique" in seed:
        dims = seed["human_critique"].get("dimensions", {})
        overall_seed = None  # CtF schema has no overall_score
    elif "human_reasoning" in seed:
        dims = seed["human_reasoning"].get("dimension_analysis", {})
        overall_seed = seed["human_reasoning"].get("overall_score")
    else:
        return None

    fields: dict[str, int] = {}
    for seed_dim, payload in dims.items():
        if not isinstance(payload, dict):
            continue
        score = payload.get("score")
        if score is None:
            continue
        out_field = SEED_TO_JUDGE_FIELD.get(seed_dim)
        if out_field is None:
            continue
        fields.setdefault(out_field, int(round(float(score) * 10)))

    if not fields:
        return None

    if overall_seed is not None:
        overall = int(round(float(overall_seed)))
    else:
        overall = int(round(sum(fields.values()) / len(fields)))

    assistant_payload = {
        "overall_score": overall,
        **fields,
        "passes_threshold": overall >= 80,
        "explanation": f"Seed-derived ground truth from {seed['seed_id']} ({seed['defect_subtlety']} defect).",
        "_seed_id": seed["seed_id"],
        "_defect_subtlety": seed["defect_subtlety"],
    }

    user = f"<wp_judge> Evaluate this WordPress code:\n\n```php\n{code}\n```"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(assistant_payload)},
        ],
        "_seed_id": seed["seed_id"],
    }


def main():
    out_path = ROOT / "output" / "diagnostic" / "seeds_as_judge_test.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_out = 0
    skipped: list[str] = []
    with out_path.open("w") as f:
        for rel in SEED_FILES:
            with open(ROOT / rel) as g:
                seeds = json.load(g)
            for s in seeds:
                n_total += 1
                rec = seed_to_record(s)
                if rec is None:
                    skipped.append(s.get("seed_id", "?"))
                    continue
                f.write(json.dumps(rec) + "\n")
                n_out += 1
    print(f"Wrote {n_out}/{n_total} judge-test records to {out_path}")
    if skipped:
        print(f"Skipped {len(skipped)} (no code or no scored dims). First 5: {skipped[:5]}")


if __name__ == "__main__":
    main()
