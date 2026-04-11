#!/usr/bin/env python3
"""Merge all data sources into data/final_dataset/wordpress_finetune.jsonl.

Converts all pipeline outputs into the OpenAI messages format expected
by export_dataset.py. Handles:
- Real code passed (wp_gen)
- Synthetic passed (wp_gen)
- Judge training data (wp_judge)
- CoT reasoning (wp_gen with reasoning)
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = PROJECT_ROOT / "data" / "final_dataset"


# ---------------------------------------------------------------------------
# Instruction enrichment (Fix 2)
# ---------------------------------------------------------------------------

def extract_instruction_from_docblock(docblock: str) -> str:
    """Extract the summary description from a PHPDoc block.

    Parses the first non-tag line after the opening comment marker.
    Returns the description or empty string if not found.
    """
    if not docblock:
        return ""

    lines = docblock.strip().split("\n")
    description_parts = []

    for line in lines:
        # Strip comment markers
        cleaned = re.sub(r"^\s*/?[*]+\s?/?", "", line).strip()
        if not cleaned:
            continue
        # Stop at @param, @return, @since, @deprecated, etc.
        if cleaned.startswith("@"):
            break
        # Stop at closing comment
        if cleaned == "/":
            break
        description_parts.append(cleaned)

    description = " ".join(description_parts).strip()
    # Clean up common artifacts
    description = re.sub(r"\s+", " ", description)
    # Skip if it's just the function name repeated or too short
    if len(description) < 10:
        return ""
    return description


def synthesize_instruction(func: dict) -> str:
    """Build an instruction from function metadata when no docblock exists.

    Uses function name, hooks, SQL patterns, and dependencies to construct
    a natural-sounding instruction.
    """
    name = func.get("function_name", "")
    hooks = func.get("hooks_used", [])
    sql = func.get("sql_patterns", [])
    deps = func.get("dependencies", [])
    class_ctx = func.get("class_context", "")

    parts = []

    # Convert function name to natural language
    # e.g., "get_user_meta_value" -> "get user meta value"
    clean_name = name.split("::")[-1] if "::" in name else name
    words = re.sub(r"([A-Z])", r" \1", clean_name)  # camelCase split
    words = words.replace("_", " ").strip().lower()

    # Build context from detected patterns
    if sql:
        sql_types = []
        if any("prepared" in s for s in sql):
            sql_types.append("prepared SQL queries")
        if any("join" in s.lower() for s in sql):
            sql_types.append("JOIN queries")
        if any("insert" in s.lower() for s in sql):
            sql_types.append("database insertions")
        if sql_types:
            parts.append(f"using {', '.join(sql_types)}")

    if hooks:
        hook_names = [h.split("(")[0] for h in hooks[:3]]
        if any("add_action" in h for h in hooks):
            parts.append("registering WordPress action hooks")
        elif any("add_filter" in h for h in hooks):
            parts.append("applying WordPress filters")

    # Security patterns from dependencies
    sec_apis = [d for d in deps if d in (
        "wp_verify_nonce", "check_ajax_referer", "current_user_can",
        "sanitize_text_field", "esc_html", "esc_attr", "esc_url",
    )]
    if sec_apis:
        parts.append(f"with {', '.join(sec_apis[:2])} for security")

    if class_ctx:
        parts.append(f"in the {class_ctx} class")

    if parts:
        return f"Write a WordPress function that {words}, {', '.join(parts)}"
    elif len(words) > 5:
        return f"Write a WordPress function that {words}"

    return ""


def build_gen_instruction(func: dict) -> str:
    """Build the best available instruction for a gen training example.

    Priority: docblock description > synthesized from metadata > fallback name-only.
    """
    # Try docblock first (available for ~90% of functions)
    instruction = extract_instruction_from_docblock(func.get("docblock", ""))
    if instruction:
        return instruction

    # Try synthesis from metadata
    instruction = synthesize_instruction(func)
    if instruction:
        return instruction

    # Fallback: original name-only format
    return f"Write a WordPress function: {func.get('function_name', 'unnamed')}"


def merge_all():
    FINAL_DIR.mkdir(exist_ok=True)
    examples = []

    # 1. Real code passed (wp_gen)
    passed_dir = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "passed"
    if passed_dir.exists():
        for f in sorted(passed_dir.glob("*.json")):
            for func in json.loads(f.read_text()):
                body = func.get("body", "")
                if not body.strip():
                    continue
                tags = func.get("assessment", {}).get("training_tags", func.get("training_tags", []))
                instruction = build_gen_instruction(func)
                examples.append({
                    "messages": [
                        {"role": "user", "content": f"<wp_gen> {instruction}"},
                        {"role": "assistant", "content": body},
                    ],
                    "metadata": {
                        "source": "real_code",
                        "source_repo": func.get("source_repo", f.stem),
                        "function_name": func.get("function_name", ""),
                        "quality_tier": func.get("quality_tier", "assessed"),
                        "training_tags": tags,
                        "task_type": "gen",
                    },
                })
    print(f"  Real code: {len(examples)}")

    # 2. Synthetic passed (wp_gen)
    s = len(examples)
    judged_dir = PROJECT_ROOT / "data" / "phase2_synthetic" / "output" / "judged"
    if judged_dir.exists():
        for f in sorted(judged_dir.glob("passed_*.json")):
            for item in json.loads(f.read_text()):
                body = item.get("body", "")
                if not body.strip():
                    continue
                instr = item.get("instruction", f"<wp_gen> Write a WordPress function: {item.get('function_name', '')}")
                if not instr.startswith("<wp_gen>"):
                    instr = f"<wp_gen> {instr}"
                examples.append({
                    "messages": [
                        {"role": "user", "content": instr},
                        {"role": "assistant", "content": body},
                    ],
                    "metadata": {
                        "source": "synthetic",
                        "gap_tag": item.get("gap_tag", ""),
                        "training_tags": item.get("training_tags", []),
                        "task_type": "gen",
                    },
                })
    print(f"  Synthetic: {len(examples) - s}")

    # 3. Judge training (wp_judge)
    s = len(examples)
    jt_dir = PROJECT_ROOT / "data" / "phase2_synthetic" / "output" / "judge_training"
    if jt_dir.exists():
        for f in sorted(jt_dir.glob("*.json")):
            for item in json.loads(f.read_text()):
                # Support both old (instruction/response) and new (messages) format
                if "messages" in item:
                    msgs = item["messages"]
                    if len(msgs) >= 2:
                        user_msg = msgs[0].get("content", "")
                        asst_msg = msgs[1].get("content", "")
                        if not user_msg.strip():
                            continue
                        examples.append({
                            "messages": msgs,
                            "metadata": item.get("metadata", {
                                "source": "judge_training",
                                "task_type": "judge",
                            }),
                        })
                else:
                    instr = item.get("instruction", "")
                    if not instr.strip():
                        continue
                    resp = item.get("response", {})
                    resp_str = json.dumps(resp, indent=2) if isinstance(resp, dict) else str(resp)
                    if not instr.startswith("<wp_judge>"):
                        instr = f"<wp_judge> {instr}"
                    examples.append({
                        "messages": [
                            {"role": "user", "content": instr},
                            {"role": "assistant", "content": resp_str},
                        ],
                        "metadata": {
                            "source": "judge_training",
                            "quality_tier": item.get("quality_tier", ""),
                            "training_tags": item.get("training_tags", []),
                            "task_type": "judge",
                        },
                    })
    print(f"  Judge training: {len(examples) - s}")

    # 4. CoT reasoning (wp_gen with reasoning)
    s = len(examples)
    cot_dir = PROJECT_ROOT / "data" / "phase3_cot" / "output"
    if cot_dir.exists():
        for f in sorted(cot_dir.glob("*.json")):
            for item in json.loads(f.read_text()):
                # Support messages format (from Claude Code agents)
                if "messages" in item and isinstance(item["messages"], list):
                    msgs = item["messages"]
                    if len(msgs) >= 2:
                        examples.append({
                            "messages": msgs,
                            "metadata": item.get("metadata", {
                                "source": "cot",
                                "task_type": "gen",
                                "has_cot": True,
                            }),
                        })
                    continue

                # Old format: instruction/response/reasoning
                instr = item.get("instruction", "")
                resp = item.get("response", "")
                reasoning = item.get("reasoning", "") or item.get("cot_reasoning", "")
                # Handle dict responses (judge rubric/security CoT have scores as dict)
                if isinstance(resp, dict):
                    resp = json.dumps(resp)
                if not isinstance(instr, str) or not isinstance(resp, str):
                    continue
                if not instr.strip() or not resp.strip():
                    continue
                # Prepend reasoning to response if present (CoT chain-of-thought)
                if reasoning:
                    full_resp = f"## Reasoning\n\n{reasoning}\n\n## Answer\n\n{resp}"
                else:
                    full_resp = resp
                # Determine task type from instruction content
                task_type = item.get("task_type", "gen")
                if "<wp_judge>" in instr:
                    task_token = "<wp_judge>"
                    meta_task = "judge"
                else:
                    task_token = "<wp_gen>"
                    meta_task = "gen"
                    if not instr.startswith("<wp_gen>"):
                        instr = f"<wp_gen> {instr}"
                examples.append({
                    "messages": [
                        {"role": "user", "content": instr},
                        {"role": "assistant", "content": full_resp},
                    ],
                    "metadata": {
                        "source": "cot",
                        "complexity": item.get("complexity", ""),
                        "training_tags": item.get("training_tags", []),
                        "task_type": meta_task,
                    },
                })
    print(f"  CoT: {len(examples) - s}")

    # Write merged JSONL
    output = FINAL_DIR / "wordpress_finetune.jsonl"
    with open(output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    gen = sum(1 for e in examples if e["metadata"].get("task_type") == "gen")
    judge = sum(1 for e in examples if e["metadata"].get("task_type") == "judge")
    print(f"\nMerged: {len(examples)} examples → {output}")
    print(f"  gen: {gen} ({gen / len(examples) * 100:.1f}%)")
    print(f"  judge: {judge} ({judge / len(examples) * 100:.1f}%)")
    return len(examples)


if __name__ == "__main__":
    merge_all()
