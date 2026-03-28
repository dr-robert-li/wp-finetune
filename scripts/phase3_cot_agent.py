#!/usr/bin/env python3
"""Generate CoT-enriched training data from all pipeline outputs.

Transforms Phase 1 passed functions, Phase 2 judged synthetics,
Phase 2 judge training data, and rejection examples into
instruction-response pairs with step-by-step reasoning.

Replaces multi-agent LLM approach with deterministic template-based
generation. Produces per-agent JSONL files plus combined
final_dataset/wordpress_finetune.jsonl.
"""

import hashlib
import json
import os
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE1_PASSED = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "passed"
PHASE2_JUDGED = PROJECT_ROOT / "data" / "phase2_synthetic" / "output" / "judged"
PHASE2_JUDGE_TRAINING = PROJECT_ROOT / "data" / "phase2_synthetic" / "output" / "judge_training"
PHASE2_MUTATED = PROJECT_ROOT / "data" / "phase2_synthetic" / "output" / "mutated"
COT_OUTPUT = PROJECT_ROOT / "data" / "phase3_cot" / "output"
FINAL_DIR = PROJECT_ROOT / "data" / "final_dataset"

SYSTEM_PROMPT = (
    "You are a senior WordPress developer and VIP platform engineer. "
    "You write production-quality PHP following WordPress Coding Standards "
    "with strict security, performance, and API correctness. "
    "You think step-by-step about architectural decisions."
)

# Seed for reproducibility.
random.seed(42)

# ---------- Instruction templates ----------

GEN_INSTRUCTION_TEMPLATES = [
    "Write a WordPress function `{func_name}` that {purpose}.",
    "Implement `{func_name}` for WordPress that {purpose}.",
    "Create a PHP function `{func_name}` for a WordPress plugin that {purpose}.",
    "Write `{func_name}` — a WordPress function that {purpose}.",
    "Build a WordPress function called `{func_name}` that {purpose}.",
]

JUDGE_INSTRUCTION_TEMPLATE = (
    "<wp_judge> Evaluate this WordPress PHP code for quality across these dimensions: "
    "WPCS compliance, security, performance, i18n readiness, accessibility awareness, "
    "and overall quality. Provide a score (0-100) for each dimension with reasoning.\n\n"
    "```php\n{code}\n```"
)

REJECTION_INSTRUCTION_TEMPLATES = [
    "Write a WordPress function `{func_name}` that {purpose}.",
    "Implement `{func_name}` for WordPress that {purpose}.",
    "Create `{func_name}` for WordPress that {purpose}.",
]


# ---------- Purpose inference ----------

def infer_purpose(func: dict) -> str:
    """Infer a natural-language purpose from function metadata."""
    name = func.get("function_name", "unnamed")
    tags = func.get("training_tags", func.get("assessment", {}).get("training_tags", []))
    body = func.get("body", func.get("code", ""))
    docblock = func.get("docblock", "")

    # Try docblock first line.
    if docblock:
        lines = [l.strip().lstrip("*/ ") for l in docblock.split("\n") if l.strip().lstrip("*/ ")]
        for line in lines:
            if line and not line.startswith("@") and len(line) > 10:
                # Clean up docblock description.
                purpose = line.rstrip(".")
                if purpose and purpose[0].isupper():
                    purpose = purpose[0].lower() + purpose[1:]
                return purpose

    # Infer from function name.
    parts = re.sub(r"([A-Z])", r"_\1", name).lower()
    parts = parts.replace("::", "_").replace("__", "_").strip("_")
    words = parts.split("_")

    # Common WordPress patterns.
    if "register" in words:
        return f"registers {' '.join(w for w in words if w != 'register')}"
    if "get" in words:
        return f"retrieves {' '.join(w for w in words if w != 'get')}"
    if "set" in words:
        return f"sets {' '.join(w for w in words if w != 'set')}"
    if "handle" in words:
        return f"handles {' '.join(w for w in words if w != 'handle')}"
    if "init" in words or "initialize" in words:
        return f"initializes {' '.join(w for w in words if w not in ('init', 'initialize'))}"
    if "save" in words:
        return f"saves {' '.join(w for w in words if w != 'save')}"
    if "delete" in words or "remove" in words:
        return f"removes {' '.join(w for w in words if w not in ('delete', 'remove'))}"
    if "render" in words or "display" in words:
        return f"renders {' '.join(w for w in words if w not in ('render', 'display'))}"
    if "enqueue" in words:
        return f"enqueues {' '.join(w for w in words if w != 'enqueue')}"
    if "filter" in words:
        return f"filters {' '.join(w for w in words if w != 'filter')}"
    if "validate" in words or "check" in words:
        return f"validates {' '.join(w for w in words if w not in ('validate', 'check'))}"

    return f"implements {' '.join(words)} functionality"


def infer_tag_concerns(tags: list[str], body: str) -> list[str]:
    """Infer security/performance/API concerns from tags and code."""
    concerns = []

    tag_set = set(t.lower() for t in tags)
    body_lower = body.lower()

    # Security concerns.
    if any("security" in t or "nonce" in t or "sanitiz" in t or "escap" in t for t in tag_set):
        concerns.append("security (input sanitization, output escaping, nonce verification)")
    elif "wp_nonce" in body_lower or "sanitize_" in body_lower or "esc_" in body_lower:
        concerns.append("security (WordPress sanitization and escaping APIs)")

    # SQL concerns.
    if any("sql" in t for t in tag_set) or "$wpdb" in body_lower:
        concerns.append("SQL injection prevention via $wpdb->prepare()")

    # Performance concerns.
    if any("perf" in t or "cach" in t or "batch" in t for t in tag_set):
        concerns.append("performance (caching, batch processing, query optimization)")
    elif "wp_cache" in body_lower or "transient" in body_lower:
        concerns.append("performance (WordPress object cache / transients)")

    # REST API.
    if any("rest" in t for t in tag_set) or "register_rest_route" in body_lower:
        concerns.append("REST API best practices (permission callbacks, schema validation)")

    # Hooks.
    if any("hook" in t or "action" in t or "filter" in t for t in tag_set):
        concerns.append("proper WordPress hook usage (actions and filters)")

    # Multisite.
    if any("multisite" in t for t in tag_set) or "switch_to_blog" in body_lower:
        concerns.append("multisite compatibility")

    if not concerns:
        concerns.append("WordPress Coding Standards compliance")

    return concerns


# ---------- CoT reasoning generation ----------

def generate_cot_reasoning(func: dict) -> str:
    """Generate step-by-step reasoning for a wp_gen example."""
    name = func.get("function_name", "unnamed")
    tags = func.get("training_tags", func.get("assessment", {}).get("training_tags", []))
    body = func.get("body", func.get("code", ""))
    concerns = infer_tag_concerns(tags, body)

    reasoning_parts = ["Let me think through this step by step.\n"]

    # Step 1: Identify what APIs are needed.
    apis_used = []
    if "add_action" in body or "add_filter" in body:
        apis_used.append("WordPress hooks API (add_action/add_filter)")
    if "$wpdb" in body:
        apis_used.append("$wpdb for direct database queries with prepare()")
    if "register_rest_route" in body:
        apis_used.append("REST API route registration")
    if "register_post_type" in body:
        apis_used.append("Custom Post Type registration")
    if "wp_enqueue_" in body:
        apis_used.append("Script/style enqueue system")
    if "get_option" in body or "update_option" in body:
        apis_used.append("Options API")
    if "wp_cache" in body or "get_transient" in body or "set_transient" in body:
        apis_used.append("Object Cache / Transient API")
    if "WP_Query" in body or "get_posts" in body:
        apis_used.append("WP_Query for post retrieval")
    if "register_taxonomy" in body:
        apis_used.append("Taxonomy registration API")
    if "wp_ajax_" in body or "wp_ajax_nopriv_" in body:
        apis_used.append("AJAX handler registration")
    if "register_block" in body or "register_block_type" in body:
        apis_used.append("Block Editor registration API")

    if apis_used:
        reasoning_parts.append(
            f"First, I need to identify the right WordPress APIs. "
            f"This function uses: {', '.join(apis_used)}."
        )
    else:
        reasoning_parts.append(
            "First, I need to consider which WordPress APIs are appropriate here."
        )

    # Step 2: Security considerations.
    security_notes = []
    if "wp_verify_nonce" in body or "check_ajax_referer" in body:
        security_notes.append("nonce verification for CSRF protection")
    if "current_user_can" in body:
        security_notes.append("capability checks for authorization")
    if "sanitize_" in body:
        security_notes.append("input sanitization")
    if "esc_html" in body or "esc_attr" in body or "esc_url" in body or "wp_kses" in body:
        security_notes.append("output escaping to prevent XSS")
    if "$wpdb->prepare" in body:
        security_notes.append("prepared statements to prevent SQL injection")

    if security_notes:
        reasoning_parts.append(
            f"For security, I need to ensure: {', '.join(security_notes)}."
        )
    else:
        reasoning_parts.append(
            "I should consider security implications, though this function has minimal attack surface."
        )

    # Step 3: Performance notes.
    perf_notes = []
    if "wp_cache" in body:
        perf_notes.append("leveraging WordPress object cache")
    if "transient" in body.lower():
        perf_notes.append("using transients for expensive operations")
    if "LIMIT" in body or "OFFSET" in body:
        perf_notes.append("paginating queries for scalability")
    if "batch" in body.lower() or "chunk" in body.lower():
        perf_notes.append("batch processing to limit memory usage")

    if perf_notes:
        reasoning_parts.append(
            f"For performance: {', '.join(perf_notes)}."
        )

    reasoning_parts.append("\nHere's the implementation:")

    return "\n\n".join(reasoning_parts)


def generate_judge_response(item: dict) -> str:
    """Generate a structured judge response with reasoning from scores."""
    scores = item.get("scores", {})
    reasoning = item.get("reasoning", "")
    code = item.get("code", item.get("body", ""))

    parts = ["Here is my assessment of this WordPress code:\n"]

    dimension_labels = {
        "wpcs_compliance": "WPCS Compliance",
        "security_score": "Security",
        "performance_score": "Performance",
        "i18n_score": "Internationalization (i18n)",
        "accessibility_score": "Accessibility",
        "overall_quality": "Overall Quality",
    }

    for key, label in dimension_labels.items():
        score = scores.get(key, "N/A")
        if isinstance(score, (int, float)):
            # Generate reasoning for each dimension.
            if score >= 80:
                quality = "strong"
            elif score >= 60:
                quality = "adequate but could improve"
            elif score >= 40:
                quality = "below expectations"
            else:
                quality = "poor"
            parts.append(f"**{label}:** {score}/100 ({quality})")
        else:
            parts.append(f"**{label}:** {score}")

    if reasoning:
        parts.append(f"\n**Summary:** {reasoning}")

    return "\n".join(parts)


def generate_rejection_cot(func: dict) -> str:
    """Generate CoT for rejection examples that proactively add security."""
    tags = func.get("training_tags", [])
    body = func.get("body", func.get("code", ""))

    if any("proactive_nonce" in t for t in tags):
        return (
            "Let me think through this step by step.\n\n"
            "Even though the task doesn't mention security explicitly, I need to "
            "add nonce verification for any form submission or state-changing action. "
            "This is a WordPress security best practice that prevents CSRF attacks.\n\n"
            "I'll use wp_nonce_field() for the form and wp_verify_nonce() or "
            "check_admin_referer() on the handler side.\n\n"
            "Here's the implementation with proactive security:"
        )
    elif any("proactive_capability" in t for t in tags):
        return (
            "Let me think through this step by step.\n\n"
            "This task doesn't mention permissions, but I need to add capability "
            "checks. Any admin action must verify the user has the right capabilities "
            "using current_user_can(). This prevents privilege escalation.\n\n"
            "Here's the implementation with proper capability checks:"
        )
    elif any("proactive_escaping" in t for t in tags):
        return (
            "Let me think through this step by step.\n\n"
            "The task doesn't mention output escaping, but any data rendered in HTML "
            "must be escaped to prevent XSS. I'll use esc_html(), esc_attr(), "
            "esc_url(), and wp_kses_post() as appropriate.\n\n"
            "Here's the implementation with proper output escaping:"
        )
    else:
        return generate_cot_reasoning(func)


# ---------- Loading functions ----------

def load_phase1_passed() -> list[dict]:
    """Load all Phase 1 passed functions, sampling wordpress-develop."""
    examples = []

    for f in sorted(PHASE1_PASSED.glob("*.json")):
        data = json.loads(f.read_text())
        repo = f.stem

        if repo == "wordpress-develop":
            # Sample ~2000 diverse functions from the 11K+ core.
            if len(data) > 2000:
                # Stratified sample: prefer functions with tags.
                tagged = [d for d in data if d.get("training_tags") or d.get("assessment", {}).get("training_tags")]
                untagged = [d for d in data if not (d.get("training_tags") or d.get("assessment", {}).get("training_tags"))]
                random.shuffle(tagged)
                random.shuffle(untagged)
                # Take all tagged (up to 1500) plus fill with untagged.
                sample = tagged[:1500]
                remaining = 2000 - len(sample)
                sample.extend(untagged[:remaining])
                data = sample
                print(f"  Sampled wordpress-develop: {len(data)} of 11132")

        for func in data:
            func["_source_repo"] = repo
            func["_pipeline_source"] = "phase1_passed"
            examples.append(func)

    return examples


def load_phase2_judged() -> list[dict]:
    """Load all Phase 2 judged synthetic examples."""
    examples = []
    for f in sorted(PHASE2_JUDGED.glob("*.json")):
        data = json.loads(f.read_text())
        for func in data:
            func["_source_repo"] = f.stem
            func["_pipeline_source"] = "synthetic"
            # Tag rejection examples.
            tags = func.get("training_tags", [])
            if any("rejection:" in t for t in tags):
                func["_pipeline_source"] = "rejection"
            examples.append(func)
    return examples


def load_phase2_judge_training() -> list[dict]:
    """Load all Phase 2 judge training data."""
    examples = []
    for f in sorted(PHASE2_JUDGE_TRAINING.glob("*.json")):
        data = json.loads(f.read_text())
        for item in data:
            item["_pipeline_source"] = "judge_training"
            item["_source_file"] = f.stem
            examples.append(item)
    return examples


def load_mutations() -> list[dict]:
    """Load contrastive mutations."""
    path = PHASE2_MUTATED / "contrastive_mutations.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    for item in data:
        item["_pipeline_source"] = "mutated"
    return data


# ---------- Convert to JSONL format ----------

def convert_gen_example(func: dict, add_cot: bool = True) -> dict:
    """Convert a wp_gen function to the JSONL training format."""
    name = func.get("function_name", "unnamed")
    body = func.get("body", func.get("code", ""))
    tags = func.get("training_tags", func.get("assessment", {}).get("training_tags", []))
    source = func.get("_pipeline_source", "unknown")
    is_rejection = source == "rejection"

    if not body.strip():
        return None

    # Generate instruction.
    purpose = infer_purpose(func)
    if is_rejection:
        template = random.choice(REJECTION_INSTRUCTION_TEMPLATES)
    else:
        template = random.choice(GEN_INSTRUCTION_TEMPLATES)
    instruction = template.format(func_name=name, purpose=purpose)

    # Generate response.
    if is_rejection and add_cot:
        cot = generate_rejection_cot(func)
        response = f"{cot}\n\n```php\n{body}\n```"
    elif add_cot:
        cot = generate_cot_reasoning(func)
        response = f"{cot}\n\n```php\n{body}\n```"
    else:
        response = body

    # Determine sample weight.
    weight = 1.0
    if is_rejection:
        weight = 1.5
    elif source == "mutated":
        weight = 1.5

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<wp_gen> {instruction}"},
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "task_type": "gen",
            "source": source,
            "source_repo": func.get("_source_repo", func.get("source_repo", "")),
            "function_name": name,
            "training_tags": tags if isinstance(tags, list) else [],
            "sample_weight": weight,
        },
    }


def convert_judge_example(item: dict) -> dict:
    """Convert a wp_judge training item to the JSONL training format.

    Handles two formats:
    1. Scored format (phase1_*_scored.json, synthetic_scored.json):
       Has code, scores, reasoning fields.
    2. Pre-formatted (high_quality*.json, low_quality*.json, synthetic_scored_batch*.json):
       Has instruction, response fields already formatted.
    """
    # Check if pre-formatted (has instruction/response).
    if item.get("instruction") and item.get("response"):
        instruction = item["instruction"]
        if not instruction.startswith("<wp_judge>"):
            instruction = f"<wp_judge> {instruction}"

        resp = item["response"]
        if isinstance(resp, dict):
            # Format structured response.
            resp_str = format_judge_dict_response(resp)
        else:
            resp_str = str(resp)

        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": resp_str},
            ],
            "metadata": {
                "task_type": "judge",
                "source": "judge_training",
                "source_repo": item.get("source_repo", ""),
                "function_name": item.get("function_name", ""),
                "training_tags": item.get("training_tags", []),
                "sample_weight": 1.0,
            },
        }

    # Scored format with code/scores/reasoning.
    code = item.get("code", item.get("body", ""))
    if not code.strip():
        return None

    instruction = JUDGE_INSTRUCTION_TEMPLATE.format(code=code)
    response = generate_judge_response(item)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            "task_type": "judge",
            "source": "judge_training",
            "source_repo": item.get("source_repo", ""),
            "function_name": item.get("function_name", ""),
            "training_tags": item.get("training_tags", []),
            "sample_weight": 1.0,
        },
    }


def format_judge_dict_response(resp: dict) -> str:
    """Format a structured judge response dict into readable text."""
    parts = ["Here is my assessment of this WordPress code:\n"]

    # Score fields.
    score_labels = {
        "wpcs_compliance": "WPCS Compliance",
        "security_score": "Security",
        "performance_score": "Performance",
        "i18n_score": "Internationalization (i18n)",
        "accessibility_score": "Accessibility",
        "documentation_score": "Documentation",
        "overall_score": "Overall Quality",
        "overall_quality": "Overall Quality",
    }

    for key, label in score_labels.items():
        if key in resp:
            score = resp[key]
            if isinstance(score, (int, float)):
                if score >= 80:
                    quality = "strong"
                elif score >= 60:
                    quality = "adequate but could improve"
                elif score >= 40:
                    quality = "below expectations"
                else:
                    quality = "poor"
                parts.append(f"**{label}:** {score}/100 ({quality})")

    if resp.get("must_fix_issues"):
        parts.append(f"\n**Must-fix issues:**")
        for issue in resp["must_fix_issues"]:
            parts.append(f"- {issue}")

    if resp.get("suggested_improvements"):
        parts.append(f"\n**Suggested improvements:**")
        for imp in resp["suggested_improvements"]:
            parts.append(f"- {imp}")

    passes = resp.get("passes_threshold")
    if passes is not None:
        parts.append(f"\n**Passes threshold:** {'Yes' if passes else 'No'}")

    if resp.get("explanation"):
        parts.append(f"\n**Summary:** {resp['explanation']}")

    return "\n".join(parts)


# ---------- Main pipeline ----------

def split_alphabetical(examples: list[dict]) -> tuple[list, list, list]:
    """Split phase1 examples into 3 groups by repo name for agent-style output."""
    by_repo = {}
    for ex in examples:
        repo = ex.get("_source_repo", "zzz")
        by_repo.setdefault(repo, []).append(ex)

    repos = sorted(by_repo.keys())
    # A-G, H-Q, R-Z.
    group1, group2, group3 = [], [], []
    for repo in repos:
        first = repo[0].lower()
        if first <= "g":
            group1.extend(by_repo[repo])
        elif first <= "q":
            group2.extend(by_repo[repo])
        else:
            group3.extend(by_repo[repo])

    return group1, group2, group3


def write_agent_jsonl(examples: list[dict], agent_num: int, label: str) -> int:
    """Write per-agent JSONL and return count."""
    path = COT_OUTPUT / f"agent_{agent_num}.jsonl"
    written = 0
    with open(path, "w") as f:
        for ex in examples:
            if ex is not None:
                f.write(json.dumps(ex) + "\n")
                written += 1
    print(f"  Agent {agent_num} ({label}): {written} examples -> {path.name}")
    return written


def main():
    COT_OUTPUT.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading pipeline outputs...")
    phase1 = load_phase1_passed()
    print(f"  Phase 1 passed: {len(phase1)} functions")

    phase2_judged = load_phase2_judged()
    rejection_examples = [e for e in phase2_judged if e.get("_pipeline_source") == "rejection"]
    synthetic_examples = [e for e in phase2_judged if e.get("_pipeline_source") == "synthetic"]
    print(f"  Phase 2 judged synthetics: {len(synthetic_examples)} functions")
    print(f"  Phase 2 rejection examples: {len(rejection_examples)} functions")

    judge_training = load_phase2_judge_training()
    print(f"  Phase 2 judge training: {len(judge_training)} examples")

    mutations = load_mutations()
    print(f"  Mutations: {len(mutations)} pairs")

    # Split phase1 into 3 groups.
    group1, group2, group3 = split_alphabetical(phase1)
    print(f"\nPhase 1 split: A-G={len(group1)}, H-Q={len(group2)}, R-Z={len(group3)}")

    # Convert all to JSONL format.
    print("\nGenerating CoT-enriched training data...")

    # Agent 1: Phase 1 repos A-G.
    agent1 = [convert_gen_example(f) for f in group1]
    agent1 = [e for e in agent1 if e is not None]
    count1 = write_agent_jsonl(agent1, 1, "Phase1 A-G")

    # Agent 2: Phase 1 repos H-Q.
    agent2 = [convert_gen_example(f) for f in group2]
    agent2 = [e for e in agent2 if e is not None]
    count2 = write_agent_jsonl(agent2, 2, "Phase1 H-Q")

    # Agent 3: Phase 1 repos R-Z (includes sampled wordpress-develop).
    agent3 = [convert_gen_example(f) for f in group3]
    agent3 = [e for e in agent3 if e is not None]
    count3 = write_agent_jsonl(agent3, 3, "Phase1 R-Z")

    # Agent 4: Phase 2 judged synthetics + mutated + rejection.
    agent4_examples = synthetic_examples + mutations + rejection_examples
    agent4 = [convert_gen_example(f) for f in agent4_examples]
    agent4 = [e for e in agent4 if e is not None]
    count4 = write_agent_jsonl(agent4, 4, "Phase2 synth+mutated+rejection")

    # Agent 5: Judge training data.
    agent5 = [convert_judge_example(item) for item in judge_training]
    agent5 = [e for e in agent5 if e is not None]
    count5 = write_agent_jsonl(agent5, 5, "Judge training")

    # Combine all into final JSONL.
    all_examples = agent1 + agent2 + agent3 + agent4 + agent5
    total = len(all_examples)

    output_path = FINAL_DIR / "wordpress_finetune.jsonl"
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    gen_count = sum(1 for e in all_examples if e["metadata"]["task_type"] == "gen")
    judge_count = sum(1 for e in all_examples if e["metadata"]["task_type"] == "judge")
    rejection_count = sum(1 for e in all_examples if e["metadata"].get("source") == "rejection")

    print(f"\n{'='*60}")
    print(f"Total examples: {total}")
    print(f"  wp_gen:   {gen_count} ({gen_count/total*100:.1f}%)")
    print(f"  wp_judge: {judge_count} ({judge_count/total*100:.1f}%)")
    print(f"  rejection: {rejection_count}")
    print(f"Output: {output_path}")
    print(f"Agent files: {COT_OUTPUT}")


if __name__ == "__main__":
    main()
