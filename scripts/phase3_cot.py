#!/usr/bin/env python3
"""Phase 3: Apply Chain-of-Thought reasoning to the entire dataset.

Takes all passed examples from Phase 1 + Phase 2 and generates:
1. Instruction-completion pairs (reverse-engineer a prompt for each code unit)
2. CoT reasoning wrappers for complex examples (architecture decisions, performance)
3. Contrastive reasoning pairs (already generated in Phase 2, enhanced here with CoT)

Final output: training-ready JSONL in data/final_dataset/
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import anthropic

from scripts.utils import call_with_backoff, load_checkpoint, save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE1_PASSED = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "passed"
PHASE2_JUDGED = PROJECT_ROOT / "data" / "phase2_synthetic" / "output" / "judged"
PHASE2_MUTATED = PROJECT_ROOT / "data" / "phase2_synthetic" / "output" / "mutated"
PHASE2_JUDGE_TRAINING = PROJECT_ROOT / "data" / "phase2_synthetic" / "output" / "judge_training"
COT_OUTPUT = PROJECT_ROOT / "data" / "phase3_cot" / "output"
FINAL_DIR = PROJECT_ROOT / "data" / "final_dataset"

SYSTEM_PROMPT = """You are a senior WordPress core contributor and VIP platform engineer.
You write production-quality PHP following WordPress Coding Standards with strict security,
performance, and API correctness. You think step-by-step about architectural decisions."""

INSTRUCTION_SYNTH_PROMPT = """Given this WordPress PHP code, write a natural instruction/task
that a developer would give to produce this code. The instruction should:
- Be specific enough to guide toward this implementation
- Mention key constraints (security, performance, WordPress APIs)
- Not be so specific that it dictates every line
- Sound like a real task assignment or Stack Overflow question

Code:
```php
{code}
```

Return only the instruction text, nothing else."""

COT_WRAPPER_PROMPT = """You are explaining your reasoning as you implement this WordPress task.

Task: {instruction}

Think step-by-step through:
1. What WordPress APIs and patterns are appropriate here
2. Security considerations (SQL injection, XSS, CSRF, capabilities)
3. Performance implications (query efficiency, caching strategy, scaling)
4. Why alternative approaches are worse for this case

Then provide the complete implementation.

The implementation must match this reference (do not change the code, explain YOUR reasoning for writing it this way):
```php
{code}
```

Format your response as:
## Reasoning
[Your step-by-step thinking]

## Implementation
```php
[The code]
```"""

# Tags that benefit from CoT reasoning (complex enough to warrant explanation).
COT_TAGS = {
    "sql:joins_across_meta",
    "sql:batch_operations",
    "sql:custom_table_creation",
    "sql:dbdelta_migrations",
    "sql:transaction_handling",
    "perf:query_caching",
    "perf:batch_processing",
    "multisite:per_site_tables",
    "arch:uninstall_cleanup",
    "arch:upgrade_routines",
    "rest:permission_callbacks",
    "security:file_upload_validation",
}


def load_all_passed() -> list[dict]:
    """Load all passed examples from Phase 1 and Phase 2."""
    all_examples = []

    # Phase 1 passed.
    for f in PHASE1_PASSED.glob("*.json"):
        with open(f) as fh:
            examples = json.load(fh)
            for ex in examples:
                ex["pipeline_source"] = "phase1_real"
            all_examples.extend(examples)

    # Phase 2 judged + passed.
    for f in PHASE2_JUDGED.glob("*_passed.json"):
        with open(f) as fh:
            examples = json.load(fh)
            for ex in examples:
                ex["pipeline_source"] = "phase2_synthetic"
            all_examples.extend(examples)

    # Phase 2 mutated contrastive pairs.
    mutations_path = PHASE2_MUTATED / "contrastive_mutations.json"
    if mutations_path.exists():
        with open(mutations_path) as fh:
            mutations = json.load(fh)
            for m in mutations:
                m["pipeline_source"] = "phase2_mutation"
            all_examples.extend(mutations)

    return all_examples


def load_judge_training() -> list[dict]:
    """Load judge training examples (already formatted as messages)."""
    judge_path = PHASE2_JUDGE_TRAINING / "judge_training.json"
    if not judge_path.exists():
        return []
    with open(judge_path) as f:
        return json.load(f)


def synthesize_instruction(code: str, client: anthropic.Anthropic) -> str:
    """Generate a natural instruction for a code example."""
    response = call_with_backoff(
        client,
        model="claude-sonnet-4-6-20250514",
        max_tokens=512,
        messages=[{"role": "user", "content": INSTRUCTION_SYNTH_PROMPT.format(code=code[:3000])}],
    )
    return response.content[0].text.strip()


def generate_cot(instruction: str, code: str, client: anthropic.Anthropic) -> str:
    """Generate chain-of-thought reasoning for a code example."""
    response = call_with_backoff(
        client,
        model="claude-opus-4-6-20250514",  # Opus for reasoning quality.
        max_tokens=6000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": COT_WRAPPER_PROMPT.format(
            instruction=instruction, code=code[:4000]
        )}],
    )
    return response.content[0].text


def needs_cot(example: dict) -> bool:
    """Determine if an example is complex enough for CoT."""
    tags = set(example.get("training_tags", []))
    # Apply CoT to examples with relevant tags or complex SQL.
    if tags & COT_TAGS:
        return True
    sql = example.get("sql_patterns", [])
    if len(sql) >= 2:  # Multiple SQL patterns = complex enough.
        return True
    if example.get("line_count", 0) > 40:  # Long functions likely have decisions worth explaining.
        return True
    return False


def format_training_example(example: dict, instruction: str, cot: str = None) -> dict:
    """Format a single training example as messages."""
    code = example.get("body", "")
    docblock = example.get("docblock", "")
    if docblock:
        code = f"{docblock}\n{code}"

    if cot:
        # CoT: the assistant response includes reasoning + code.
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": cot},
            ],
            "metadata": {
                "source": example.get("pipeline_source", "unknown"),
                "tags": example.get("training_tags", []),
                "has_cot": True,
                "source_repo": example.get("source_repo", "synthetic"),
            },
        }
    else:
        # Direct: instruction -> code.
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": f"```php\n{code}\n```"},
            ],
            "metadata": {
                "source": example.get("pipeline_source", "unknown"),
                "tags": example.get("training_tags", []),
                "has_cot": False,
                "source_repo": example.get("source_repo", "synthetic"),
            },
        }


def main():
    COT_OUTPUT.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()
    all_examples = load_all_passed()

    if not all_examples:
        print("No passed examples found. Complete Phases 1 and 2 first.")
        sys.exit(1)

    print(f"Phase 3: Chain-of-Thought Processing")
    print(f"{'='*50}")
    print(f"Total examples: {len(all_examples)}")
    print(f"Examples needing CoT: {sum(1 for ex in all_examples if needs_cot(ex))}")
    print()

    # Load checkpoint — authoritative resume pointer.
    checkpoint = load_checkpoint("phase3_cot")
    completed = set(checkpoint["completed"])
    if completed:
        print(f"  Resuming from checkpoint: {len(completed)} examples already processed")

    training_data = []
    cot_count = 0
    direct_count = 0

    for i, example in enumerate(all_examples):
        example_id = example.get("id") or example.get("function_name") or str(i)

        # Skip already-processed examples.
        if example_id in completed:
            continue

        # Handle mutated contrastive pairs differently.
        if example.get("pipeline_source") == "phase2_mutation":
            bad_code = example.get("bad_code", "")
            good_code = example.get("good_code", "")
            violation = example.get("violation_description", "")
            if bad_code and good_code:
                # Generate CoT explanation for the contrastive pair.
                contrastive_prompt = (
                    f"This code has a specific defect: {violation}\n\n"
                    f"Bad version:\n```php\n{bad_code[:2000]}\n```\n\n"
                    f"Corrected version:\n```php\n{good_code[:2000]}\n```\n\n"
                    f"Explain step-by-step what is wrong with the bad version, "
                    f"why it matters, and how the corrected version fixes it."
                )
                try:
                    cot_response = call_with_backoff(
                        client,
                        model="claude-sonnet-4-6-20250514",
                        max_tokens=2000,
                        system=SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": contrastive_prompt}],
                    )
                    cot_text = cot_response.content[0].text
                except anthropic.APIError:
                    cot_text = f"Defect: {violation}"

                training_data.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"Review this WordPress code and identify the security/quality "
                            f"issue, then show the corrected version:\n\n```php\n{bad_code[:2000]}\n```"
                        )},
                        {"role": "assistant", "content": (
                            f"{cot_text}\n\n## Corrected Code\n```php\n{good_code[:2000]}\n```"
                        )},
                    ],
                    "metadata": {
                        "source": "phase2_mutation",
                        "tags": example.get("training_tags", []),
                        "has_cot": True,
                        "mutation_type": example.get("mutation_type", "unknown"),
                    },
                })
                cot_count += 1
            completed.add(example_id)
            # Save checkpoint every 100 examples.
            if len(completed) % 100 == 0:
                save_checkpoint("phase3_cot", {
                    "completed": list(completed),
                    "failed": checkpoint["failed"],
                    "batch_job_ids": [],
                })
            continue

        code = example.get("body", "")
        if not code or len(code) < 50:
            completed.add(example_id)
            continue

        # For synthetic examples that already have prompts, use them.
        if example.get("pipeline_source") == "phase2_synthetic" and example.get("prompt"):
            instruction = example["prompt"]
        else:
            # Synthesize an instruction for real code.
            instruction = synthesize_instruction(code, client)

        if needs_cot(example):
            # Generate chain-of-thought reasoning.
            cot = generate_cot(instruction, code, client)
            training_example = format_training_example(example, instruction, cot=cot)
            cot_count += 1
        else:
            training_example = format_training_example(example, instruction)
            direct_count += 1

        training_data.append(training_example)
        completed.add(example_id)

        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(all_examples)} "
                  f"(CoT: {cot_count}, Direct: {direct_count})")

        # Checkpoint every 500 examples — progress JSONL (useful for recovery).
        if len(completed) % 500 == 0:
            checkpoint_path = COT_OUTPUT / f"checkpoint_{len(completed)}.jsonl"
            with open(checkpoint_path, "w") as f:
                for td in training_data:
                    f.write(json.dumps(td) + "\n")
            print(f"  Progress file saved: {checkpoint_path}")

        # Save utils.py checkpoint every 100 examples (authoritative resume pointer).
        if len(completed) % 100 == 0:
            save_checkpoint("phase3_cot", {
                "completed": list(completed),
                "failed": checkpoint["failed"],
                "batch_job_ids": [],
            })

    # Final save of utils.py checkpoint after processing all examples.
    save_checkpoint("phase3_cot", {
        "completed": list(completed),
        "failed": checkpoint["failed"],
        "batch_job_ids": [],
    })

    # Merge judge training data (<wp_judge> examples).
    judge_data = load_judge_training()
    judge_count = len(judge_data)
    if judge_data:
        training_data.extend(judge_data)
        print(f"\n  Merged {judge_count} judge training examples")

    mutation_count = sum(
        1 for td in training_data if td.get("metadata", {}).get("source") == "phase2_mutation"
    )

    # Write final dataset.
    final_path = FINAL_DIR / "wordpress_finetune.jsonl"
    with open(final_path, "w") as f:
        for td in training_data:
            f.write(json.dumps(td) + "\n")

    # Write metadata.
    metadata = {
        "total_examples": len(training_data),
        "cot_examples": cot_count,
        "direct_examples": direct_count,
        "judge_examples": judge_count,
        "mutation_contrastive": mutation_count,
        "phase1_real": sum(1 for td in training_data if td["metadata"]["source"] == "phase1_real"),
        "phase2_synthetic": sum(1 for td in training_data if td["metadata"]["source"] == "phase2_synthetic"),
        "phase2_mutation": mutation_count,
        "phase2_judge": judge_count,
        "unique_tags": list(set(
            tag for td in training_data for tag in td["metadata"].get("tags", [])
        )),
    }

    with open(FINAL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Phase 3 Complete")
    print(f"  Total training examples: {len(training_data)}")
    print(f"  With CoT reasoning: {cot_count}")
    print(f"  Direct (code only): {direct_count}")
    print(f"  Judge (<wp_judge>): {judge_count}")
    print(f"  Mutation contrastive: {mutation_count}")
    print(f"  From real code: {metadata['phase1_real']}")
    print(f"  From synthetic: {metadata['phase2_synthetic']}")
    print(f"\nFinal dataset: {final_path}")
    print(f"Metadata: {FINAL_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
