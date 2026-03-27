#!/usr/bin/env python3
"""Phase 2, Step 2: Generate synthetic examples to fill coverage gaps.

Reads the gap report from phase2_gap_analysis.py and generates targeted
synthetic WordPress code using Claude. Real code from Phase 1 is used as
style anchors to ground the generation.
"""

import itertools
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import anthropic
import yaml

from scripts.utils import (
    call_with_backoff,
    load_checkpoint,
    save_checkpoint,
    batch_or_direct,
    make_batch_request,
    submit_batch,
    poll_batch,
    parse_batch_results,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAP_REPORT_PATH = PROJECT_ROOT / "data" / "phase2_synthetic" / "gap_report.json"
PASSED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "passed"
PROMPTS_PATH = PROJECT_ROOT / "config" / "synthetic_prompts.yaml"
GENERATED_DIR = PROJECT_ROOT / "data" / "phase2_synthetic" / "output" / "generated"

SYSTEM_PROMPT = """You are a senior WordPress core contributor and VIP platform engineer.
Generate production-quality PHP code that:
- Follows WordPress Coding Standards (WPCS) exactly
- Uses $wpdb->prepare() for ALL dynamic SQL — no exceptions
- Never uses extract(), eval(), or compact() on user data
- Always escapes output for the correct context (esc_html, esc_attr, esc_url, wp_kses)
- Always verifies nonces and checks capabilities on state-changing operations
- Uses proper hook priorities and argument counts
- Prefers taxonomy queries over meta queries for filterable data at scale
- Uses transients/object cache for expensive queries
- Includes PHPDoc for all functions with @param, @return, @since
- Handles edge cases (empty results, missing permissions, invalid input)

Return only the PHP code with PHPDoc. No markdown wrapping, no explanations outside code comments."""


def load_style_anchors(gap_tag: str) -> list[str]:
    """Load real code examples from Phase 1 as style reference."""
    tag_prefix = gap_tag.split(":")[0]
    anchors = []

    for passed_file in PASSED_DIR.glob("*.json"):
        with open(passed_file) as f:
            functions = json.load(f)
        for func in functions:
            tags = func.get("training_tags", [])
            if any(tag_prefix in t for t in tags):
                body = func.get("body", "")
                docblock = func.get("docblock", "")
                if docblock:
                    anchors.append(f"{docblock}\n{body}")
                else:
                    anchors.append(body)
                if len(anchors) >= 5:
                    break
        if len(anchors) >= 5:
            break

    return anchors[:3]  # Return top 3 as few-shot anchors.


def build_prompt(template: str, anchors: list[str], variation_seed: int,
                 complexities: list, contexts: list, constraints: list) -> str:
    """Build a complete generation prompt with variation."""
    complexity = complexities[variation_seed % len(complexities)]
    context = contexts[variation_seed % len(contexts)]
    constraint = constraints[variation_seed % len(constraints)]

    filled = template.format(
        complexity=complexity,
        context=context,
        constraint=constraint,
    )

    prompt = ""
    if anchors:
        prompt += "Here are real-world examples from high-quality WordPress plugins for reference style:\n\n"
        for i, anchor in enumerate(anchors):
            prompt += f"--- Example {i + 1} ---\n```php\n{anchor[:1500]}\n```\n\n"
        prompt += "---\n\n"

    prompt += f"Task: {filled}\n\n"
    prompt += f"Variation seed: {variation_seed}. Make this implementation distinct."

    return prompt


def generate_one(prompt: str, gap_tag: str, client: anthropic.Anthropic) -> dict:
    """Generate a single synthetic example using call_with_backoff."""
    # Use Opus for contrastive/reasoning, Sonnet for standard generation.
    model = ("claude-opus-4-6-20250514"
             if "contrastive" in gap_tag or "cot" in gap_tag
             else "claude-sonnet-4-6-20250514")

    response = call_with_backoff(
        client,
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "source": "synthetic",
        "gap_tag": gap_tag,
        "model": model,
        "prompt": prompt,
        "body": response.content[0].text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    if not GAP_REPORT_PATH.exists():
        print("No gap report found. Run phase2_gap_analysis.py first.")
        sys.exit(1)

    with open(GAP_REPORT_PATH) as f:
        gap_report = json.load(f)

    with open(PROMPTS_PATH) as f:
        prompt_config = yaml.safe_load(f)

    gaps = gap_report.get("gaps", {})
    if not gaps:
        print("No gaps found — Phase 1 coverage meets all minimums.")
        print("Proceeding to contrastive pair generation only.")

    complexities = prompt_config["complexities"]
    contexts = prompt_config["contexts"]
    constraints = prompt_config["constraints"]
    templates = prompt_config.get("templates", {})
    contrastive = prompt_config.get("contrastive_templates", {})

    client = anthropic.Anthropic()
    total_generated = 0
    total_tokens = 0

    # Load checkpoint for resume support.
    checkpoint = load_checkpoint("phase2_generate")
    completed_tags = set(checkpoint.get("completed", []))

    # Generate for each gap.
    for gap_tag, gap_info in sorted(gaps.items(), key=lambda x: -x[1]["deficit"]):
        if gap_tag in completed_tags:
            print(f"  [{gap_tag}] Already generated, skipping (checkpoint).")
            continue

        deficit = gap_info["deficit"]
        tag_templates = templates.get(gap_tag, [])

        if not tag_templates:
            print(f"  [{gap_tag}] No templates configured, skipping. Add to config/synthetic_prompts.yaml")
            continue

        anchors = load_style_anchors(gap_tag)
        generated = []

        print(f"\n  [{gap_tag}] Generating {deficit} examples (have {gap_info['have']}, need {gap_info['need']})...")

        route = batch_or_direct(deficit)

        if route == "batch":
            # Build batch requests.
            batch_requests = []
            for seed in range(deficit):
                template = tag_templates[seed % len(tag_templates)]
                prompt = build_prompt(template, anchors, seed, complexities, contexts, constraints)
                model = "claude-sonnet-4-6-20250514"
                batch_requests.append(
                    make_batch_request(
                        custom_id=f"{gap_tag}_seed_{seed}",
                        system=SYSTEM_PROMPT,
                        user_content=prompt,
                        model=model,
                        max_tokens=4096,
                    )
                )

            print(f"    Submitting batch of {len(batch_requests)} requests...")
            batch_id = submit_batch(client, batch_requests)
            checkpoint["batch_job_ids"].append(batch_id)
            save_checkpoint("phase2_generate", checkpoint)

            results = poll_batch(client, batch_id)
            successes, failures = parse_batch_results(results)

            for success in successes:
                generated.append({
                    "source": "synthetic",
                    "gap_tag": gap_tag,
                    "model": "claude-sonnet-4-6-20250514",
                    "body": success.get("content", ""),
                    "batch_id": batch_id,
                })
                total_tokens += 0  # Batch tokens not tracked here
                total_generated += 1

            if failures:
                print(f"    {len(failures)} batch requests failed for {gap_tag}")

        else:
            for seed in range(deficit):
                template = tag_templates[seed % len(tag_templates)]
                prompt = build_prompt(template, anchors, seed, complexities, contexts, constraints)

                try:
                    result = generate_one(prompt, gap_tag, client)
                    generated.append(result)
                    total_tokens += result["input_tokens"] + result["output_tokens"]
                    total_generated += 1

                    if (seed + 1) % 10 == 0:
                        print(f"    Generated {seed + 1}/{deficit}")

                except anthropic.APIError as e:
                    print(f"    API error at seed {seed}: {e}")
                    continue

        # Save generated examples.
        if generated:
            output_path = GENERATED_DIR / f"{gap_tag.replace(':', '_')}.json"
            with open(output_path, "w") as f:
                json.dump(generated, f, indent=2)
            print(f"    Saved {len(generated)} to {output_path.name}")

        # Checkpoint after each gap_tag.
        checkpoint["completed"].append(gap_tag)
        save_checkpoint("phase2_generate", checkpoint)

    # Generate contrastive pairs.
    print(f"\nGenerating contrastive pairs...")
    for pair_type, pair_templates in contrastive.items():
        contrastive_tag = f"contrastive:{pair_type}"
        if contrastive_tag in completed_tags:
            print(f"  [{pair_type}] Already generated, skipping (checkpoint).")
            continue

        anchors = load_style_anchors(pair_type)
        generated = []

        for seed, template in enumerate(pair_templates):
            prompt = template
            if anchors:
                prompt = "Reference style from real WordPress code:\n\n"
                prompt += f"```php\n{anchors[0][:1000]}\n```\n\n---\n\n" + template

            try:
                result = generate_one(prompt, contrastive_tag, client)
                generated.append(result)
                total_tokens += result["input_tokens"] + result["output_tokens"]
                total_generated += 1
            except anthropic.APIError as e:
                print(f"    API error on contrastive {pair_type}: {e}")
                continue

        if generated:
            output_path = GENERATED_DIR / f"contrastive_{pair_type}.json"
            with open(output_path, "w") as f:
                json.dump(generated, f, indent=2)
            print(f"  [{pair_type}] Saved {len(generated)} contrastive pairs")

        checkpoint["completed"].append(contrastive_tag)
        save_checkpoint("phase2_generate", checkpoint)

    # Generate rejection examples (~700 attempts targeting ~500 survivors).
    print(f"\nGenerating rejection examples...")
    rejection_checkpoint_key = "rejection_examples"
    if rejection_checkpoint_key not in completed_tags:
        prompts_config = yaml.safe_load(open(PROMPTS_PATH))
        rejection_templates = prompts_config.get("rejection_templates", {})
        rejection_examples = []
        target_rejection = 700  # Over-generate to account for judge rejection

        if rejection_templates:
            for template_type, type_templates in rejection_templates.items():
                per_type_target = target_rejection // len(rejection_templates)
                rejection_count = 0

                for template in itertools.cycle(type_templates):
                    if rejection_count >= per_type_target:
                        break
                    # Vary the {context} placeholder.
                    context = random.choice(prompts_config.get("contexts", ["standalone plugin"]))
                    filled = template.replace("{context}", context)

                    try:
                        response = call_with_backoff(
                            client,
                            model="claude-sonnet-4-6-20250514",
                            max_tokens=2048,
                            system=SYSTEM_PROMPT,
                            messages=[{"role": "user", "content": filled}],
                        )
                        rejection_examples.append({
                            "source": "synthetic",
                            "gap_tag": f"rejection:proactive_{template_type}",
                            "model": "claude-sonnet-4-6-20250514",
                            "body": response.content[0].text,
                            "metadata": {
                                "task_type": "gen",
                            },
                        })
                        total_tokens += response.usage.input_tokens + response.usage.output_tokens
                        total_generated += 1
                        rejection_count += 1
                    except anthropic.APIError as e:
                        print(f"    API error on rejection {template_type}: {e}")
                        rejection_count += 1  # Count as attempted to avoid infinite loop
                        continue

            if rejection_examples:
                output_path = GENERATED_DIR / "rejection_examples.json"
                with open(output_path, "w") as f:
                    json.dump(rejection_examples, f, indent=2)
                print(f"  Saved {len(rejection_examples)} rejection examples")

        checkpoint["completed"].append(rejection_checkpoint_key)
        save_checkpoint("phase2_generate", checkpoint)

    print(f"\n{'='*50}")
    print(f"Phase 2 Generation Complete")
    print(f"  Total generated: {total_generated}")
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Estimated cost: ~${total_tokens / 1_000_000 * 5:.2f}")
    print(f"\nRun phase2_judge.py next.")


if __name__ == "__main__":
    main()
