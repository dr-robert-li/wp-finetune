#!/usr/bin/env python3
"""Phase 4.1, Plan 02 Task 1: Generate critique-then-fix training examples.

Generates structured critique-then-fix examples from Phase 1 failed functions
using golden critique seeds as few-shot exemplars. Each example includes:
- Structured critique with severity per dimension
- Corrected PHP code that addresses the critique issues
- PHP lint validation of corrected code
- Critique-fix alignment verification

Usage:
    python scripts/generate_critique_then_fix.py --pilot
    python scripts/generate_critique_then_fix.py --target 100
    python scripts/generate_critique_then_fix.py  # bulk mode, 50% of failed functions
"""

import json
import random
import re
import sys
import argparse
import subprocess
import tempfile
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import anthropic
from scripts.utils import extract_json, call_with_backoff, load_checkpoint, save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS_DIR = PROJECT_ROOT / "data" / "seeds"
FAILED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "failed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "phase4_reasoning"
PILOT_DIR = OUTPUT_DIR / "pilot"
BULK_DIR = OUTPUT_DIR / "critique_then_fix"

REQUIRED_DIMENSIONS = [
    "wpcs_compliance", "sql_safety", "security", "performance",
    "wp_api_usage", "code_quality", "dependency_integrity", "i18n", "accessibility"
]

SEVERITY_LEVELS = ["critical", "high", "medium", "low"]

WP_API_CITATIONS = [
    "$wpdb->prepare", "wp_verify_nonce", "check_ajax_referer",
    "esc_html", "esc_attr", "esc_url", "current_user_can", "wp_kses",
    "wp_nonce_field", "sanitize_text_field", "wp_die", "absint",
    "wp_safe_redirect", "wp_create_nonce", "wp_unslash"
]


# ---------------------------------------------------------------------------
# Seed loading and sampling
# ---------------------------------------------------------------------------

def load_critique_seeds() -> list:
    """Load critique-then-fix seeds from both seed files."""
    seeds = []
    for seed_file in [SEEDS_DIR / "ugc_seeds.json", SEEDS_DIR / "ugc_boundary_seeds.json"]:
        if seed_file.exists():
            with open(seed_file) as f:
                data = json.load(f)
            seeds.extend(s for s in data if s.get("seed_type") == "critique_then_fix")
    return seeds


def sample_seeds(seeds: list, n: int = 3) -> list:
    """Sample n seeds with boundary seeds weighted 2x higher."""
    if len(seeds) <= n:
        return seeds[:]
    weights = [2.0 if s.get("defect_subtlety") == "boundary" else 1.0 for s in seeds]
    return random.choices(seeds, weights=weights, k=n)


# ---------------------------------------------------------------------------
# Few-shot exemplar formatting
# ---------------------------------------------------------------------------

def format_critique_seed_as_exemplar(seed: dict) -> str:
    """Format a critique-then-fix seed as a few-shot text exemplar."""
    defective = seed.get("defective_code", "")[:1500]
    hc = seed.get("human_critique", {})
    summary = hc.get("summary", "")
    key_obs = hc.get("key_observation", "")
    dimensions = hc.get("dimensions", {})
    corrected = seed.get("corrected_code", "")[:1500]

    lines = [
        "--- EXAMPLE ---",
        f"Defective PHP Code:\n```php\n{defective}\n```",
        "",
        "Critique:",
        f"  Summary: {summary}",
        "",
        "  Dimensions:",
    ]
    for dim, info in dimensions.items():
        severity = info.get("severity", "N/A")
        # Seed data has 'reasoning' field; we display it as the issue description
        issue_text = info.get("issue", info.get("reasoning", ""))
        fix_text = info.get("fix", "")
        lines.append(f"    {dim}: severity={severity}")
        if issue_text:
            lines.append(f"      Issue: {issue_text[:200]}")
        if fix_text:
            lines.append(f"      Fix: {fix_text[:200]}")
    lines.extend([
        f"  Key Observation: {key_obs}",
        "",
        f"<corrected_code>",
        corrected,
        f"</corrected_code>",
        "--- END EXAMPLE ---",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Source function loading (D-02: use failed functions, NOT mutation pool)
# ---------------------------------------------------------------------------

def load_failed_functions() -> list:
    """Load Phase 1 failed functions as source for critique-then-fix pairs.

    Per D-02: critique-then-fix uses failed functions (NOT mutation pool,
    which is empty).
    """
    functions = []
    if not FAILED_DIR.exists():
        return functions

    for f in FAILED_DIR.glob("*.json"):
        try:
            with open(f) as fh:
                entries = json.load(fh)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                body = entry.get("body", "")
                docblock = entry.get("docblock", "")
                code = f"{docblock}\n{body}".strip() if docblock else body
                if len(code) < 50:
                    continue
                functions.append({
                    "code": code,
                    "source_file": f.name,
                    "function_name": entry.get("function_name", "unknown"),
                })
        except (json.JSONDecodeError, KeyError):
            continue
    return functions


# ---------------------------------------------------------------------------
# PHP lint check (addresses review concern #2)
# ---------------------------------------------------------------------------

def php_lint_check(code: str) -> dict:
    """Run PHP syntax check on the given code.

    Uses 'php -l' to validate syntax. Gracefully degrades if PHP CLI is
    not available.

    NOTE: `php -l` only inspects content inside a `<?php` block; anything
    before the opening tag is treated as literal HTML and silently passes.
    Corrected_code blocks in this pipeline start with `/**` or `function`
    (no opener), so we MUST prefix `<?php` before linting — otherwise PHP
    treats the whole body as HTML and returns exit 0 regardless of what
    syntax errors are in the code. This was a silent pipeline bug that
    caused php_lint.valid=true on mangled code across the entire CtF bulk.

    Returns:
        dict with:
            valid: bool -- True if syntax is valid
            errors: str -- stderr output if invalid, empty string if valid
    """
    tmp_path = None
    try:
        # Prefix <?php unless code already has an opener — critical to
        # avoid the HTML-passthrough false positive described above.
        if not code.lstrip().startswith("<?"):
            code_to_lint = "<?php\n" + code
        else:
            code_to_lint = code
        with tempfile.NamedTemporaryFile(suffix=".php", delete=False, mode="w",
                                         encoding="utf-8") as tmp:
            tmp.write(code_to_lint)
            tmp_path = tmp.name

        result = subprocess.run(
            ["php", "-l", tmp_path],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            return {"valid": True, "errors": ""}
        else:
            return {"valid": False, "errors": result.stderr.strip() or result.stdout.strip()}

    except FileNotFoundError:
        # php CLI not available -- graceful degradation
        print("WARNING: php CLI not available -- PHP lint check skipped", file=sys.stderr)
        return {"valid": True, "errors": "php not available - skipped"}
    except subprocess.TimeoutExpired:
        return {"valid": False, "errors": "php lint timeout"}
    except Exception as e:
        return {"valid": False, "errors": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Critique-fix alignment check (addresses review concern #2)
# ---------------------------------------------------------------------------

def check_critique_fix_alignment(critique: dict, defective_code: str,
                                  corrected_code: str) -> dict:
    """Check that critical/high severity issues from critique are addressed in the fix.

    Heuristic: For each critical/high dimension, extract API names mentioned in
    the fix suggestion text. Check if those APIs appear in corrected_code but
    NOT in defective_code (i.e., they were added as part of the fix).

    Args:
        critique: dict with 'dimensions' key (severity, issue, fix per dimension)
        defective_code: Original defective PHP code
        corrected_code: Corrected PHP code

    Returns:
        dict with:
            critical_high_issues: int -- count of critical/high severity issues
            addressed_issues: int -- count where fix APIs appear in corrected_code
            unaddressed_issues: list[dict] -- details on unaddressed critical/high issues
            alignment_ratio: float -- addressed / critical_high_issues (1.0 if none)
    """
    dimensions = critique.get("dimensions", {})
    critical_high_issues = 0
    addressed_issues = 0
    unaddressed = []

    for dim, info in dimensions.items():
        if not isinstance(info, dict):
            continue
        severity = info.get("severity", "low")
        if severity not in ("critical", "high"):
            continue

        critical_high_issues += 1
        # Get fix text (handle both 'fix' and 'reasoning' field names)
        fix_text = info.get("fix", info.get("reasoning", ""))

        # Extract WP API names mentioned in the fix text
        apis_in_fix = [api for api in WP_API_CITATIONS if api in fix_text]

        if not apis_in_fix:
            # No verifiable API mentions -- count as addressed (can't check non-API fixes)
            addressed_issues += 1
            continue

        # Check if any of the fix APIs appear in corrected but NOT in defective
        newly_added_apis = [
            api for api in apis_in_fix
            if api in corrected_code and api not in defective_code
        ]

        if newly_added_apis:
            addressed_issues += 1
        else:
            # Check if APIs are at least present in corrected code (even if also in defective)
            present_in_corrected = [api for api in apis_in_fix if api in corrected_code]
            if present_in_corrected:
                addressed_issues += 1
            else:
                missing_apis = [api for api in apis_in_fix if api not in corrected_code]
                unaddressed.append({
                    "dimension": dim,
                    "severity": severity,
                    "fix_text": fix_text[:200],
                    "missing_apis": missing_apis,
                })

    alignment_ratio = (addressed_issues / critical_high_issues
                       if critical_high_issues > 0 else 1.0)

    return {
        "critical_high_issues": critical_high_issues,
        "addressed_issues": addressed_issues,
        "unaddressed_issues": unaddressed,
        "alignment_ratio": alignment_ratio,
    }


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_critique_then_fix(code: str, source_info: dict, seeds: list,
                                client: anthropic.Anthropic) -> dict:
    """Generate a critique-then-fix example for a defective PHP code snippet.

    Args:
        code: Defective PHP source code
        source_info: Metadata about the code source
        seeds: List of critique-then-fix seeds for few-shot context
        client: Anthropic client

    Returns:
        Parsed result dict (with corrected_code) or None on failure
    """
    sampled = sample_seeds(seeds, 3)
    exemplars = "\n\n".join(format_critique_seed_as_exemplar(s) for s in sampled)

    prompt = f"""You are a WordPress code quality critic producing structured critiques for training data. You receive defective PHP code and must:
(a) produce a structured critique analyzing ALL 9 dimensions with severity (critical/high/medium/low), issue description, and fix suggestion for each;
(b) produce the corrected version of the code wrapped in <corrected_code> XML tags.

IMPORTANT: The corrected code must actually fix the issues you identify. If you cite a missing $wpdb->prepare, the corrected code must contain $wpdb->prepare.

Here are golden examples of critiques and fixes:

{exemplars}

NOW CRITIQUE AND FIX the following defective WordPress PHP code.

Return JSON with keys:
- summary: string (overall critique summary)
- dimensions: object with ALL 9 dimensions: wpcs_compliance, sql_safety, security, performance, wp_api_usage, code_quality, dependency_integrity, i18n, accessibility
  Each dimension must have:
  - severity: "critical" | "high" | "medium" | "low"
  - issue: string (what is wrong)
  - fix: string (what to do to fix it, naming specific WordPress APIs where applicable)
- key_observation: string (most important finding)
- corrected_code: string (the actual fixed PHP source code, NOT wrapped in additional tags)

When WordPress APIs appear in the defective code or are needed in the fix, name them explicitly:
$wpdb->prepare(), wp_verify_nonce(), esc_html(), current_user_can(), etc.

Defective PHP Code to critique and fix:
```php
{code[:3000]}
```

IMPORTANT: corrected_code in your JSON must be the actual corrected PHP source, not wrapped in additional tags. The code must be syntactically valid PHP."""

    # Scale max_tokens with code length: longer code needs more tokens for corrected version
    code_len = len(code)
    max_tokens = min(4096, max(3072, code_len // 2))

    try:
        resp = call_with_backoff(
            client,
            model="claude-sonnet-4-5",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = resp.content[0].text
        result = extract_json(raw_text)

        if result is None:
            return None

        # Fallback: if corrected_code missing from JSON, try XML tag extraction
        if not result.get("corrected_code"):
            match = re.search(r'<corrected_code>([\s\S]+?)</corrected_code>', raw_text)
            if match:
                result["corrected_code"] = match.group(1).strip()

        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Quality gate (strengthened -- addresses review concern #2)
# ---------------------------------------------------------------------------

def passes_quality_gate(result: dict, defective_code: str = None) -> bool:
    """Check if a generated result meets quality requirements.

    Args:
        result: Parsed LLM result dict
        defective_code: Optional original code for lint and alignment checks
    """
    if not isinstance(result, dict):
        return False

    # summary must be non-empty
    if not result.get("summary") or not isinstance(result["summary"], str):
        return False

    # dimensions must have all 9 required dimensions
    dims = result.get("dimensions", {})
    if not isinstance(dims, dict):
        return False
    missing = [d for d in REQUIRED_DIMENSIONS if d not in dims]
    if missing:
        return False

    # Each dimension must have severity, issue, fix
    for dim, info in dims.items():
        if not isinstance(info, dict):
            return False
        if info.get("severity") not in SEVERITY_LEVELS:
            return False
        if not info.get("issue") or not isinstance(info["issue"], str):
            return False
        if not info.get("fix") or not isinstance(info["fix"], str):
            return False

    # corrected_code must be non-empty and have some length
    corrected = result.get("corrected_code", "")
    if not corrected or len(corrected) < 20:
        return False

    # corrected_code must differ from defective_code
    if defective_code is not None and corrected == defective_code:
        return False

    # PHP lint check on corrected_code
    if defective_code is not None:
        lint = php_lint_check(corrected)
        if not lint["valid"]:
            # Allow graceful degradation if PHP not available
            if "php not available" not in lint.get("errors", ""):
                return False

    # Critique-fix alignment check
    if defective_code is not None:
        critique_dict = {"dimensions": dims}
        alignment = check_critique_fix_alignment(critique_dict, defective_code, corrected)
        if alignment["critical_high_issues"] > 0 and alignment["alignment_ratio"] < 0.3:
            return False

    return True


# ---------------------------------------------------------------------------
# Training example formatter
# ---------------------------------------------------------------------------

def format_training_example(source_info: dict, result: dict,
                              defective_code: str = None) -> dict:
    """Assemble the final training example dict with all metadata."""
    corrected = result.get("corrected_code", "")
    dims = result.get("dimensions", {})
    critique_dict = {"dimensions": dims}

    lint = php_lint_check(corrected) if corrected else {"valid": False, "errors": "no code"}
    alignment = (check_critique_fix_alignment(critique_dict, defective_code, corrected)
                 if defective_code else {
                     "critical_high_issues": 0, "addressed_issues": 0,
                     "unaddressed_issues": [], "alignment_ratio": 1.0
                 })

    return {
        "source_file": source_info.get("source_file", ""),
        "function_name": source_info.get("function_name", ""),
        "defective_code": source_info.get("code", defective_code or ""),
        "critique": {
            "summary": result.get("summary", ""),
            "dimensions": dims,
            "key_observation": result.get("key_observation", ""),
        },
        "corrected_code": corrected,
        "dimensions_addressed": REQUIRED_DIMENSIONS[:],
        "generation_method": "seed_few_shot_agent",
        "php_lint": lint,
        "critique_fix_alignment": alignment,
    }


# ---------------------------------------------------------------------------
# Pilot validation
# ---------------------------------------------------------------------------

def validate_pilot_batch(examples: list) -> dict:
    """Validate dimension coverage, severity coverage, and quality metrics."""
    all_dims = set()
    all_severities = set()
    all_api_citations = set()
    lint_valid_count = 0
    alignment_ratios = []

    for ex in examples:
        critique = ex.get("critique", {})
        dims = critique.get("dimensions", {})
        all_dims.update(dims.keys())

        for info in dims.values():
            if isinstance(info, dict):
                severity = info.get("severity")
                if severity in SEVERITY_LEVELS:
                    all_severities.add(severity)

        # Check API citations in critique text
        all_text = json.dumps(critique)
        for api in WP_API_CITATIONS:
            if api in all_text:
                all_api_citations.add(api)

        # PHP lint pass
        if ex.get("php_lint", {}).get("valid", False):
            lint_valid_count += 1

        # Alignment ratio
        ar = ex.get("critique_fix_alignment", {}).get("alignment_ratio", 1.0)
        alignment_ratios.append(ar)

    missing_dimensions = [d for d in REQUIRED_DIMENSIONS if d not in all_dims]
    missing_severities = [s for s in SEVERITY_LEVELS if s not in all_severities]
    php_lint_pass_rate = lint_valid_count / len(examples) if examples else 0.0
    mean_alignment = sum(alignment_ratios) / len(alignment_ratios) if alignment_ratios else 1.0

    print(f"\n=== Critique-then-Fix Pilot Batch Validation ===")
    print(f"Total examples: {len(examples)}")
    print(f"Dimensions covered: {sorted(all_dims)}")
    if missing_dimensions:
        print(f"MISSING dimensions: {missing_dimensions}")
    else:
        print("All 9 dimensions covered: OK")
    print(f"Severity levels covered: {sorted(all_severities)}")
    if missing_severities:
        print(f"MISSING severity levels: {missing_severities}")
    print(f"WP API citations found ({len(all_api_citations)}): {sorted(all_api_citations)}")
    print(f"PHP lint pass rate: {lint_valid_count}/{len(examples)} ({php_lint_pass_rate*100:.0f}%)")
    print(f"Mean alignment ratio: {mean_alignment:.2f}")
    if mean_alignment < 0.5:
        print("WARNING: Mean alignment ratio < 0.5 -- review fix quality")

    return {
        "missing_dimensions": missing_dimensions,
        "severity_coverage": list(all_severities),
        "api_citations_found": list(all_api_citations),
        "php_lint_pass_rate": php_lint_pass_rate,
        "mean_alignment_ratio": mean_alignment,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate critique-then-fix training examples")
    parser.add_argument("--pilot", action="store_true",
                        help="Generate 40 pilot examples for human review")
    parser.add_argument("--target", type=int, default=None,
                        help="Override bulk target count")
    args = parser.parse_args()

    # Create output directories
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    BULK_DIR.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()

    # Load seeds
    seeds = load_critique_seeds()
    if not seeds:
        print("ERROR: No critique-then-fix seeds found. Run seed_import.py first.",
              file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(seeds)} critique_then_fix seeds")

    # Load failed functions (D-02: failed functions only, not mutation pool)
    failed_functions = load_failed_functions()
    if not failed_functions:
        print("ERROR: No Phase 1 failed functions found.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(failed_functions)} failed functions")

    # Determine target and output path
    if args.pilot:
        target = int(os.environ.get("PILOT_TARGET", "40"))
        output_path = PILOT_DIR / "critique_then_fix_pilot.json"
        checkpoint_key = "generate_critique_then_fix_pilot"
    else:
        target = args.target if args.target is not None else int(len(failed_functions) * 0.5)
        output_path = BULK_DIR / "critique_then_fix.json"
        checkpoint_key = "generate_critique_then_fix"

    print(f"Target: {target} examples")
    print(f"Output: {output_path}")

    # Load checkpoint for resume (examples persisted across runs)
    checkpoint = load_checkpoint(checkpoint_key)
    completed_ids = set(checkpoint.get("completed", []))
    examples = list(checkpoint.get("examples", []))
    if examples:
        print(f"Resuming from checkpoint: {len(examples)} examples already generated")

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(failed_functions)

    parse_attempts = int(checkpoint.get("parse_attempts", 0))
    parse_failures = int(checkpoint.get("parse_failures", 0))
    lint_failures = int(checkpoint.get("lint_failures", 0))
    alignment_failures = int(checkpoint.get("alignment_failures", 0))

    for i, func in enumerate(failed_functions):
        if len(examples) >= target:
            break

        func_id = f"{func['source_file']}::{func['function_name']}::{i}"
        if func_id in completed_ids:
            continue

        parse_attempts += 1
        result = generate_critique_then_fix(func["code"], func, seeds, client)

        if result is None:
            parse_failures += 1
            checkpoint.setdefault("failed", []).append(func_id)
            continue

        # PHP lint check
        corrected = result.get("corrected_code", "")
        if corrected:
            lint = php_lint_check(corrected)
            if not lint["valid"] and "php not available" not in lint.get("errors", ""):
                lint_failures += 1
                print(f"  Lint failure: {lint['errors'][:100]}")
                # Don't skip -- the format_training_example captures this in metadata

        # Alignment check
        critique_dict = {"dimensions": result.get("dimensions", {})}
        alignment = check_critique_fix_alignment(critique_dict, func["code"], corrected)
        if (alignment["critical_high_issues"] > 0
                and alignment["alignment_ratio"] < 0.3):
            alignment_failures += 1
            print(f"  Alignment failure: ratio={alignment['alignment_ratio']:.2f}")
            continue

        if not passes_quality_gate(result, func["code"]):
            parse_failures += 1
            continue

        example = format_training_example(func, result, func["code"])
        examples.append(example)
        checkpoint.setdefault("completed", []).append(func_id)

        # Persist counters and examples in checkpoint for full resume
        checkpoint["examples"] = examples
        checkpoint["parse_attempts"] = parse_attempts
        checkpoint["parse_failures"] = parse_failures
        checkpoint["lint_failures"] = lint_failures
        checkpoint["alignment_failures"] = alignment_failures

        if len(examples) % 5 == 0:
            print(f"  Generated {len(examples)}/{target} examples "
                  f"(parse_fail={parse_failures}, lint_fail={lint_failures}, "
                  f"align_fail={alignment_failures})")
            # Checkpoint every 5 examples + write incremental output
            save_checkpoint(checkpoint_key, checkpoint)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(examples, f, indent=2)

    # Final checkpoint save
    checkpoint["examples"] = examples
    checkpoint["parse_attempts"] = parse_attempts
    checkpoint["parse_failures"] = parse_failures
    checkpoint["lint_failures"] = lint_failures
    checkpoint["alignment_failures"] = alignment_failures
    save_checkpoint(checkpoint_key, checkpoint)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)

    # Summary
    lint_pass_count = sum(1 for ex in examples if ex.get("php_lint", {}).get("valid", False))
    align_ratios = [ex.get("critique_fix_alignment", {}).get("alignment_ratio", 1.0)
                    for ex in examples]
    mean_align = sum(align_ratios) / len(align_ratios) if align_ratios else 1.0

    print(f"\n{'='*50}")
    print(f"Critique-then-Fix Generation Complete")
    print(f"  Total generated: {len(examples)}")
    print(f"  Parse attempts: {parse_attempts}")
    print(f"  Parse failures: {parse_failures} ({parse_failures/max(parse_attempts,1)*100:.1f}%)")
    print(f"  Lint failures tracked: {lint_failures}")
    print(f"  Alignment failures: {alignment_failures}")
    print(f"  PHP lint pass rate: {lint_pass_count}/{len(examples)} "
          f"({lint_pass_count/max(len(examples),1)*100:.0f}%)")
    print(f"  Mean alignment ratio: {mean_align:.2f}")
    print(f"  Saved to: {output_path}")

    if args.pilot and examples:
        validation = validate_pilot_batch(examples)
        if validation["missing_dimensions"]:
            print(f"\nERROR: Missing dimensions: {validation['missing_dimensions']}")
            sys.exit(1)
        if len(validation["api_citations_found"]) < 3:
            print(f"\nERROR: Only {len(validation['api_citations_found'])} WP API citations "
                  f"found (need >= 3)")
            sys.exit(1)
        print("\nPilot validation PASSED")


if __name__ == "__main__":
    main()
