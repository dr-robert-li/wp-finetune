#!/usr/bin/env python3
"""Phase 2, Step 1b: Generate contrastive pairs via automated mutation.

Takes passed functions from Phase 1 and programmatically introduces
controlled violations to create bad->good training pairs. Mutations
are verified: the bad version must fail PHPCS or security checks,
and the good version is the original (already passed).

Run AFTER phase2_gap_analysis.py and BEFORE phase2_generate.py.
These contrastive pairs supplement the Claude-generated contrastive
examples in phase2_generate.py.
"""

import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PASSED_DIR = PROJECT_ROOT / "phase1_extraction" / "output" / "passed"
MUTATED_DIR = PROJECT_ROOT / "phase2_synthetic" / "output" / "mutated"


# ─── Mutation strategies ────────────────────────────────

def mutate_remove_prepare(code: str) -> tuple[str, str]:
    """Remove $wpdb->prepare() and inline values directly (SQL injection)."""
    # Match: $wpdb->prepare( "SELECT ... WHERE id = %d", $var )
    pattern = r'\$wpdb->prepare\(\s*(["\'])(.*?)\1\s*,\s*(.*?)\)'
    match = re.search(pattern, code, re.DOTALL)
    if not match:
        return None, None

    query_str = match.group(2)
    vars_str = match.group(3)

    # Replace placeholders with direct variable interpolation.
    vars_list = [v.strip() for v in vars_str.split(",")]
    mutated_query = query_str
    for var in vars_list:
        mutated_query = re.sub(r'%[sdf]', f"' . {var} . '", mutated_query, count=1)

    bad_code = code[:match.start()] + f'"{mutated_query}"' + code[match.end():]
    return bad_code, "sql_injection: Removed $wpdb->prepare(), concatenating variables directly into SQL"


def mutate_remove_nonce(code: str) -> tuple[str, str]:
    """Remove nonce verification from form handlers."""
    nonce_patterns = [
        r'if\s*\(\s*!\s*wp_verify_nonce\s*\(.*?\)\s*\)\s*\{[^}]*\}\s*',
        r'if\s*\(\s*!\s*isset\s*\(\s*\$_POST\[.+?\]\s*\)\s*\|\|\s*!\s*wp_verify_nonce.*?\{[^}]*\}\s*',
        r'check_ajax_referer\s*\([^)]+\)\s*;\s*',
        r'wp_verify_nonce\s*\([^)]+\)\s*;\s*',
    ]
    for pattern in nonce_patterns:
        match = re.search(pattern, code, re.DOTALL)
        if match:
            bad_code = code[:match.start()] + code[match.end():]
            return bad_code, "csrf: Removed nonce verification from form/AJAX handler"
    return None, None


def mutate_remove_escaping(code: str) -> tuple[str, str]:
    """Remove output escaping functions, leaving raw output."""
    esc_funcs = ["esc_html", "esc_attr", "esc_url", "esc_textarea", "wp_kses_post", "wp_kses"]
    mutated = code
    found = False
    for func in esc_funcs:
        # Match: esc_html( $var ) -> $var
        pattern = rf'{func}\s*\(\s*([^)]+)\s*\)'
        if re.search(pattern, mutated):
            mutated = re.sub(pattern, r'\1', mutated)
            found = True
    if found:
        return mutated, "xss: Removed output escaping functions, raw variables in HTML output"
    return None, None


def mutate_remove_capability_check(code: str) -> tuple[str, str]:
    """Remove capability/permission checks."""
    patterns = [
        r'if\s*\(\s*!\s*current_user_can\s*\(.*?\)\s*\)\s*\{[^}]*\}\s*',
        r'if\s*\(\s*!\s*current_user_can\s*\(.*?\)\s*\)\s*\{\s*return[^}]*\}\s*',
    ]
    for pattern in patterns:
        match = re.search(pattern, code, re.DOTALL)
        if match:
            bad_code = code[:match.start()] + code[match.end():]
            return bad_code, "authorization: Removed current_user_can() capability check"
    return None, None


def mutate_remove_sanitization(code: str) -> tuple[str, str]:
    """Remove input sanitization, use raw $_POST/$_GET values."""
    sanitize_funcs = [
        "sanitize_text_field", "sanitize_email", "sanitize_title",
        "sanitize_file_name", "sanitize_key", "absint", "intval",
        "wp_kses", "wp_kses_post",
    ]
    mutated = code
    found = False
    for func in sanitize_funcs:
        pattern = rf'{func}\s*\(\s*([^)]+)\s*\)'
        if re.search(pattern, mutated):
            mutated = re.sub(pattern, r'\1', mutated)
            found = True
    if found:
        return mutated, "input_validation: Removed sanitization functions, using raw user input"
    return None, None


def mutate_remove_i18n(code: str) -> tuple[str, str]:
    """Strip translation wrappers, leaving hardcoded English strings."""
    i18n_funcs = [
        (r"esc_html__\(\s*'([^']+)'\s*,\s*'[^']+'\s*\)", r"'\1'"),
        (r"esc_html_e\(\s*'([^']+)'\s*,\s*'[^']+'\s*\)", r"echo '\1'"),
        (r"esc_attr__\(\s*'([^']+)'\s*,\s*'[^']+'\s*\)", r"'\1'"),
        (r"__\(\s*'([^']+)'\s*,\s*'[^']+'\s*\)", r"'\1'"),
        (r"_e\(\s*'([^']+)'\s*,\s*'[^']+'\s*\)", r"echo '\1'"),
    ]
    mutated = code
    found = False
    for pattern, replacement in i18n_funcs:
        if re.search(pattern, mutated):
            mutated = re.sub(pattern, replacement, mutated)
            found = True
    if found:
        return mutated, "i18n: Removed translation wrappers, hardcoded English strings"
    return None, None


def mutate_select_star(code: str) -> tuple[str, str]:
    """Replace targeted SELECT columns with SELECT *."""
    pattern = r'(SELECT\s+)(\w+(?:\.\w+)?(?:\s*,\s*\w+(?:\.\w+)?)+)(\s+FROM)'
    match = re.search(pattern, code, re.IGNORECASE)
    if match:
        bad_code = code[:match.start()] + match.group(1) + "*" + match.group(3) + code[match.end():]
        return bad_code, "performance: Replaced targeted column SELECT with SELECT *, wasting memory and bandwidth"
    return None, None


MUTATIONS = [
    ("sql_injection", mutate_remove_prepare),
    ("csrf", mutate_remove_nonce),
    ("xss", mutate_remove_escaping),
    ("authorization", mutate_remove_capability_check),
    ("input_validation", mutate_remove_sanitization),
    ("i18n", mutate_remove_i18n),
    ("performance", mutate_select_star),
]


def verify_mutation_detectable(bad_code: str) -> bool:
    """Verify the mutation is detectable by PHPCS or simple security checks."""
    if not bad_code.strip().startswith("<?php"):
        bad_code = f"<?php\n{bad_code}"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".php", delete=False) as f:
            f.write(bad_code)
            tmp_path = f.name
        result = subprocess.run(
            ["phpcs", "--standard=WordPress-Extra", "--report=json", tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        report = json.loads(result.stdout)
        file_report = list(report["files"].values())[0]
        # Mutation should introduce at least 1 error.
        return file_report["errors"] > 0
    except subprocess.TimeoutExpired:
        return False
    except json.JSONDecodeError:
        return False
    except FileNotFoundError:
        print("ERROR: PHPCS disappeared mid-run", file=sys.stderr)
        sys.exit(1)
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


def _require_phpcs():
    """Exit hard if PHPCS is not available."""
    try:
        result = subprocess.run(
            ["phpcs", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print("ERROR: phpcs returned non-zero. Mutation detection requires PHPCS.", file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError:
        print(
            "ERROR: phpcs not found. Mutation detection requires PHPCS. "
            "Install via: composer global require squizlabs/php_codesniffer",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    _require_phpcs()
    MUTATED_DIR.mkdir(parents=True, exist_ok=True)

    passed_files = list(PASSED_DIR.glob("*.json"))
    if not passed_files:
        print("No passed functions found. Run Phase 1 first.")
        sys.exit(1)

    # Load all passed functions.
    all_passed = []
    for f in passed_files:
        with open(f) as fh:
            functions = json.load(fh)
            # Skip core — only mutate assessed code for cleaner pairs.
            all_passed.extend(fn for fn in functions if fn.get("quality_tier") != "core")

    print(f"Loaded {len(all_passed)} assessed+passed functions for mutation")

    random.seed(42)
    random.shuffle(all_passed)

    contrastive_pairs = []
    mutation_counts = {name: 0 for name, _ in MUTATIONS}

    for func in all_passed:
        body = func.get("body", "")
        docblock = func.get("docblock", "")
        good_code = f"{docblock}\n{body}" if docblock else body

        # Try each applicable mutation.
        for mutation_name, mutation_fn in MUTATIONS:
            bad_code, violation_desc = mutation_fn(body)
            if bad_code is None:
                continue

            # Verify the mutation is detectable.
            if not verify_mutation_detectable(bad_code):
                continue

            contrastive_pairs.append({
                "source": "automated_mutation",
                "mutation_type": mutation_name,
                "violation_description": violation_desc,
                "bad_code": bad_code,
                "good_code": good_code,
                "source_repo": func.get("source_repo", "unknown"),
                "source_function": func.get("function_name", "unknown"),
                "training_tags": [f"contrastive:{mutation_name}"],
            })
            mutation_counts[mutation_name] += 1

    # Save.
    output_path = MUTATED_DIR / "contrastive_mutations.json"
    with open(output_path, "w") as f:
        json.dump(contrastive_pairs, f, indent=2)

    print(f"\nMutation Results:")
    print(f"{'='*50}")
    for name, count in sorted(mutation_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count} pairs")
    print(f"\n  Total contrastive pairs: {len(contrastive_pairs)}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
