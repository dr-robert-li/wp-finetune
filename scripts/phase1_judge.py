#!/usr/bin/env python3
"""Phase 1, Step 3: Judge extracted functions using Claude.

- WordPress Core (quality_tier: "core") -> auto-passed, tagged only
- Everything else (quality_tier: "assessed") -> judged by Claude, pass/fail

Outputs:
  data/phase1_extraction/output/passed/   -> Functions that passed assessment
  data/phase1_extraction/output/failed/   -> Functions that failed (kept for analysis)
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from scripts.claude_agent import generate_json
from scripts.utils import (
    extract_json,
    load_checkpoint,
    save_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "extracted"
PASSED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "passed"
FAILED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "failed"
JUDGE_SYSTEM_PATH = PROJECT_ROOT / "config" / "judge_system.md"
TAXONOMY_PATH = PROJECT_ROOT / "config" / "taxonomy.yaml"


def phpcs_prefilter(code: str, max_errors_per_100_lines: float = 5.0) -> dict:
    """Run PHPCS as a cheap pre-filter before sending to Claude.

    Returns dict with 'passed', 'errors', 'warnings', 'error_density'.
    Functions that fail PHPCS badly are rejected without spending API tokens.
    """
    # Wrap bare function in <?php for PHPCS parsing.
    if not code.strip().startswith("<?php"):
        code = f"<?php\n{code}"

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".php", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            ["phpcs", "--standard=WordPress-Extra", "--report=json", tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        report = json.loads(result.stdout)
        file_report = list(report["files"].values())[0]
        line_count = max(code.count("\n"), 1)
        error_density = file_report["errors"] / line_count * 100

        return {
            "passed": error_density <= max_errors_per_100_lines,
            "errors": file_report["errors"],
            "warnings": file_report["warnings"],
            "error_density": round(error_density, 2),
        }
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        # PHPCS not installed or failed — skip pre-filter, let Claude decide.
        return {"passed": True, "errors": -1, "warnings": -1, "error_density": 0, "skipped": True}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def load_judge_system() -> str:
    with open(JUDGE_SYSTEM_PATH) as f:
        return f.read()


def load_taxonomy() -> dict:
    with open(TAXONOMY_PATH) as f:
        return yaml.safe_load(f)


def auto_tag_function(func: dict, taxonomy: dict) -> list[str]:
    """Assign taxonomy tags based on code content (for core auto-pass)."""
    tags = []
    body = func.get("body", "").lower()
    hooks = func.get("hooks_used", [])
    sql = func.get("sql_patterns", [])

    # SQL tags.
    if sql:
        if "prepared_query" in sql:
            tags.append("sql:prepared_statements")
        if "join" in sql:
            tags.append("sql:joins_across_meta")
        if "dbdelta" in sql:
            tags.append("sql:dbdelta_migrations")
        if any(p in sql for p in ["get_var", "get_col", "get_row"]):
            tags.append("sql:targeted_select")

    # Hook tags.
    if any("add_action" in h for h in hooks):
        tags.append("hooks:action_registration")
    if any("add_filter" in h for h in hooks):
        tags.append("hooks:filter_registration")

    # Security tags.
    if "wp_verify_nonce(" in body or "check_ajax_referer(" in body:
        tags.append("security:nonce_verification")
    if "current_user_can(" in body:
        tags.append("security:capability_checks")
    if any(esc in body for esc in ["esc_html(", "esc_attr(", "esc_url(", "wp_kses("]):
        tags.append("security:output_escaping")
    if any(s in body for s in ["sanitize_text_field(", "sanitize_email(", "absint("]):
        tags.append("security:input_sanitization")

    # Data modeling tags.
    if "register_post_type(" in body:
        tags.append("data:custom_post_types")
    if "register_taxonomy(" in body:
        tags.append("data:custom_taxonomies")
    if "register_rest_route(" in body:
        tags.append("rest:route_registration")
    if "set_transient(" in body or "get_transient(" in body:
        tags.append("data:transients")
    if "wp_cache_set(" in body or "wp_cache_get(" in body:
        tags.append("data:object_cache")

    # Performance tags.
    if "set_transient(" in body or "wp_cache_set(" in body:
        tags.append("perf:query_caching")
    if "wp_schedule_event(" in body:
        tags.append("cron:scheduled_events")

    # Theme tags.
    if "wp_enqueue_script(" in body or "wp_enqueue_style(" in body:
        tags.append("theme:enqueue_scripts")
    if "register_block_pattern(" in body:
        tags.append("theme:block_patterns")

    # Architecture tags.
    if "register_activation_hook(" in body:
        tags.append("arch:activation_hooks")
    if "register_deactivation_hook(" in body:
        tags.append("arch:deactivation_hooks")

    # Multisite tags.
    if "switch_to_blog(" in body:
        tags.append("multisite:site_switching")

    # i18n tags.
    if any(fn in body for fn in ["__(", "_e(", "esc_html__(", "esc_html_e(", "esc_attr__("]):
        tags.append("i18n:translation_functions")
    if "_n(" in body:
        tags.append("i18n:pluralization")

    # Accessibility tags.
    if any(a in body for a in ['aria-label', 'aria-describedby', 'role="', 'screen-reader-text']):
        tags.append("a11y:aria_attributes")
    if "<label" in body and 'for="' in body:
        tags.append("a11y:form_labels")

    return list(set(tags))


def _make_judge_prompt(func: dict) -> str:
    """Build the judge prompt string for a function."""
    code_block = func.get("body", "")
    docblock = func.get("docblock", "") or ""
    full_code = f"{docblock}\n{code_block}" if docblock else code_block

    return f"""Assess this WordPress function extracted from the plugin "{func['source_repo']}".

File: {func['source_file']}
Function: {func['function_name']}
Dependencies referenced: {json.dumps(func.get('dependencies', []))}
SQL patterns detected: {json.dumps(func.get('sql_patterns', []))}
Hooks used: {json.dumps(func.get('hooks_used', []))}

```php
{full_code}
```

Return your assessment as JSON matching the format in your instructions."""


def _apply_security_auto_fail(result: dict, func: dict) -> dict:
    """Enforce security auto-FAIL rule: security score < 5 overrides verdict."""
    security_score = result.get("scores", {}).get("security", 10)
    if security_score < 5:
        result["verdict"] = "FAIL"
        critical_failures = result.get("critical_failures", [])
        if "security_auto_fail" not in critical_failures:
            critical_failures.append("security_auto_fail")
        result["critical_failures"] = critical_failures
    return result


def judge_function(func: dict, system: str) -> dict:
    """Send a function to Claude for quality assessment via Claude Code agent."""
    prompt = _make_judge_prompt(func)

    try:
        result = generate_json(prompt, system=system)
        if result is None:
            return {
                "function_name": func["function_name"],
                "verdict": "FAIL",
                "notes": "Judge parse error: generate_json returned None",
                "scores": {},
                "critical_failures": ["parse_fail"],
                "dependency_chain": [],
                "training_tags": [],
            }
        return _apply_security_auto_fail(result, func)
    except Exception as e:
        return {
            "function_name": func["function_name"],
            "verdict": "FAIL",
            "notes": f"Judge error: {e}",
            "scores": {},
            "critical_failures": ["judge_error"],
            "dependency_chain": [],
            "training_tags": [],
        }


def main():
    PASSED_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    judge_system = load_judge_system()
    taxonomy = load_taxonomy()

    extracted_files = list(EXTRACTED_DIR.glob("*.json"))
    if not extracted_files:
        print("No extracted files found. Run phase1_extract.py first.")
        sys.exit(1)

    checkpoint = load_checkpoint("phase1_judge")
    completed_repos = set(checkpoint["completed"])
    failed_repos = list(checkpoint["failed"])

    total_passed = 0
    total_failed = 0
    total_auto = 0

    for extracted_file in extracted_files:
        repo_name = extracted_file.stem

        if repo_name in completed_repos:
            print(f"\nSkipping {repo_name} (checkpointed)")
            continue

        print(f"\nProcessing {extracted_file.name}...")
        with open(extracted_file) as f:
            functions = json.load(f)

        passed = []
        failed = []

        # Separate functions by tier.
        core_funcs = []
        assessed_funcs = []
        for func in functions:
            if func.get("line_count", 0) < 5:
                continue
            if func["quality_tier"] == "core":
                core_funcs.append(func)
            else:
                assessed_funcs.append(func)

        # Auto-pass WordPress Core.
        for func in core_funcs:
            func["verdict"] = "PASS"
            func["assessment"] = {"verdict": "PASS", "notes": "WordPress core - auto-passed"}
            func["training_tags"] = auto_tag_function(func, taxonomy)
            passed.append(func)
            total_auto += 1

        if core_funcs:
            print(f"  [{repo_name}] Auto-passed {len(core_funcs)} core functions")

        # PHPCS pre-filter assessed functions.
        phpcs_passed = []
        for func in assessed_funcs:
            code_to_check = func.get("body", "")
            docblock = func.get("docblock", "")
            if docblock:
                code_to_check = f"{docblock}\n{code_to_check}"

            phpcs_result = phpcs_prefilter(code_to_check)
            func["phpcs_prefilter"] = phpcs_result

            if not phpcs_result.get("passed", True):
                func["assessment"] = {
                    "verdict": "FAIL",
                    "notes": f"PHPCS pre-filter: {phpcs_result['errors']} errors "
                             f"({phpcs_result['error_density']} per 100 lines)",
                    "scores": {},
                    "critical_failures": ["phpcs_density_exceeded"],
                }
                func["training_tags"] = []
                failed.append(func)
                total_failed += 1
            else:
                phpcs_passed.append(func)

        # Judge functions via Claude Code agent.
        if phpcs_passed:
            print(f"  [{repo_name}] Judging {len(phpcs_passed)} functions via Claude Code agent")

            assessments = []
            for i, func in enumerate(phpcs_passed):
                assessment = judge_function(func, judge_system)
                assessments.append(assessment)
                if (i + 1) % 10 == 0:
                    p = sum(1 for a in assessments if a.get("verdict") == "PASS")
                    fail = len(assessments) - p
                    print(f"  [{repo_name}] Assessed {i + 1}/{len(phpcs_passed)} "
                          f"(passed: {p}, failed: {fail})")

            # Apply assessments to functions.
            for func, assessment in zip(phpcs_passed, assessments):
                func["assessment"] = assessment
                func["training_tags"] = assessment.get("training_tags", [])
                if assessment.get("verdict") == "PASS":
                    passed.append(func)
                    total_passed += 1
                else:
                    failed.append(func)
                    total_failed += 1

        # Save results per repo.
        if passed:
            with open(PASSED_DIR / f"{repo_name}.json", "w") as f:
                json.dump(passed, f, indent=2)

        if failed:
            with open(FAILED_DIR / f"{repo_name}.json", "w") as f:
                json.dump(failed, f, indent=2)

        print(f"  [{repo_name}] Done: {len(passed)} passed, {len(failed)} failed")

        # Checkpoint: mark repo complete.
        completed_repos.add(repo_name)
        save_checkpoint("phase1_judge", {
            "completed": list(completed_repos),
            "failed": failed_repos,
        })

    print(f"\n{'='*50}")
    print(f"Phase 1 Complete")
    print(f"  Core auto-passed: {total_auto}")
    print(f"  Assessed passed:  {total_passed}")
    print(f"  Assessed failed:  {total_failed}")
    print(f"  Pass rate:        {total_passed / max(total_passed + total_failed, 1):.1%}")
    print(f"\nResults in:")
    print(f"  {PASSED_DIR}")
    print(f"  {FAILED_DIR}")
    print(f"\nRun phase2_gap_analysis.py next.")


if __name__ == "__main__":
    main()
