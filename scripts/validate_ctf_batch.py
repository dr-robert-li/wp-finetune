#!/usr/bin/env python3
"""Per-batch CtF quick validator.

Runs after a single Claude Code agent produces batch_NNN.json and before the
full merge_reasoning_batches.py gate. Catches the "learned-from-failures"
issues that the previous heuristic pipeline slipped through:

  1. `$args` template bug pattern (undefined variable reference)
  2. `'plugin-slug'` generic text domain placeholder
  3. wrong-context admin checks on customer/frontend code (heuristic warn)
  4. corrected_code identical to defective after normalization
  5. missing or malformed dimensions
  6. PHP lint failure (with the fixed `<?php` prefix)
  7. generic cache-wrapper evasion (wp_cache_get without addressing SQL)

Exit 0 on ALL-PASS, 1 on any fail. Also prints a per-example breakdown so we
can attribute failures to specific agents.

Usage:
    python scripts/validate_ctf_batch.py data/phase4_reasoning/critique_then_fix/batches/batch_000.json
    python scripts/validate_ctf_batch.py data/phase4_reasoning/critique_then_fix/batches/batch_*.json
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_critique_then_fix import php_lint_check  # uses the FIXED <?php-prefix lint

REQUIRED_DIMENSIONS = [
    "wpcs_compliance", "sql_safety", "security", "performance",
    "wp_api_usage", "code_quality", "dependency_integrity", "i18n", "accessibility",
]
SEVERITY_LEVELS = {"critical", "high", "medium", "low"}

# Bogus patterns from prior failed runs
BOGUS_PATTERNS = [
    (re.compile(r"Defensive:\s*Added input sanitization per WPCS review"), "bogus_defensive_template"),
    (re.compile(r"Input validation added per code quality review"), "bogus_input_validation_template"),
    (re.compile(r"if\s*\(\s*empty\(\s*\$args\s*\)\s*&&\s*func_num_args\("), "bogus_args_empty_check"),
    (re.compile(r"\$sanitized_input\s*=\s*array_map"), "bogus_args_sanitize"),
    (re.compile(r"\$args\s*\?\?\s*null"), "undefined_args_null_coalesce"),
    (re.compile(r"['\"]plugin-slug['\"]"), "generic_plugin_slug_placeholder"),
]

# Frontend/customer context hints — admin cap checks on these are wrong-context
FRONTEND_HINTS = [
    "checkout", "payment", "paypal_return", "maybe_return", "front_end",
    "render", "widget", "shortcode", "display", "customer",
]

def normalize_code(code: str) -> str:
    no_comments = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    no_comments = re.sub(r"//.*?$", "", no_comments, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", no_comments).strip()


def variables_in_signature(code: str) -> set[str]:
    """Extract $varname parameters from the function signature."""
    m = re.search(r"function\s+\S*\s*\(([^)]*)\)", code)
    if not m:
        return set()
    return set(re.findall(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", m.group(1)))


def variables_referenced(code: str) -> set[str]:
    return set(re.findall(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", code))


def validate_example(ex: dict, idx: int) -> list[str]:
    errors = []
    fn = ex.get("function_name", f"<entry {idx}>")

    # Required top-level fields
    critique = ex.get("critique", {})
    defective = ex.get("defective_code", "") or ""
    corrected = ex.get("corrected_code", "") or ""
    source_file = ex.get("source_file", "")

    if not critique.get("summary"):
        errors.append("no_summary")
    dims = critique.get("dimensions", {})
    missing = [d for d in REQUIRED_DIMENSIONS if d not in dims]
    if missing:
        errors.append(f"missing_dims:{','.join(missing)}")
    for d in REQUIRED_DIMENSIONS:
        if d not in dims:
            continue
        v = dims[d]
        if not isinstance(v, dict):
            errors.append(f"{d}:not_dict")
            continue
        sev = v.get("severity")
        if sev not in SEVERITY_LEVELS:
            errors.append(f"{d}:bad_severity:{sev}")
        issue = v.get("issue", "") or ""
        fix = v.get("fix", "") or ""
        # Match the merge gate's bar (non-empty, not min-length). Phrases like
        # "No changes needed." are legitimate terse answers for inapplicable
        # dims — the merge gate accepts them, so the validator should too.
        if len(issue) < 10:
            errors.append(f"{d}:short_issue({len(issue)})")
        if len(fix) < 10:
            errors.append(f"{d}:short_fix({len(fix)})")

    # corrected_code sanity
    if len(corrected.strip()) <= 20:
        errors.append("corrected_too_short")
    if normalize_code(corrected) == normalize_code(defective):
        errors.append("corrected_identical_to_defective")

    # Bogus patterns
    for pat, label in BOGUS_PATTERNS:
        if pat.search(corrected):
            errors.append(f"bogus:{label}")

    # Undefined variable reference — vars in corrected that aren't in signature
    # and aren't $_POST/$_GET/$wpdb/$this/$wp_/core WP globals.
    sig_vars = variables_in_signature(corrected) or variables_in_signature(defective)
    ref_vars = variables_referenced(corrected)
    allowed = sig_vars | {
        "_POST", "_GET", "_REQUEST", "_FILES", "_SERVER", "_COOKIE", "_SESSION", "_ENV",
        "wpdb", "this", "wp", "wp_query", "post", "current_user", "user_ID",
        "cache_key", "cached", "nonce", "result", "response", "data", "new_date",
        "date_created", "order", "order_id", "tmp", "tmp_path", "properties",
        # common locals introduced by fixes — generous allowlist
        "value", "values", "field", "fields", "key", "item", "row", "rows",
        "query", "sql", "params", "output", "html", "url", "link", "id",
    }
    # Any var referenced but not in the allowlist AND not defined locally (via = or as loop var)
    local_defined = set(re.findall(r"\$([a-zA-Z_][a-zA-Z0-9_]*)\s*=", corrected))
    loop_defined = set(re.findall(r"foreach\s*\([^)]*\bas\b\s*\$([a-zA-Z_][a-zA-Z0-9_]*)", corrected))
    loop_kv = set(re.findall(r"=>\s*\$([a-zA-Z_][a-zA-Z0-9_]*)", corrected))
    defined = sig_vars | allowed | local_defined | loop_defined | loop_kv
    undefined = ref_vars - defined
    # Filter out obvious false positives
    undefined = {v for v in undefined if not v.startswith("_")}
    if undefined:
        # Only flag if there are "bad" looking undefined names
        suspicious = {v for v in undefined if v in {"args", "params", "data"}}
        if suspicious:
            errors.append(f"undefined_vars:{','.join(sorted(suspicious))}")

    # Wrong-context admin check heuristic (warn — not hard fail)
    fn_lower = (fn or "").lower()
    if any(h in fn_lower for h in FRONTEND_HINTS):
        if re.search(r"current_user_can\s*\(\s*['\"]manage_options['\"]", corrected):
            errors.append("warn:frontend_with_manage_options")

    # Cache-wrapper evasion: if critique flagged critical sql_safety missing prepare,
    # corrected_code must contain $wpdb->prepare (not just wp_cache_*)
    sql_crit = dims.get("sql_safety", {}).get("severity") in {"critical", "high"}
    sql_issue = (dims.get("sql_safety", {}).get("issue", "") or "").lower()
    if sql_crit and ("prepare" in sql_issue or "injection" in sql_issue):
        has_prepare = "$wpdb->prepare" in corrected or "wpdb->prepare" in corrected
        # Acceptable alternatives: if raw $wpdb->query was REPLACED with no raw dynamic SQL
        has_dangerous_raw = re.search(r"\$wpdb->(query|get_results|get_row|get_var|get_col)\s*\(\s*[\"'][^\"']*\$", corrected)
        if not has_prepare and has_dangerous_raw:
            errors.append("cache_wrapper_evasion:sql_unfixed")

    # PHP lint (with fixed <?php prefix).
    # For class methods, wrap in a synthetic class scope so `public`/`self::`/`parent::` work.
    if corrected:
        # Strip any leading <?php that the agent inserted — php_lint_check will
        # add its own prefix, and double-prefix inside a class wrap breaks parsing.
        stripped = corrected
        stripped = re.sub(r"^\s*<\?php\s*", "", stripped)
        stripped = re.sub(r"\?>\s*$", "", stripped)

        needs_class_scope = bool(
            re.search(r"^\s*(public|private|protected|static|final|abstract)\s+function", stripped, re.MULTILINE)
            or ("::" in fn and "::" not in fn.split("::", 1)[0])  # fn is Class::method
        )
        code_to_lint = stripped
        if needs_class_scope:
            # Wrap in a class that matches the fn's class name if parseable, else a generic one
            cls_name = fn.split("::", 1)[0] if "::" in fn else "WpFtValidatorShim"
            # Avoid reserved-keyword or invalid class names
            if cls_name.lower() in {"false", "true", "null", "self", "static", "parent", "class"} or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", cls_name):
                cls_name = "WpFtValidatorShim"
            # Add a parent class to satisfy `parent::` references and an interface for good measure.
            # If the method uses parent:: we need an actual parent in scope.
            if "parent::" in stripped:
                code_to_lint = (
                    f"class WpFtValidatorShimParent {{\n"
                    f"    public function __construct() {{}}\n"
                    f"}}\n"
                    f"class {cls_name} extends WpFtValidatorShimParent {{\n"
                    f"{stripped}\n"
                    f"}}"
                )
            else:
                code_to_lint = f"class {cls_name} {{\n{stripped}\n}}"
        lint = php_lint_check(code_to_lint)
        if lint.get("valid") is False and "php not available" not in (lint.get("errors") or ""):
            err_text = lint.get("errors") or ""
            errors.append(f"php_lint_invalid:{err_text[:150]}")

    return errors


def validate_batch(path: Path) -> tuple[int, int, list]:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return 0, 1, [(None, f"parse_error:{e}")]
    if not isinstance(data, list):
        return 0, 1, [(None, "not_a_list")]
    pass_count = 0
    fail_count = 0
    failures = []
    for i, ex in enumerate(data):
        errs = validate_example(ex, i)
        # Split warnings vs hard errors
        hard = [e for e in errs if not e.startswith("warn:")]
        warns = [e for e in errs if e.startswith("warn:")]
        if hard:
            fail_count += 1
            failures.append((i + 1, ex.get("function_name", "?"), hard, warns))
        else:
            pass_count += 1
            if warns:
                failures.append((i + 1, ex.get("function_name", "?"), [], warns))
    return pass_count, fail_count, failures


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    total_pass = 0
    total_fail = 0
    total_entries = 0

    for arg in sys.argv[1:]:
        for p in sorted(Path().glob(arg)):
            if not p.is_file():
                continue
            pass_n, fail_n, failures = validate_batch(p)
            total_entries += pass_n + fail_n
            total_pass += pass_n
            total_fail += fail_n
            status = "PASS" if fail_n == 0 else "FAIL"
            print(f"\n=== {p.name}: {status} ({pass_n}/{pass_n + fail_n} passed) ===")
            for item in failures:
                if len(item) == 4:
                    idx, fn, hard, warns = item
                    marker = "FAIL" if hard else "WARN"
                    print(f"  [{marker}] #{idx} {fn}")
                    for e in hard:
                        print(f"         ✗ {e}")
                    for w in warns:
                        print(f"         ⚠ {w}")
                else:
                    idx, msg = item
                    print(f"  [FAIL] {msg}")

    print(f"\n=== TOTAL: {total_pass}/{total_entries} passed, {total_fail} failed ===")
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
