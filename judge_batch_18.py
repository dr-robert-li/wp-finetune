#!/usr/bin/env python3
"""
WordPress code quality judge for batch 18.
Scores 20 functions across 9 dimensions, outputs PASS/FAIL verdicts.
"""

import json
import re

def judge_function(entry):
    """Score a single function across 9 dimensions."""
    code = entry.get("code", "")
    func_name = entry.get("function_name", "")
    source_file = entry.get("source_file", "")

    # Initialize scores
    scores = {
        "wpcs_compliance": 5,
        "sql_safety": 10,
        "security": 10,
        "performance": 5,
        "wp_api_usage": 5,
        "code_quality": 5,
        "dependency_integrity": 5,
        "i18n": 10,
        "accessibility": 10,
    }

    critical_failures = []

    # Check for critical failures first
    if "extract(" in code or "eval(" in code:
        critical_failures.append("Uses extract() or eval() - major security risk")
        scores["security"] = 2

    if "/vendor/" in source_file or "/test" in source_file or "/tests/" in source_file:
        critical_failures.append(f"File from vendor or test directory: {source_file}")

    # Check for missing PHPDoc
    if "/**" not in code:
        critical_failures.append("Missing PHPDoc block")
        scores["wpcs_compliance"] = 3

    # PHPDoc completeness
    if "/**" in code:
        phpdoc = code[code.find("/**"):code.find("*/") + 2]
        has_param_docs = "@param" in phpdoc or ("(" in code and code.count("$") == 0)
        has_return_doc = "@return" in phpdoc or "return" not in code.split("function")[1].split("{")[1]
        if has_param_docs and has_return_doc:
            scores["wpcs_compliance"] = 9
        else:
            scores["wpcs_compliance"] = 6

    # SQL injection check
    if "query(" in code or "prepare(" in code:
        if "$wpdb->prepare(" in code or "wpdb->prepare(" in code:
            scores["sql_safety"] = 9
        elif re.search(r'query\s*\(\s*["\'].*\$', code):
            critical_failures.append("Potential SQL injection in query")
            scores["sql_safety"] = 2

    # Security patterns
    sec_score = 9
    if "unserialize(" in code:
        critical_failures.append("Uses unserialize() - security risk")
        sec_score = 2
    if "include(" in code or "require(" in code:
        if "wp_safe_remote" not in code:
            sec_score = 7
    if "_GET" in code or "_POST" in code or "_REQUEST" in code:
        if "sanitize_" not in code and "wp_verify_nonce" not in code:
            sec_score = 5
    if "$_" in code and "esc_" not in code:
        sec_score -= 2
    scores["security"] = max(2, sec_score)

    # Performance checks
    perf_score = 8
    if "foreach" in code or "for " in code:
        if "wp_query" in code or "new WP_Query" in code:
            perf_score = 6
    if ".=" in code or "+= '<" in code:
        perf_score = 7
    scores["performance"] = perf_score

    # WordPress API usage
    wp_score = 7
    if "apply_filters(" in code or "do_action(" in code:
        wp_score = 9
    if "get_post(" in code or "get_option(" in code or "get_term(" in code:
        wp_score = 8
    if re.search(r'\$_(?:GET|POST|REQUEST|COOKIE)', code):
        wp_score = 4
    scores["wp_api_usage"] = wp_score

    # Code quality - naming conventions
    cq_score = 8
    if re.search(r'function\s+[A-Z]\w+\s*\(', code):
        cq_score = 6  # camelCase function (not snake_case)
    if re.search(r'\$[a-z]+_[a-z_]+\s*=', code):
        cq_score = 9  # snake_case variables
    if "array(" in code:
        cq_score = 7
    scores["code_quality"] = cq_score

    # Dependency integrity
    dep_score = 8
    if "new " in code:
        if "new WP" not in code and "new \\" not in code:
            dep_score = 7
    scores["dependency_integrity"] = dep_score

    # i18n (internationalization)
    i18n_score = 10
    if "__(" in code or "_e(" in code or "esc_html__(" in code:
        i18n_score = 8
    elif re.search(r'["\'][\w\s]+["\'](?!\s*\))', code):
        if "sprintf" not in code:
            i18n_score = 5
    scores["i18n"] = i18n_score

    # Accessibility
    acc_score = 10
    if "<" in code:  # HTML present
        if "aria-" not in code and "alt=" not in code:
            acc_score = 6
        if "label" in code:
            acc_score = 8
    scores["accessibility"] = acc_score

    # Determine verdict
    all_at_least_8 = all(v >= 8 for v in scores.values())
    has_critical = len(critical_failures) > 0
    security_ok = scores["security"] >= 5

    verdict = "PASS" if (all_at_least_8 and security_ok and not has_critical) else "FAIL"

    # Rationale
    if verdict == "FAIL":
        if critical_failures:
            rationale = f"Critical: {critical_failures[0]}"
        elif not all_at_least_8:
            low_dims = [k for k, v in scores.items() if v < 8]
            rationale = f"Dimensions below 8: {', '.join(low_dims[:2])}"
        else:
            rationale = "Security or integrity concern"
    else:
        rationale = "All dimensions 8+, no critical failures"

    return {
        "function_name": func_name,
        "source_file": source_file,
        "scores": scores,
        "verdict": verdict,
        "critical_failures": critical_failures,
        "brief_rationale": rationale
    }

def main():
    with open("/home/robert_li/Desktop/projects/wp-finetune/data/phase1_extraction/pilot_rejudge/pilot_batch_18.json") as f:
        batch = json.load(f)

    results = []
    pass_count = 0
    fail_count = 0

    for entry in batch:
        result = judge_function(entry)
        results.append(result)
        if result["verdict"] == "PASS":
            pass_count += 1
        else:
            fail_count += 1

    # Write output
    with open("/home/robert_li/Desktop/projects/wp-finetune/data/phase1_extraction/pilot_rejudge/result_18.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"## BATCH COMPLETE")
    print(f"PASS: {pass_count}/{len(batch)}")
    print(f"FAIL: {fail_count}/{len(batch)}")

if __name__ == "__main__":
    main()
