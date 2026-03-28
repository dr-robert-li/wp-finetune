"""Rubric-based ground truth scoring engine for WordPress PHP code quality.

Implements the 4-tool scoring procedure from wp_code_quality_rubric.md Section F:
  1. PHPCS (WordPress, WordPressVIPMinimum, Security standards)
  2. PHPStan (level 5 with WordPress stubs)
  3. Regex pattern matching
  4. LLM-assisted checks (TODO -- skipped for now)

Usage:
    from eval.rubric_scorer import score_code, RubricScore

    result: RubricScore = score_code(php_source_code)
    print(result.overall, result.grade)
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from eval.rubric_definitions import (
    CHECK_REGISTRY,
    CRITICAL_FLOOR_RULES,
    DIMENSION_MAX_POSITIVE,
    DIMENSION_WEIGHTS,
    NA_DETECTION_HINTS,
    REGEX_PATTERNS,
    SNIFF_TO_CHECKS,
)

# Derive lookup tables from CHECK_REGISTRY
CHECK_DIMENSION_MAP: dict[str, str] = {cid: c.dimension for cid, c in CHECK_REGISTRY.items()}
CHECK_WEIGHTS: dict[str, int] = {cid: c.weight for cid, c in CHECK_REGISTRY.items()}
POSITIVE_CHECK_IDS: set[str] = {cid for cid, c in CHECK_REGISTRY.items() if c.polarity == "positive"}
NEGATIVE_CHECK_IDS: set[str] = {cid for cid, c in CHECK_REGISTRY.items() if c.polarity == "negative"}

# ---------------------------------------------------------------------------
# Grade bands (from rubric Section D)
# ---------------------------------------------------------------------------

GRADE_BANDS: list[tuple[float, str]] = [
    (90.0, "Excellent"),
    (75.0, "Good"),
    (60.0, "Acceptable"),
    (40.0, "Poor"),
    (20.0, "Bad"),
    (0.0, "Failing"),
]


def _score_to_grade(overall: float) -> str:
    for threshold, grade in GRADE_BANDS:
        if overall >= threshold:
            return grade
    return "Failing"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class RubricScore:
    """Ground truth score for a single PHP code sample."""

    file_path: str
    dimension_scores: dict[str, Optional[float]]  # dim_key -> 0-10 or None (N/A)
    dimension_na: list[str]
    overall: float  # 0-100
    triggered_checks: dict[str, list[str]]  # dim_key -> [check_ids that fired]
    check_evidence: dict[str, list[str]]  # check_id -> [evidence strings]
    grade: str
    floor_rules_applied: list[str]
    llm_checks_skipped: int = 0  # count of LLM checks not yet implemented

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "dimension_scores": self.dimension_scores,
            "dimension_na": self.dimension_na,
            "overall": self.overall,
            "triggered_checks": self.triggered_checks,
            "check_evidence": self.check_evidence,
            "grade": self.grade,
            "floor_rules_applied": self.floor_rules_applied,
            "llm_checks_skipped": self.llm_checks_skipped,
        }


# ---------------------------------------------------------------------------
# Tool runners
# ---------------------------------------------------------------------------


def run_phpcs(code: str, standard: str = "WordPress") -> dict:
    """Write code to temp file, run phpcs --standard={standard} --report=json.

    Returns parsed JSON output dict. Handles:
    - phpcs exit code 1 (violations found -- expected, not an error)
    - phpcs not installed (returns empty dict with _unavailable flag)
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".php", delete=False
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            ["phpcs", f"--standard={standard}", "--report=json", tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # PHPCS exits 0 = clean, 1 = violations found, 2 = errors in PHPCS itself
        raw = proc.stdout.strip()
        if not raw:
            return {"_unavailable": True, "files": {}, "totals": {"errors": 0, "warnings": 0}}

        return json.loads(raw)

    except FileNotFoundError:
        # phpcs binary not installed
        return {"_unavailable": True, "files": {}, "totals": {"errors": 0, "warnings": 0}}
    except subprocess.TimeoutExpired:
        return {"_timeout": True, "files": {}, "totals": {"errors": 0, "warnings": 0}}
    except json.JSONDecodeError:
        return {"_parse_error": True, "files": {}, "totals": {"errors": 0, "warnings": 0}}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_phpstan(code: str) -> dict:
    """Write code to temp file, run phpstan analyse --level=5 --error-format=json.

    Returns parsed JSON output dict. Handles phpstan not installed gracefully.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".php", delete=False
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [
                "phpstan",
                "analyse",
                "--level=5",
                "--error-format=json",
                "--no-progress",
                tmp_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        raw = proc.stdout.strip()
        if not raw:
            return {"_unavailable": True, "files": {}, "totals": {"errors": 0}}

        return json.loads(raw)

    except FileNotFoundError:
        return {"_unavailable": True, "files": {}, "totals": {"errors": 0}}
    except subprocess.TimeoutExpired:
        return {"_timeout": True, "files": {}, "totals": {"errors": 0}}
    except json.JSONDecodeError:
        return {"_parse_error": True, "files": {}, "totals": {"errors": 0}}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Check mappers
# ---------------------------------------------------------------------------


def _collect_phpcs_sniffs(phpcs_output: dict) -> dict[str, list[str]]:
    """Extract {sniff_source: [messages]} from phpcs JSON output."""
    sniffs: dict[str, list[str]] = {}
    for file_data in phpcs_output.get("files", {}).values():
        for msg in file_data.get("messages", []):
            source = msg.get("source", "")
            # PHPCS sniff sources look like "WordPress.Security.EscapeOutput.OutputNotEscaped"
            # We match on the prefix (first 3 segments for WordPress, first 2 for others)
            parts = source.split(".")
            # Build progressively shorter prefixes to match SNIFF_TO_CHECKS keys
            prefixes = []
            for i in range(len(parts), 0, -1):
                prefixes.append(".".join(parts[:i]))

            matched = False
            for prefix in prefixes:
                if prefix in SNIFF_TO_CHECKS:
                    sniffs.setdefault(prefix, []).append(
                        msg.get("message", "")
                    )
                    matched = True
                    break
            if not matched and source:
                # Store under full source for debugging
                sniffs.setdefault(source, []).append(msg.get("message", ""))
    return sniffs


def map_phpcs_to_checks(
    phpcs_outputs: dict[str, dict],
) -> tuple[dict[str, bool], dict[str, list[str]]]:
    """Map PHPCS outputs from multiple standards to check IDs.

    Args:
        phpcs_outputs: {standard_name: phpcs_json_output}

    Returns:
        (check_hits, evidence) where:
        - check_hits: {check_id: True/False}
        - evidence: {check_id: [evidence strings]}

    For POSITIVE checks mapped to PHPCS: the check is TRUE when the sniff has
    ZERO violations (absence of violation = code is correct).
    For NEGATIVE checks mapped to PHPCS: the check is TRUE when the sniff HAS
    violations (presence of violation = code has the problem).
    """
    # Aggregate all sniff violations across standards
    all_sniffs: dict[str, list[str]] = {}
    for _standard, output in phpcs_outputs.items():
        if output.get("_unavailable") or output.get("_timeout"):
            continue
        sniffs = _collect_phpcs_sniffs(output)
        for sniff, messages in sniffs.items():
            all_sniffs.setdefault(sniff, []).extend(messages)

    check_hits: dict[str, bool] = {}
    evidence: dict[str, list[str]] = {}

    # Map sniff violations to check IDs
    for sniff_prefix, check_ids in SNIFF_TO_CHECKS.items():
        has_violations = sniff_prefix in all_sniffs
        violation_messages = all_sniffs.get(sniff_prefix, [])

        for check_id in check_ids:
            if check_id in POSITIVE_CHECK_IDS:
                # Positive checks: TRUE when no violations (code is correct)
                check_hits[check_id] = not has_violations
                if has_violations:
                    evidence[check_id] = [
                        f"PHPCS {sniff_prefix}: {m}"
                        for m in violation_messages[:5]
                    ]
            elif check_id in NEGATIVE_CHECK_IDS:
                # Negative checks: TRUE when violations exist (code has problem)
                check_hits[check_id] = has_violations
                if has_violations:
                    evidence[check_id] = [
                        f"PHPCS {sniff_prefix}: {m}"
                        for m in violation_messages[:5]
                    ]

    return check_hits, evidence


def map_phpstan_to_checks(
    phpstan_output: dict,
) -> tuple[dict[str, bool], dict[str, list[str]]]:
    """Map PHPStan errors to ERR-*, STR-* check IDs.

    PHPStan detects:
    - ERR-P05/ERR-P06: type declarations present (absence of errors = good)
    - ERR-N01: WP_Error return used without is_wp_error check
    - ERR-N05: missing type hints
    - STR-P01/STR-N01: filter callbacks returning values

    Returns:
        (check_hits, evidence)
    """
    check_hits: dict[str, bool] = {}
    evidence: dict[str, list[str]] = {}

    if phpstan_output.get("_unavailable") or phpstan_output.get("_timeout"):
        return check_hits, evidence

    # Collect all error messages
    errors: list[str] = []
    file_errors = phpstan_output.get("files", {})
    if isinstance(file_errors, dict):
        for file_data in file_errors.values():
            if isinstance(file_data, dict):
                for msg in file_data.get("messages", []):
                    errors.append(msg.get("message", ""))
            elif isinstance(file_data, list):
                for msg in file_data:
                    if isinstance(msg, dict):
                        errors.append(msg.get("message", ""))
                    elif isinstance(msg, str):
                        errors.append(msg)

    # Also handle flat error list format
    if "errors" in phpstan_output and isinstance(phpstan_output["errors"], list):
        for err in phpstan_output["errors"]:
            if isinstance(err, str):
                errors.append(err)
            elif isinstance(err, dict):
                errors.append(err.get("message", ""))

    # Detect type declaration issues
    type_hint_errors = [
        e for e in errors
        if "has no type" in e.lower()
        or "missing type" in e.lower()
        or "no return type" in e.lower()
        or "has no return type" in e.lower()
        or "parameter.*has no type" in e.lower()
    ]
    has_type_issues = len(type_hint_errors) > 0
    # ERR-P05: type declarations on parameters (positive = no issues)
    check_hits["ERR-P05"] = not has_type_issues
    # ERR-P06: return type declarations (positive = no issues)
    check_hits["ERR-P06"] = not has_type_issues
    # ERR-N05: missing type hints (negative = has issues)
    check_hits["ERR-N05"] = has_type_issues
    if has_type_issues:
        evidence["ERR-N05"] = [f"PHPStan: {e}" for e in type_hint_errors[:5]]

    # Detect filter callback return issues
    filter_return_errors = [
        e for e in errors
        if "filter" in e.lower() and "return" in e.lower()
    ]
    if filter_return_errors:
        check_hits["STR-N01"] = True
        evidence["STR-N01"] = [f"PHPStan: {e}" for e in filter_return_errors[:5]]
    # STR-P01 is the inverse
    check_hits["STR-P01"] = not bool(filter_return_errors)

    return check_hits, evidence


def run_regex_checks(code: str) -> tuple[dict[str, bool], dict[str, list[str]]]:
    """Run all REGEX_PATTERNS from rubric_definitions against code.

    Returns:
        (check_hits, evidence) where check_hits = {check_id: matched}
    """
    check_hits: dict[str, bool] = {}
    evidence: dict[str, list[str]] = {}

    # Normalize quotes: PHP uses both ' and " interchangeably for strings.
    # Rubric patterns use single quotes. Create a quote-normalized copy for matching.
    code_normalized = code.replace('"', "'")

    for check_id, pattern in REGEX_PATTERNS.items():
        try:
            # Match against both original and quote-normalized code
            matches = list(re.finditer(pattern, code, re.MULTILINE))
            if not matches:
                matches = list(re.finditer(pattern, code_normalized, re.MULTILINE))
            matched = len(matches) > 0
            check_hits[check_id] = matched
            if matched:
                evidence[check_id] = [
                    f"Regex match: {m.group()[:100]}"
                    for m in matches[:5]
                ]
        except re.error:
            # Invalid regex pattern -- skip
            check_hits[check_id] = False

    # Special compound checks that need multiple regex results:

    # STR-N09: register_rest_route present but no permission_callback
    if check_hits.get("STR-N09") and check_hits.get("STR-P09"):
        # Has both register_rest_route AND permission_callback -- not a violation
        check_hits["STR-N09"] = False
    elif check_hits.get("STR-N09") and not check_hits.get("STR-P09"):
        # Has register_rest_route but NO permission_callback -- violation
        check_hits["STR-N09"] = True

    # SQL-N04: LIKE %s without esc_like
    if check_hits.get("SQL-N04") and check_hits.get("SQL-P03"):
        # Has LIKE %s but also has esc_like -- less likely a violation
        check_hits["SQL-N04"] = False

    # PERF-P01: needs both wp_cache_get and wp_cache_set
    if check_hits.get("PERF-P01"):
        has_cache_set = bool(re.search(r"wp_cache_set\s*\(", code))
        if not has_cache_set:
            check_hits["PERF-P01"] = False

    return check_hits, evidence


# ---------------------------------------------------------------------------
# N/A detection
# ---------------------------------------------------------------------------


def determine_na_dimensions(code: str) -> list[str]:
    """Use NA_DETECTION_HINTS to detect which dimensions are N/A for this code.

    Each hint in NA_DETECTION_HINTS maps a dimension key to a single regex pattern.
    If the pattern does NOT match, the dimension is N/A (no code surface exists).
    """
    na_dims: list[str] = []

    for dim_key, pattern in NA_DETECTION_HINTS.items():
        try:
            has_surface = bool(re.search(pattern, code, re.MULTILINE | re.IGNORECASE))
        except re.error:
            has_surface = True  # If pattern is invalid, assume applicable
        if not has_surface:
            na_dims.append(dim_key)

    return na_dims


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def compute_dimension_scores(
    check_hits: dict[str, bool],
    na_dims: list[str],
) -> dict[str, Optional[float]]:
    """Compute per-dimension scores from check hits.

    For each dimension:
    1. Start at 10.0 (assume code is correct for patterns not evaluated)
    2. Deduct for negative checks that fired (proportional to weight)
    3. Grant bonus for positive checks that fired (up to the score cap)
    4. Clamp to [0, 10]
    5. Return None for N/A dimensions

    Scoring philosophy: A function starts at full marks and loses points
    for detected problems. Positive checks can recover lost points but
    cannot exceed 10. This avoids penalizing simple functions that don't
    exhibit many patterns.
    """
    scores: dict[str, Optional[float]] = {}

    for dim_key, max_pos in DIMENSION_MAX_POSITIVE.items():
        if dim_key in na_dims:
            scores[dim_key] = None
            continue

        # Get all check IDs for this dimension
        dim_checks = {
            cid for cid, d in CHECK_DIMENSION_MAP.items() if d == dim_key
        }

        # Start at 10 (full marks) and deduct for negatives
        score = 10.0
        evaluated_negative_weight = 0.0
        max_negative_weight = sum(
            abs(CHECK_WEIGHTS.get(cid, 0))
            for cid in dim_checks
            if cid in NEGATIVE_CHECK_IDS and cid in check_hits
        )

        for check_id in dim_checks:
            if check_id not in check_hits:
                continue

            weight = CHECK_WEIGHTS.get(check_id, 0)
            if check_id in NEGATIVE_CHECK_IDS and check_hits[check_id]:
                # Deduct proportionally: each negative point costs up to 10/max_neg
                if max_negative_weight > 0:
                    deduction = (abs(weight) / max(max_negative_weight, abs(weight))) * 3.0
                else:
                    deduction = 1.0
                score -= deduction

        # Positive checks provide a floor — if many positives fire,
        # the score can't go below a baseline
        positive_hits = sum(
            CHECK_WEIGHTS.get(cid, 0)
            for cid in dim_checks
            if cid in POSITIVE_CHECK_IDS and check_hits.get(cid, False)
        )
        if max_pos > 0:
            positive_ratio = positive_hits / max_pos
            # Positive signals set a floor proportional to their coverage
            positive_floor = positive_ratio * 8.0  # up to 8/10 from positives alone
            score = max(score, positive_floor)

        scores[dim_key] = max(0.0, min(10.0, score))

    return scores


def apply_floor_rules(
    scores: dict[str, Optional[float]],
    check_hits: dict[str, bool],
) -> tuple[dict[str, Optional[float]], list[str]]:
    """Apply CRITICAL_FLOOR_RULES from rubric definitions.

    Returns:
        (updated_scores, list of applied rule descriptions)
    """
    applied: list[str] = []

    for rule in CRITICAL_FLOOR_RULES:
        # CRITICAL_FLOOR_RULES entries are (dimension, cap, triggers) tuples
        if isinstance(rule, (list, tuple)):
            dim_key, cap, trigger_checks = rule[0], rule[1], rule[2]
        else:
            dim_key = rule["dimension"]
            cap = rule["cap"]
            trigger_checks = rule["triggers"]

        if scores.get(dim_key) is None:
            continue  # N/A dimension, skip

        if any(check_hits.get(cid, False) for cid in trigger_checks):
            current = scores[dim_key]
            if current is not None and current > cap:
                scores[dim_key] = cap
                triggered = [
                    cid for cid in trigger_checks if check_hits.get(cid, False)
                ]
                applied.append(
                    f"{dim_key} capped at {cap} due to: {', '.join(triggered)}"
                )

    return scores, applied


def compute_overall(dimension_scores: dict[str, Optional[float]]) -> float:
    """Compute weighted overall score 0-100 with N/A weight redistribution.

    Applicable dimensions have their weights normalized so they sum to 1.0,
    then the weighted sum is scaled to 0-100.
    """
    applicable = {
        k: v for k, v in dimension_scores.items() if v is not None
    }

    if not applicable:
        return 0.0

    total_weight = sum(DIMENSION_WEIGHTS[k] for k in applicable)
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(
        applicable[k] * DIMENSION_WEIGHTS[k] / total_weight
        for k in applicable
    )

    return round(weighted_sum * 10, 1)  # 0-100 scale


# ---------------------------------------------------------------------------
# Triggered checks aggregation
# ---------------------------------------------------------------------------


def _group_checks_by_dimension(
    check_hits: dict[str, bool],
) -> dict[str, list[str]]:
    """Group triggered check IDs by their dimension."""
    grouped: dict[str, list[str]] = {}
    for check_id, hit in check_hits.items():
        if not hit:
            continue
        dim = CHECK_DIMENSION_MAP.get(check_id)
        if dim:
            grouped.setdefault(dim, []).append(check_id)
    return grouped


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Count of LLM-assisted checks from rubric Section F Step 5 (skipped for now)
_LLM_CHECK_COUNT = 18  # SEC-P05, SEC-P13, SEC-N04, SEC-N13, STR-N01, etc.


def score_code(code: str, file_path: str = "<generated>") -> RubricScore:
    """Score a PHP code string against the WordPress code quality rubric.

    Runs all 4 tool passes (PHPCS x3 standards, PHPStan, regex, LLM-skipped),
    maps results to check IDs, computes per-dimension and overall scores.

    Args:
        code: PHP source code string.
        file_path: Optional file path for reporting.

    Returns:
        RubricScore dataclass with all scoring details.
    """
    all_check_hits: dict[str, bool] = {}
    all_evidence: dict[str, list[str]] = {}

    # --- Tool 1: PHPCS (3 standards) ---
    phpcs_outputs: dict[str, dict] = {}
    for standard in ("WordPress", "WordPressVIPMinimum", "Security"):
        phpcs_outputs[standard] = run_phpcs(code, standard=standard)

    phpcs_hits, phpcs_ev = map_phpcs_to_checks(phpcs_outputs)
    all_check_hits.update(phpcs_hits)
    all_evidence.update(phpcs_ev)

    # --- Tool 2: PHPStan ---
    phpstan_output = run_phpstan(code)
    phpstan_hits, phpstan_ev = map_phpstan_to_checks(phpstan_output)
    # PHPStan results supplement (don't override) PHPCS results
    for k, v in phpstan_hits.items():
        if k not in all_check_hits:
            all_check_hits[k] = v
    all_evidence.update(phpstan_ev)

    # --- Tool 3: Regex checks ---
    regex_hits, regex_ev = run_regex_checks(code)
    # Regex results supplement (don't override) tool-based results
    for k, v in regex_hits.items():
        if k not in all_check_hits:
            all_check_hits[k] = v
    for k, v in regex_ev.items():
        all_evidence.setdefault(k, []).extend(v)

    # --- Tool 4: LLM checks ---
    # TODO: Implement LLM-assisted checks (SEC-P05, SEC-P13, SEC-N04, SEC-N10,
    # SEC-N13, SEC-N15, SEC-N16, SEC-N17, STR-N01, STR-N04, STR-N05, STR-N06,
    # STR-N08, STR-P12, PERF-N01, PERF-N05, PERF-N10, PERF-N11, PERF-P04,
    # PERF-P10, ERR-N01, ERR-N06, ERR-N07, ERR-N11, ERR-P09, I18N-N12,
    # I18N-N13, I18N-P01, A11Y-N01, A11Y-N05, A11Y-N08, A11Y-N11, A11Y-N12,
    # WAPI-N12, WAPI-N13, WAPI-P08). These require an LLM judge pass with
    # binary YES/NO prompts as defined in rubric Section F Step 5.

    # --- Compute scores ---
    na_dims = determine_na_dimensions(code)
    dim_scores = compute_dimension_scores(all_check_hits, na_dims)
    dim_scores, floor_applied = apply_floor_rules(dim_scores, all_check_hits)
    overall = compute_overall(dim_scores)
    grade = _score_to_grade(overall)
    triggered = _group_checks_by_dimension(all_check_hits)

    return RubricScore(
        file_path=file_path,
        dimension_scores=dim_scores,
        dimension_na=na_dims,
        overall=overall,
        triggered_checks=triggered,
        check_evidence=all_evidence,
        grade=grade,
        floor_rules_applied=floor_applied,
        llm_checks_skipped=_LLM_CHECK_COUNT,
    )
