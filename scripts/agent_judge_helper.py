#!/usr/bin/env python3
"""Helper utility for agent-based function judging.

Used by Claude Code agents to:
1. List repos with extracted functions but no passed/failed output yet
2. Split an assessments file into passed/failed and write to output directories

Usage:
    # List unjudged repos
    python3 scripts/agent_judge_helper.py list

    # Split assessments for a repo
    python3 scripts/agent_judge_helper.py split <repo_name> <assessments.json>
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "extracted"
PASSED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "passed"
FAILED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "failed"


def list_unjudged() -> list:
    """Return list of repos that have extracted/ but no passed/ or failed/ file."""
    unjudged = []
    for extracted_path in sorted(EXTRACTED_DIR.glob("*.json")):
        repo_name = extracted_path.stem
        passed_path = PASSED_DIR / f"{repo_name}.json"
        failed_path = FAILED_DIR / f"{repo_name}.json"
        if not passed_path.exists() and not failed_path.exists():
            unjudged.append(repo_name)
    return unjudged


def _apply_security_auto_fail(assessment: dict) -> dict:
    """Enforce security auto-FAIL: security score < 5 forces FAIL verdict."""
    security_score = assessment.get("scores", {}).get("security", 10)
    if security_score < 5:
        assessment["verdict"] = "FAIL"
        critical_failures = assessment.get("critical_failures", [])
        if "security_auto_fail" not in critical_failures:
            critical_failures.append("security_auto_fail")
        assessment["critical_failures"] = critical_failures
    return assessment


def split_results(repo_name: str, assessments_file: str) -> dict:
    """Read assessments JSON, merge with extracted functions, write passed/failed.

    Args:
        repo_name: Name of the repo (must match extracted/{repo_name}.json)
        assessments_file: Path to JSON file containing list of assessment objects

    Returns:
        dict with keys: passed_count, failed_count
    """
    PASSED_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    extracted_path = EXTRACTED_DIR / f"{repo_name}.json"
    if not extracted_path.exists():
        raise FileNotFoundError(f"No extracted file for repo: {repo_name}")

    with open(extracted_path) as f:
        functions = json.load(f)

    with open(assessments_file) as f:
        assessments_list = json.load(f)

    # Build a map from function_name to assessment.
    # If multiple functions share a name, use file_path as tiebreaker.
    assessment_map = {}
    for a in assessments_list:
        key = (a.get("function_name", ""), a.get("file_path", ""))
        assessment_map[key] = a

    passed = []
    failed = []

    for func in functions:
        func_name = func.get("function_name", "")
        file_path = func.get("source_file", "")
        key = (func_name, file_path)

        if key in assessment_map:
            assessment = _apply_security_auto_fail(assessment_map[key])
        elif func_name in {k[0] for k in assessment_map}:
            # Fallback: match by function_name only
            for k, v in assessment_map.items():
                if k[0] == func_name:
                    assessment = _apply_security_auto_fail(v)
                    break
        else:
            # No assessment found — FAIL with note
            assessment = {
                "function_name": func_name,
                "file_path": file_path,
                "verdict": "FAIL",
                "scores": {},
                "critical_failures": ["no_assessment"],
                "dependency_chain": [],
                "training_tags": [],
                "notes": "No assessment provided by agent — auto-failed",
            }

        func["assessment"] = assessment
        func["training_tags"] = assessment.get("training_tags", [])

        if assessment.get("verdict") == "PASS":
            passed.append(func)
        else:
            failed.append(func)

    # Write results
    if passed:
        passed_path = PASSED_DIR / f"{repo_name}.json"
        with open(passed_path, "w") as f:
            json.dump(passed, f, indent=2)
        print(f"Wrote {len(passed)} passed functions to {passed_path}")

    if failed:
        failed_path = FAILED_DIR / f"{repo_name}.json"
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"Wrote {len(failed)} failed functions to {failed_path}")

    if not passed and not failed:
        # No functions at all — write empty passed file to mark repo as done
        passed_path = PASSED_DIR / f"{repo_name}.json"
        with open(passed_path, "w") as f:
            json.dump([], f, indent=2)
        print(f"Wrote empty passed file for {repo_name} (no functions)")

    return {"passed_count": len(passed), "failed_count": len(failed)}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        unjudged = list_unjudged()
        if unjudged:
            print(f"Unjudged repos ({len(unjudged)}):")
            for repo in unjudged:
                extracted_path = EXTRACTED_DIR / f"{repo}.json"
                with open(extracted_path) as f:
                    funcs = json.load(f)
                print(f"  {repo}: {len(funcs)} functions")
        else:
            print("All repos have been judged.")

    elif command == "split":
        if len(sys.argv) < 4:
            print("Usage: agent_judge_helper.py split <repo_name> <assessments.json>")
            sys.exit(1)
        repo_name = sys.argv[2]
        assessments_file = sys.argv[3]
        result = split_results(repo_name, assessments_file)
        print(f"Result: {result['passed_count']} passed, {result['failed_count']} failed")

    else:
        print(f"Unknown command: {command}")
        print("Commands: list, split")
        sys.exit(1)


if __name__ == "__main__":
    main()
