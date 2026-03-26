#!/usr/bin/env python3
"""Pipeline Orchestrator — checks state and reports what needs to be done.

This script is the brain of the pipeline. It:
1. Scans all output directories to determine current state
2. Compares against targets
3. Produces a structured action plan (JSON) for Claude Code to execute

Usage:
    python scripts/pipeline_orchestrator.py status    # Show current state
    python scripts/pipeline_orchestrator.py plan      # Show what needs doing
    python scripts/pipeline_orchestrator.py export     # Run the final export

Claude Code reads the plan output and spawns agents accordingly.
"""

import json
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Targets ──────────────────────────────────────────────────────────
TARGETS = {
    "real_code_passed": 15000,
    "synthetic_passed": 200,
    "judge_high": 1500,
    "judge_low": 1000,
    "judge_synth": 1500,
    "cot_total": 500,
}

# ── Paths ────────────────────────────────────────────────────────────
REPOS_DIR = PROJECT_ROOT / "phase1_extraction" / "repos"
EXTRACTED_DIR = PROJECT_ROOT / "phase1_extraction" / "output" / "extracted"
PASSED_DIR = PROJECT_ROOT / "phase1_extraction" / "output" / "passed"
FAILED_DIR = PROJECT_ROOT / "phase1_extraction" / "output" / "failed"
GAP_REPORT = PROJECT_ROOT / "phase2_synthetic" / "gap_report.json"
GENERATED_DIR = PROJECT_ROOT / "phase2_synthetic" / "output" / "generated"
JUDGED_DIR = PROJECT_ROOT / "phase2_synthetic" / "output" / "judged"
JUDGE_TRAINING_DIR = PROJECT_ROOT / "phase2_synthetic" / "output" / "judge_training"
COT_DIR = PROJECT_ROOT / "phase3_cot" / "output"
FINAL_DIR = PROJECT_ROOT / "final_dataset"
REPOS_YAML = PROJECT_ROOT / "config" / "repos.yaml"


def count_json_items(directory: Path, glob_pattern: str = "*.json") -> int:
    """Count total items across all JSON array files in a directory."""
    total = 0
    if not directory.exists():
        return 0
    for f in directory.glob(glob_pattern):
        try:
            data = json.loads(f.read_text())
            total += len(data) if isinstance(data, list) else 0
        except (json.JSONDecodeError, Exception):
            pass
    return total


def count_jsonl_lines(filepath: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    if not filepath.exists():
        return 0
    return sum(1 for line in filepath.read_text().splitlines() if line.strip())


def get_status() -> dict:
    """Get current pipeline state."""
    # Repos
    repos_cloned = len(list(REPOS_DIR.iterdir())) if REPOS_DIR.exists() else 0
    repos_extracted = len(list(EXTRACTED_DIR.glob("*.json"))) if EXTRACTED_DIR.exists() else 0

    # Judging
    extracted_names = {f.stem for f in EXTRACTED_DIR.glob("*.json")} if EXTRACTED_DIR.exists() else set()
    passed_names = {f.stem for f in PASSED_DIR.glob("*.json")} if PASSED_DIR.exists() else set()
    failed_names = {f.stem for f in FAILED_DIR.glob("*.json")} if FAILED_DIR.exists() else set()
    judged_names = passed_names | failed_names
    unjudged = sorted(extracted_names - judged_names)

    real_passed = count_json_items(PASSED_DIR)
    real_failed = count_json_items(FAILED_DIR)

    # Synthetic
    synthetic_generated = count_json_items(GENERATED_DIR)
    synthetic_passed = count_json_items(JUDGED_DIR, "passed_*.json")
    synthetic_failed = count_json_items(JUDGED_DIR, "failed_*.json")

    # Judge training
    judge_high = count_json_items(JUDGE_TRAINING_DIR, "high_quality*.json")
    judge_low = count_json_items(JUDGE_TRAINING_DIR, "low_quality*.json")
    judge_synth = count_json_items(JUDGE_TRAINING_DIR, "synthetic*.json")

    # CoT
    cot_total = count_json_items(COT_DIR)

    # Gap analysis
    has_gap_report = GAP_REPORT.exists()
    gap_deficit = 0
    if has_gap_report:
        gaps = json.loads(GAP_REPORT.read_text()).get("gaps", {})
        gap_deficit = sum(g["deficit"] for g in gaps.values())

    # Export
    has_export = (FINAL_DIR / "openai_train.jsonl").exists()
    export_count = count_jsonl_lines(FINAL_DIR / "wordpress_finetune.jsonl") if has_export else 0

    return {
        "repos_cloned": repos_cloned,
        "repos_extracted": repos_extracted,
        "repos_unjudged": unjudged,
        "repos_unjudged_count": len(unjudged),
        "real_passed": real_passed,
        "real_failed": real_failed,
        "pass_rate": round(real_passed / (real_passed + real_failed) * 100, 1) if (real_passed + real_failed) > 0 else 0,
        "synthetic_generated": synthetic_generated,
        "synthetic_passed": synthetic_passed,
        "synthetic_failed": synthetic_failed,
        "judge_high": judge_high,
        "judge_low": judge_low,
        "judge_synth": judge_synth,
        "judge_total": judge_high + judge_low + judge_synth,
        "cot_total": cot_total,
        "gap_deficit": gap_deficit,
        "has_gap_report": has_gap_report,
        "has_export": has_export,
        "export_count": export_count,
    }


def get_plan(status: dict) -> dict:
    """Determine what actions are needed based on current state."""
    actions = []
    phase = "complete"

    # Phase 1: Clone & Extract
    if not REPOS_YAML.exists():
        actions.append({
            "step": "generate_repos_yaml",
            "command": "python scripts/csv_to_repos.py",
            "type": "script",
            "description": "Generate repos.yaml from CSV data",
        })
        phase = "phase1_setup"

    if status["repos_cloned"] == 0:
        actions.append({
            "step": "clone_repos",
            "command": "python scripts/phase1_clone.py",
            "type": "script",
            "description": "Clone all repositories from repos.yaml",
        })
        phase = "phase1_clone"

    if status["repos_extracted"] == 0 and status["repos_cloned"] > 0:
        actions.append({
            "step": "extract_functions",
            "command": "python scripts/phase1_extract.py",
            "type": "script",
            "description": "Extract PHP functions from cloned repos",
        })
        phase = "phase1_extract"

    # Phase 1: Judge (via agents)
    if status["repos_unjudged_count"] > 0:
        # Group unjudged repos into agent batches
        batches = []
        remaining = list(status["repos_unjudged"])
        batch_size = 5
        while remaining:
            batch = remaining[:batch_size]
            remaining = remaining[batch_size:]
            batches.append(batch)

        actions.append({
            "step": "judge_repos",
            "type": "agent",
            "description": f"Judge {status['repos_unjudged_count']} unjudged repos via Claude Code agents",
            "agent_count": len(batches),
            "batches": batches,
            "rubric": "config/judge_system.md",
            "input_dir": str(EXTRACTED_DIR),
            "output_passed": str(PASSED_DIR),
            "output_failed": str(FAILED_DIR),
        })
        phase = "phase1_judge"

    # Phase 2: Gap analysis (script)
    if not status["has_gap_report"] and status["real_passed"] > 0:
        actions.append({
            "step": "gap_analysis",
            "command": "python scripts/phase2_gap_analysis.py",
            "type": "script",
            "description": "Analyze taxonomy coverage gaps",
        })
        phase = "phase2_gaps"

    # Phase 2: Mutations (script)
    mutated_dir = PROJECT_ROOT / "phase2_synthetic" / "output" / "mutated"
    if not mutated_dir.exists() and status["real_passed"] > 0:
        actions.append({
            "step": "mutations",
            "command": "python scripts/phase2_mutate.py",
            "type": "script",
            "description": "Generate contrastive mutation pairs",
        })

    # Phase 2: Synthetic generation (agents)
    if status["synthetic_generated"] < TARGETS["synthetic_passed"] and status["has_gap_report"]:
        deficit = TARGETS["synthetic_passed"] - status["synthetic_generated"]
        actions.append({
            "step": "synthetic_generation",
            "type": "agent",
            "description": f"Generate ~{deficit} synthetic examples to fill taxonomy gaps",
            "agent_count": max(1, deficit // 25),
            "gap_report": str(GAP_REPORT),
            "output_dir": str(GENERATED_DIR),
        })
        phase = "phase2_generate"

    # Phase 2: Judge synthetics (agents)
    unjudged_synth = status["synthetic_generated"] - status["synthetic_passed"] - status["synthetic_failed"]
    if unjudged_synth > 0:
        actions.append({
            "step": "judge_synthetics",
            "type": "agent",
            "description": f"Judge {unjudged_synth} synthetic examples",
            "agent_count": max(1, unjudged_synth // 70),
            "input_dir": str(GENERATED_DIR),
            "output_dir": str(JUDGED_DIR),
        })
        phase = "phase2_judge_synth"

    # Phase 2: Judge training data (agents) — spawn until target
    for category, target, glob_pat in [
        ("judge_high", TARGETS["judge_high"], "high_quality*.json"),
        ("judge_low", TARGETS["judge_low"], "low_quality*.json"),
        ("judge_synth", TARGETS["judge_synth"], "synthetic*.json"),
    ]:
        current = count_json_items(JUDGE_TRAINING_DIR, glob_pat)
        if current < target:
            deficit = target - current
            batch_size = 200
            agent_count = max(1, (deficit + batch_size - 1) // batch_size)
            actions.append({
                "step": f"judge_training_{category}",
                "type": "agent",
                "description": f"Generate {deficit} {category} judge training examples ({current}/{target})",
                "agent_count": agent_count,
                "batch_size": batch_size,
                "deficit": deficit,
                "current": current,
                "target": target,
                "output_dir": str(JUDGE_TRAINING_DIR),
            })
            phase = "phase2_judge_training"

    # Phase 3: CoT (agents) — spawn until target
    if status["cot_total"] < TARGETS["cot_total"]:
        deficit = TARGETS["cot_total"] - status["cot_total"]
        actions.append({
            "step": "cot_reasoning",
            "type": "agent",
            "description": f"Generate {deficit} CoT reasoning examples ({status['cot_total']}/{TARGETS['cot_total']})",
            "agent_count": max(1, deficit // 80),
            "deficit": deficit,
            "output_dir": str(COT_DIR),
        })
        phase = "phase3_cot"

    # Export
    all_targets_met = all([
        status["real_passed"] >= TARGETS["real_code_passed"],
        status["synthetic_passed"] >= TARGETS["synthetic_passed"],
        status["judge_high"] >= TARGETS["judge_high"],
        status["judge_low"] >= TARGETS["judge_low"],
        status["judge_synth"] >= TARGETS["judge_synth"],
        status["cot_total"] >= TARGETS["cot_total"],
    ])

    if all_targets_met and not status["has_export"]:
        actions.append({
            "step": "merge_and_export",
            "type": "script",
            "description": "Merge all data and run export_dataset.py",
            "commands": [
                "python scripts/merge_dataset.py",
                "python scripts/export_dataset.py",
            ],
        })
        phase = "export"

    if all_targets_met and status["has_export"]:
        phase = "complete"

    return {
        "phase": phase,
        "all_targets_met": all_targets_met,
        "actions": actions,
        "action_count": len(actions),
        "targets": TARGETS,
    }


def print_status(status: dict):
    """Pretty-print pipeline status."""
    print("=" * 60)
    print("  PIPELINE STATUS")
    print("=" * 60)
    print()
    print(f"  {'Phase 1: Extract & Judge':─<40}")
    print(f"    Repos cloned:      {status['repos_cloned']:>6}")
    print(f"    Repos extracted:   {status['repos_extracted']:>6}")
    print(f"    Repos unjudged:    {status['repos_unjudged_count']:>6}")
    print(f"    Functions passed:  {status['real_passed']:>6} / {TARGETS['real_code_passed']:,} {'✓' if status['real_passed'] >= TARGETS['real_code_passed'] else ''}")
    print(f"    Functions failed:  {status['real_failed']:>6}")
    print(f"    Pass rate:         {status['pass_rate']:>5}%")
    print()
    print(f"  {'Phase 2: Synthetic & Judge Training':─<40}")
    print(f"    Synthetic gen:     {status['synthetic_generated']:>6}")
    print(f"    Synthetic passed:  {status['synthetic_passed']:>6} / {TARGETS['synthetic_passed']} {'✓' if status['synthetic_passed'] >= TARGETS['synthetic_passed'] else ''}")
    print(f"    Judge high-score:  {status['judge_high']:>6} / {TARGETS['judge_high']:,} {'✓' if status['judge_high'] >= TARGETS['judge_high'] else ''}")
    print(f"    Judge low-score:   {status['judge_low']:>6} / {TARGETS['judge_low']:,} {'✓' if status['judge_low'] >= TARGETS['judge_low'] else ''}")
    print(f"    Judge synth-score: {status['judge_synth']:>6} / {TARGETS['judge_synth']:,} {'✓' if status['judge_synth'] >= TARGETS['judge_synth'] else ''}")
    print()
    print(f"  {'Phase 3: CoT & Export':─<40}")
    print(f"    CoT total:         {status['cot_total']:>6} / {TARGETS['cot_total']} {'✓' if status['cot_total'] >= TARGETS['cot_total'] else ''}")
    print(f"    Export done:       {'Yes' if status['has_export'] else 'No'}")
    if status['has_export']:
        print(f"    Export examples:   {status['export_count']:>6}")
    print()

    all_met = all([
        status['real_passed'] >= TARGETS['real_code_passed'],
        status['synthetic_passed'] >= TARGETS['synthetic_passed'],
        status['judge_high'] >= TARGETS['judge_high'],
        status['judge_low'] >= TARGETS['judge_low'],
        status['judge_synth'] >= TARGETS['judge_synth'],
        status['cot_total'] >= TARGETS['cot_total'],
    ])
    print(f"  ALL TARGETS MET: {'✓ YES' if all_met else '✗ NO'}")


def print_plan(plan: dict):
    """Pretty-print action plan."""
    print()
    print("=" * 60)
    print(f"  ACTION PLAN — Phase: {plan['phase']}")
    print("=" * 60)

    if plan["all_targets_met"] and plan["phase"] == "complete":
        print("\n  All targets met. Pipeline complete. Nothing to do.")
        return

    print(f"\n  {plan['action_count']} action(s) needed:\n")
    for i, action in enumerate(plan["actions"], 1):
        atype = action["type"]
        icon = "🐍" if atype == "script" else "🤖"
        print(f"  {i}. {icon} [{atype}] {action['description']}")
        if "command" in action:
            print(f"     → {action['command']}")
        if "agent_count" in action:
            print(f"     → Spawn {action['agent_count']} agent(s)")
        if "deficit" in action:
            print(f"     → Deficit: {action['deficit']}")
        print()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "status":
        status = get_status()
        print_status(status)
    elif cmd == "plan":
        status = get_status()
        plan = get_plan(status)
        print_plan(plan)
        # Also output JSON for machine consumption
        print("\n--- JSON Plan ---")
        print(json.dumps(plan, indent=2, default=str))
    elif cmd == "plan-json":
        status = get_status()
        plan = get_plan(status)
        print(json.dumps(plan, indent=2, default=str))
    elif cmd == "status-json":
        status = get_status()
        print(json.dumps(status, indent=2, default=str))
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python scripts/pipeline_orchestrator.py [status|plan|plan-json|status-json]")
        sys.exit(1)
