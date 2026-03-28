#!/usr/bin/env python3
"""Pipeline Orchestrator — checks state and reports what needs to be done.

This script is the brain of the pipeline. It:
1. Scans all output directories to determine current state
2. Computes dynamic percentage-based targets from actual data
3. Produces a structured action plan (JSON) for Claude Code to execute

Targets are percentage-based, derived from extracted+judged function counts:
- Judge training: % of passed/failed functions
- CoT: 4-way split (gen_pattern, judge_rubric, judge_contrastive, security)
  with floor of 500 per type OR 10% of base, whichever is larger
- Synthetic: fill taxonomy gaps until gap_report.json shows 0 deficit

Usage:
    python scripts/pipeline_orchestrator.py status    # Show current state
    python scripts/pipeline_orchestrator.py plan      # Show what needs doing
    python scripts/pipeline_orchestrator.py plan-json # Machine-readable plan
    python scripts/pipeline_orchestrator.py status-json

Claude Code reads the plan output and spawns agents accordingly.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Percentage-based target config ────────────────────────────────────
# All targets are derived from the actual judged function counts.
# Hard numbers are gone — percentages scale with the dataset.

TARGET_PERCENTAGES = {
    # Judge training: % of their respective source pools
    "judge_high_pct": 0.10,      # 10% of passed functions
    "judge_low_pct": 0.10,       # 10% of failed functions
    "judge_synth_pct": 0.10,     # 10% of synthetic passed

    # CoT: 4-way split, each is max(500, 10% of base)
    "cot_gen_pattern_pct": 0.10,       # 10% of passed (gen pattern reasoning)
    "cot_judge_rubric_pct": 0.10,      # 10% of (passed + failed) (rubric walkthrough)
    "cot_judge_contrastive_pct": 0.10, # 10% of failed (contrastive bad→fix)
    "cot_security_pct": 0.10,          # 10% of security-tagged functions
}

COT_FLOOR = 500  # Minimum examples per CoT type

# ── Paths ────────────────────────────────────────────────────────────
REPOS_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "repos"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "extracted"
PASSED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "passed"
FAILED_DIR = PROJECT_ROOT / "data" / "phase1_extraction" / "output" / "failed"
GAP_REPORT = PROJECT_ROOT / "data" / "phase2_synthetic" / "gap_report.json"
GENERATED_DIR = PROJECT_ROOT / "data" / "phase2_synthetic" / "output" / "generated"
JUDGED_DIR = PROJECT_ROOT / "data" / "phase2_synthetic" / "output" / "judged"
JUDGE_TRAINING_DIR = PROJECT_ROOT / "data" / "phase2_synthetic" / "output" / "judge_training"
COT_DIR = PROJECT_ROOT / "data" / "phase3_cot" / "output"
FINAL_DIR = PROJECT_ROOT / "data" / "final_dataset"
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


def count_security_tagged(passed_dir: Path, failed_dir: Path) -> int:
    """Count functions with security-relevant surface (superglobals, echo, forms)."""
    import re
    security_pattern = re.compile(
        r'\$_(GET|POST|REQUEST|COOKIE|FILES|SERVER)'
        r'|check_ajax_referer|wp_verify_nonce|current_user_can'
        r'|esc_html|esc_attr|esc_url|wp_kses'
        r'|sanitize_text_field|absint|intval'
        r'|wp_nonce_field|wp_create_nonce',
        re.MULTILINE
    )
    count = 0
    for d in [passed_dir, failed_dir]:
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                for fn in data:
                    if isinstance(fn, dict) and security_pattern.search(fn.get("body", "")):
                        count += 1
            except (json.JSONDecodeError, Exception):
                pass
    return count


def compute_targets(status: dict) -> dict:
    """Compute dynamic targets based on actual data counts.

    All targets scale with the dataset. No hardcoded numbers except
    the COT_FLOOR minimum per CoT type.
    """
    passed = status["real_passed"]
    failed = status["real_failed"]
    total_judged = passed + failed
    synth_passed = status["synthetic_passed"]
    security_tagged = status["security_tagged"]

    def cot_target(base: int, pct: float) -> int:
        """max(COT_FLOOR, pct * base)"""
        return max(COT_FLOOR, int(base * pct))

    return {
        # Phase 1: no target — judge everything extracted
        "real_code_passed": 0,  # Not a target, natural outcome

        # Phase 2: synthetic fills gap deficit (0 = no more needed)
        "synthetic_passed": max(status["gap_deficit"], 0),

        # Phase 2: judge training — % of source pools
        "judge_high": int(passed * TARGET_PERCENTAGES["judge_high_pct"]),
        "judge_low": int(failed * TARGET_PERCENTAGES["judge_low_pct"]),
        "judge_synth": int(synth_passed * TARGET_PERCENTAGES["judge_synth_pct"]),

        # Phase 3: CoT — 4-way split, max(500, 10% of base)
        "cot_gen_pattern": cot_target(passed, TARGET_PERCENTAGES["cot_gen_pattern_pct"]),
        "cot_judge_rubric": cot_target(total_judged, TARGET_PERCENTAGES["cot_judge_rubric_pct"]),
        "cot_judge_contrastive": cot_target(failed, TARGET_PERCENTAGES["cot_judge_contrastive_pct"]),
        "cot_security": cot_target(security_tagged, TARGET_PERCENTAGES["cot_security_pct"]),
    }


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

    # CoT — 4-way split
    cot_gen_pattern = count_json_items(COT_DIR, "cot_real*.json") + count_json_items(COT_DIR, "cot_gen_pattern*.json")
    cot_judge_rubric = count_json_items(COT_DIR, "cot_judge_rubric*.json")
    cot_judge_contrastive = count_json_items(COT_DIR, "cot_contrastive*.json") + count_json_items(COT_DIR, "cot_judge_contrastive*.json")
    cot_security = count_json_items(COT_DIR, "cot_security*.json")
    cot_synthetic = count_json_items(COT_DIR, "cot_synthetic*.json")
    cot_total = cot_gen_pattern + cot_judge_rubric + cot_judge_contrastive + cot_security + cot_synthetic

    # Security-tagged functions
    security_tagged = count_security_tagged(PASSED_DIR, FAILED_DIR)

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
        "total_judged": real_passed + real_failed,
        "pass_rate": round(real_passed / (real_passed + real_failed) * 100, 1) if (real_passed + real_failed) > 0 else 0,
        "synthetic_generated": synthetic_generated,
        "synthetic_passed": synthetic_passed,
        "synthetic_failed": synthetic_failed,
        "judge_high": judge_high,
        "judge_low": judge_low,
        "judge_synth": judge_synth,
        "judge_total": judge_high + judge_low + judge_synth,
        "cot_gen_pattern": cot_gen_pattern,
        "cot_judge_rubric": cot_judge_rubric,
        "cot_judge_contrastive": cot_judge_contrastive,
        "cot_security": cot_security,
        "cot_total": cot_total,
        "security_tagged": security_tagged,
        "gap_deficit": gap_deficit,
        "has_gap_report": has_gap_report,
        "has_export": has_export,
        "export_count": export_count,
    }


def get_plan(status: dict) -> dict:
    """Determine what actions are needed based on current state."""
    actions = []
    phase = "complete"
    targets = compute_targets(status)

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
            "command": "python -m scripts.phase1_clone",
            "type": "script",
            "description": "Clone all repositories from repos.yaml",
        })
        phase = "phase1_clone"

    if status["repos_extracted"] == 0 and status["repos_cloned"] > 0:
        actions.append({
            "step": "extract_functions",
            "command": "python -m scripts.phase1_extract",
            "type": "script",
            "description": "Extract PHP functions from cloned repos",
        })
        phase = "phase1_extract"

    # Phase 1: Judge ALL extracted repos (no target — judge everything)
    if status["repos_unjudged_count"] > 0:
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
    mutated_dir = PROJECT_ROOT / "data" / "phase2_synthetic" / "output" / "mutated"
    if not mutated_dir.exists() and status["real_passed"] > 0:
        actions.append({
            "step": "mutations",
            "command": "python scripts/phase2_mutate.py",
            "type": "script",
            "description": "Generate contrastive mutation pairs",
        })

    # Phase 2: Synthetic generation — fill gap deficit (not hardcoded target)
    if status["gap_deficit"] > 0 and status["has_gap_report"]:
        actions.append({
            "step": "synthetic_generation",
            "type": "agent",
            "description": f"Generate ~{status['gap_deficit']} synthetic examples to fill taxonomy gaps",
            "agent_count": max(1, status["gap_deficit"] // 25),
            "gap_report": str(GAP_REPORT),
            "output_dir": str(GENERATED_DIR),
        })
        phase = "phase2_generate"

    # Phase 2: Judge synthetics
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

    # Phase 2: Judge training data — percentage-based targets
    for category, current_key, target_key, glob_pat in [
        ("judge_high", "judge_high", "judge_high", "high_quality*.json"),
        ("judge_low", "judge_low", "judge_low", "low_quality*.json"),
        ("judge_synth", "judge_synth", "judge_synth", "synthetic*.json"),
    ]:
        current = status[current_key]
        target = targets[target_key]
        if target > 0 and current < target:
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

    # Phase 3: CoT — 4-way split with percentage targets and floor
    cot_types = [
        {
            "step": "cot_gen_pattern",
            "current": status["cot_gen_pattern"],
            "target": targets["cot_gen_pattern"],
            "source": "passed functions",
            "description_template": "Gen Pattern CoT: requirement → pattern → implementation → reasoning",
            "output_prefix": "cot_gen_pattern",
            "source_dir": str(PASSED_DIR),
        },
        {
            "step": "cot_judge_rubric",
            "current": status["cot_judge_rubric"],
            "target": targets["cot_judge_rubric"],
            "source": "mixed passed+failed functions",
            "description_template": "Judge Rubric CoT: code → walk 9 dimensions → scores → verdict",
            "output_prefix": "cot_judge_rubric",
            "source_dir": str(PASSED_DIR) + "," + str(FAILED_DIR),
        },
        {
            "step": "cot_judge_contrastive",
            "current": status["cot_judge_contrastive"],
            "target": targets["cot_judge_contrastive"],
            "source": "failed functions",
            "description_template": "Judge Contrastive CoT: bad code → issues → fixes → good version",
            "output_prefix": "cot_judge_contrastive",
            "source_dir": str(FAILED_DIR),
        },
        {
            "step": "cot_security",
            "current": status["cot_security"],
            "target": targets["cot_security"],
            "source": "security-tagged functions",
            "description_template": "Security CoT: security analysis → nonce/cap/escape → verdict",
            "output_prefix": "cot_security",
            "source_dir": str(PASSED_DIR) + "," + str(FAILED_DIR),
        },
    ]

    for cot in cot_types:
        if cot["target"] > 0 and cot["current"] < cot["target"]:
            deficit = cot["target"] - cot["current"]
            actions.append({
                "step": cot["step"],
                "type": "agent",
                "description": f"{cot['description_template']} ({cot['current']}/{cot['target']}, deficit {deficit})",
                "agent_count": max(1, deficit // 80),
                "deficit": deficit,
                "current": cot["current"],
                "target": cot["target"],
                "source": cot["source"],
                "output_prefix": cot["output_prefix"],
                "output_dir": str(COT_DIR),
                "source_dir": cot["source_dir"],
            })
            phase = "phase3_cot"

    # Check all targets met
    all_targets_met = all([
        status["repos_unjudged_count"] == 0,
        status["gap_deficit"] == 0 or not status["has_gap_report"],
        status["judge_high"] >= targets["judge_high"],
        status["judge_low"] >= targets["judge_low"],
        status["judge_synth"] >= targets["judge_synth"],
        status["cot_gen_pattern"] >= targets["cot_gen_pattern"],
        status["cot_judge_rubric"] >= targets["cot_judge_rubric"],
        status["cot_judge_contrastive"] >= targets["cot_judge_contrastive"],
        status["cot_security"] >= targets["cot_security"],
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
        "targets": targets,
        "percentages": TARGET_PERCENTAGES,
        "cot_floor": COT_FLOOR,
    }


def print_status(status: dict):
    """Pretty-print pipeline status."""
    targets = compute_targets(status)

    print("=" * 70)
    print("  PIPELINE STATUS (percentage-based targets)")
    print("=" * 70)
    print()
    print(f"  {'Phase 1: Extract & Judge':─<50}")
    print(f"    Repos cloned:        {status['repos_cloned']:>6}")
    print(f"    Repos extracted:     {status['repos_extracted']:>6}")
    print(f"    Repos unjudged:      {status['repos_unjudged_count']:>6}   {'✓' if status['repos_unjudged_count'] == 0 else '✗ judge everything'}")
    print(f"    Functions passed:    {status['real_passed']:>6}")
    print(f"    Functions failed:    {status['real_failed']:>6}")
    print(f"    Total judged:        {status['total_judged']:>6}")
    print(f"    Pass rate:           {status['pass_rate']:>5}%")
    print(f"    Security-tagged:     {status['security_tagged']:>6}")
    print()
    print(f"  {'Phase 2: Synthetic & Judge Training':─<50}")
    print(f"    Synthetic gen:       {status['synthetic_generated']:>6}")
    print(f"    Synthetic passed:    {status['synthetic_passed']:>6}   (gap deficit: {status['gap_deficit']})")
    _pct_line("Judge high-score", status["judge_high"], targets["judge_high"], "10% of passed")
    _pct_line("Judge low-score", status["judge_low"], targets["judge_low"], "10% of failed")
    _pct_line("Judge synth-score", status["judge_synth"], targets["judge_synth"], "10% of synth passed")
    print()
    print(f"  {'Phase 3: CoT (4-way split)':─<50}")
    _pct_line("Gen Pattern CoT", status["cot_gen_pattern"], targets["cot_gen_pattern"], f"max(500, 10% of {status['real_passed']})")
    _pct_line("Judge Rubric CoT", status["cot_judge_rubric"], targets["cot_judge_rubric"], f"max(500, 10% of {status['total_judged']})")
    _pct_line("Judge Contrastive CoT", status["cot_judge_contrastive"], targets["cot_judge_contrastive"], f"max(500, 10% of {status['real_failed']})")
    _pct_line("Security CoT", status["cot_security"], targets["cot_security"], f"max(500, 10% of {status['security_tagged']})")
    print(f"    CoT total:           {status['cot_total']:>6}")
    print()
    print(f"  {'Export':─<50}")
    print(f"    Export done:         {'Yes' if status['has_export'] else 'No'}")
    if status["has_export"]:
        print(f"    Export examples:     {status['export_count']:>6}")
    print()

    plan = get_plan(status)
    print(f"  ALL TARGETS MET: {'✓ YES' if plan['all_targets_met'] else '✗ NO'}")


def _pct_line(label: str, current: int, target: int, formula: str):
    """Print a single status line with current/target and pass/fail."""
    icon = "✓" if current >= target else ""
    print(f"    {label + ':':24s} {current:>6} / {target:<6} {icon:2s} ({formula})")


def print_plan(plan: dict):
    """Pretty-print action plan."""
    print()
    print("=" * 70)
    print(f"  ACTION PLAN — Phase: {plan['phase']}")
    print("=" * 70)

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
