"""Wave orchestrator — manages sequential agent batches for training data generation.

Usage:
  python scripts/orchestrator.py spawn --stream both --rounds 20
  python scripts/orchestrator.py check
  python scripts/orchestrator.py merge
  python scripts/orchestrator.py validate
"""
import json, argparse, sys, random, re
from pathlib import Path
from datetime import datetime, timezone

random.seed(42)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Config
BATCH_SIZE = 5  # functions per agent
WAVE_DIR = Path("data/phase4_reasoning/_orchestrator/wave_0001")
OUTPUT_DIR = Path("data/phase4_reasoning/_orchestrator")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Targets
TARGETS = {"cot": 5000, "ctf": 1000}


# ─── Helpers ──
def get_inline_cot_exemplars():
    seeds = json.loads((PROJECT_ROOT / "data/seeds/ugc_seeds.json").read_text())
    boundary = json.loads((PROJECT_ROOT / "data/seeds/ugc_boundary_seeds.json").read_text())
    all_seeds = [s for s in seeds + boundary if s.get("seed_type") == "deep_judge_cot"]
    boundary_seeds = [s for s in all_seeds if s.get("defect_subtlety") == "boundary"]
    exemplars = boundary_seeds[:2] if boundary_seeds else all_seeds[:2]
    blocks = []
    for s in exemplars:
        code = s.get("code", "")[:1500]
        reason = json.dumps(s.get("human_reasoning", {}), indent=2)[:1500]
        blocks.append(f"=== EXEMPLAR ===\nCODE:\n{code}\n\nREASONING:\n{reason}")
    return "\n\n".join(blocks)


def get_inline_ctf_exemplars():
    seeds = json.loads((PROJECT_ROOT / "data/seeds/ugc_seeds.json").read_text())
    boundary = json.loads((PROJECT_ROOT / "data/seeds/ugc_boundary_seeds.json").read_text())
    all_seeds = [s for s in seeds + boundary if s.get("seed_type") == "critique_then_fix"]
    boundary_seeds = [s for s in all_seeds if s.get("defect_subtlety") == "boundary"]
    exemplars = boundary_seeds[:2] if boundary_seeds else all_seeds[:2]
    blocks = []
    for s in exemplars:
        code = s.get("code", "")[:1500]
        crit = json.dumps(s.get("human_critique", {}), indent=2)[:1500]
        fix = s.get("corrected_code", "")[:1500]
        blocks.append(f"=== EXEMPLAR ===\nDEFECTIVE:\n{code}\n\nCRITIQUE:\n{crit}\n\nFIX:\n{fix}")
    return "\n\n".join(blocks)


def load_functions(source):
    base = Path(f"data/phase1_extraction/output/{source}/")
    if not base.exists():
        return []
    funcs = []
    for f in sorted(base.glob("*.json")):
        data = json.loads(f.read_text())
        for fn in (data if isinstance(data, list) else [data]):
            if isinstance(fn, dict) and "function_name" in fn:
                funcs.append({
                    "source_file": f.name,
                    "function_name": fn["function_name"],
                    "body": fn.get("body", ""),
                    "line_count": fn.get("line_count", 0),
                })
    return funcs


def build_cot_prompt(idx, batch_funcs, exemplars):
    batch_json = json.dumps(batch_funcs, indent=2)
    return f"""You are generating deep judge Chain-of-Thought training examples for WordPress code quality analysis.

FEW-SHOT EXEMPLARS (2 examples from human-curated seeds — use as style guide):
{exemplars}

SOURCE FUNCTIONS ({BATCH_SIZE}):
{batch_json}

For EACH function, generate a deep judge CoT example:
{{
  'code': <full PHP>,
  'source_file': <from input>,
  'function_name': <from input>,
  'reasoning': {{
    'verdict': 'PASS'|'FAIL',
    'dimension_analysis': {{
      'wpcs_compliance': {{'score': <1-10>, 'analysis': '<text>'}},
      'sql_safety': {{'score': <1-10>, 'analysis': '<text>'}},
      'security': {{'score': <1-10>, 'analysis': '<text>'}},
      'performance': {{'score': <1-10>, 'analysis': '<text>'}},
      'wp_api_usage': {{'score': <1-10>, 'analysis': '<text>'}},
      'code_quality': {{'score': <1-10>, 'analysis': '<text>'}},
      'dependency_integrity': {{'score': <1-10>, 'analysis': '<text>'}},
      'i18n': {{'score': <1-10>, 'analysis': '<text>'}},
      'accessibility': {{'score': <1-10>, 'analysis': '<text>'}}
    }},
    'overall_score': <0-100>,
    'key_observation': '<one-sentence>'
  }},
  'dimensions_addressed': ['wpcs_compliance','sql_safety','security','performance','wp_api_usage','code_quality','dependency_integrity','i18n','accessibility'],
  'generation_method': 'claude_code_agent_few_shot'
}}

RULES:
- ALL 9 dimensions required. Every dimension gets score (1-10) + analysis.
- Score 1-10 with real analysis text. Max 2 dimensions N/A (score=null) with "not applicable" + 20+ char justification.
- Cite specific WP APIs by EXACT name from source code (wpdb->prepare(), wp_verify_nonce(), esc_html(), current_user_can(), check_ajax_referer(), esc_attr(), esc_url(), wp_kses). Do NOT invent names.
- Be thorough and specific about code patterns.

Write the JSON array of {BATCH_SIZE} examples to:
data/phase4_reasoning/_orchestrator/wave_0001/batch_cot_r{idx:04d}.json

Use the Write tool. Do NOT use the Anthropic API. After writing, print: 'BATCH COT R{idx} COMPLETE: {BATCH_SIZE} examples'
"""


def build_ctf_prompt(idx, batch_funcs, exemplars):
    batch_json = json.dumps(batch_funcs, indent=2)
    return f"""You are generating critique-then-fix training examples for WordPress code quality.

FEW-SHOT EXEMPLARS (2 examples from human-curated seeds — use as style guide):
{exemplars}

DEFECTIVE FUNCTIONS ({BATCH_SIZE}):
{batch_json}

For EACH function, generate a critique-then-fix example:
{{
  'source_file': <from input>,
  'function_name': <from input>,
  'defective_code': <full PHP>,
  'critique': {{
    'summary': '<overall critique>',
    'dimensions': {{
      'wpcs_compliance': {{'severity': 'critical'|'high'|'medium'|'low', 'issue': '<text>', 'fix': '<text>'}},
      'sql_safety': {{'severity': 'critical'|'high'|'medium'|'low', 'issue': '<text>', 'fix': '<text>'}},
      'security': {{'severity': 'critical'|'high'|'medium'|'low', 'issue': '<text>', 'fix': '<text>'}},
      'performance': {{'severity': 'critical'|'high'|'medium'|'low', 'issue': '<text>', 'fix': '<text>'}},
      'wp_api_usage': {{'severity': 'critical'|'high'|'medium'|'low', 'issue': '<text>', 'fix': '<text>'}},
      'code_quality': {{'severity': 'critical'|'high'|'medium'|'low', 'issue': '<text>', 'fix': '<text>'}},
      'dependency_integrity': {{'severity': 'critical'|'high'|'medium'|'low', 'issue': '<text>', 'fix': '<text>'}},
      'i18n': {{'severity': 'critical'|'high'|'medium'|'low', 'issue': '<text>', 'fix': '<text>'}},
      'accessibility': {{'severity': 'critical'|'high'|'medium'|'low', 'issue': '<text>', 'fix': '<text>'}}
    }},
    'key_observation': '<one-sentence>'
  }},
  'corrected_code': '<full PHP with critical+high fixed>',
  'dimensions_addressed': ['wpcs_compliance','sql_safety','security','performance','wp_api_usage','code_quality','dependency_integrity','i18n','accessibility'],
  'generation_method': 'claude_code_agent_few_shot'
}}

RULES:
- ALL 9 dimensions with severity in [critical, high, medium, low]. NO N/A.
- corrected_code MUST be valid PHP and actually fix critical/high issues.
- corrected_code MUST SUBSTANTIALLY DIFFER from defective_code.
- Cite specific WP APIs by EXACT name. Do NOT invent names.

Write the JSON array of {BATCH_SIZE} examples to:
data/phase4_reasoning/_orchestrator/wave_0001/batch_ctf_r{idx:04d}.json

Use the Write tool. Do NOT use the Anthropic API. After writing, print: 'BATCH CTF R{idx} COMPLETE: {BATCH_SIZE} examples'
"""


# ─── Spawn mode ──
def spawn(rounds, stream):
    """Prepare round inputs and print agent spawn commands."""
    WAVE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cot_exemplars = get_inline_cot_exemplars()
    ctf_exemplars = get_inline_ctf_exemplars()
    cot_funcs = load_functions("passed")
    ctf_funcs = load_functions("failed")
    random.shuffle(cot_funcs)
    random.shuffle(ctf_funcs)

    state = json.loads((OUTPUT_DIR / "state.json").read_text()) if (OUTPUT_DIR / "state.json").exists() else {}
    cot_start = state.get("cot_next_idx", 0)
    ctf_start = state.get("ctf_next_idx", 0)

    print(f"\n{'='*60}")
    print(f"SPAWNING {rounds} rounds")
    print(f"{'='*60}")

    streams_to_process = [stream] if stream != "both" else ["cot", "ctf"]
    for s in streams_to_process:
            funcs = cot_funcs if s == "cot" else ctf_funcs
            exemplars = cot_exemplars if s == "cot" else ctf_exemplars
            prefix = "cot" if s == "cot" else "ctf"
            remaining = max(0, TARGETS[s] - state.get("total_accepted", {}).get(s, 0))
            if remaining <= 0:
                print(f"\n{s.upper()}: TARGET MET")
                continue
            needed = (remaining + BATCH_SIZE - 1) // BATCH_SIZE
            n = min(rounds, needed)
            start = cot_start if s == "cot" else ctf_start
            print(f"\n{s.upper()}: {remaining} needed, {n} rounds, starting at idx {start}")

            for r in range(n):
                idx = start + r
                batch = funcs[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
                (WAVE_DIR / f"_input_{prefix}_{idx:04d}.json").write_text(json.dumps(batch, indent=2))
                prompt = build_cot_prompt(idx, batch, exemplars) if s == "cot" else build_ctf_prompt(idx, batch, exemplars)
                print(f"\n  === {s.upper()} round {r+1}/{n} ===")
                print(f"  idx: {idx}")
                print(f"  input: _input_{prefix}_{idx:04d}.json ({(WAVE_DIR / f'_input_{prefix}_{idx:04d}.json').stat().st_size} bytes)")
                print(f"  prompt: {len(prompt)} chars")
                print(f"\n  Agent(description='{s.upper()} round {r+1}', prompt='''{prompt}''', run_in_background=true)")
                print(f"  Output: batch_{prefix}_r{idx:04d}.json")

            if s == "cot":
                state["cot_next_idx"] = start + n
            else:
                state["ctf_next_idx"] = start + n

    (OUTPUT_DIR / "state.json").write_text(json.dumps(state, indent=2))
    print(f"\nState saved: {OUTPUT_DIR / 'state.json'}")
    print(f"\nAfter all agents complete: python scripts/orchestrator.py merge")


def merge():
    """Merge wave outputs."""
    # Count existing batch files
    cot_batches = sorted(WAVE_DIR.glob("batch_cot_r*.json"))
    ctf_batches = sorted(WAVE_DIR.glob("batch_ctf_r*.json"))
    print(f"\n{'='*60}")
    print(f"MERGE — Batch files found:")
    print(f"{'='*60}")
    print(f"CoT batches: {len(cot_batches)}")
    print(f"CtF batches: {len(ctf_batches)}")
    for f in cot_batches[:5]:
        data = f.read_text().strip()
        count = len([l for l in data.split("\n") if l.strip()]) if data else 0
        print(f"  {f.name}: {count} examples")
    for f in ctf_batches[:5]:
        data = f.read_text().strip()
        count = len([l for l in data.split("\n") if l.strip()]) if data else 0
        print(f"  {f.name}: {count} examples")
    if len(cot_batches) > 5:
        print(f"  ... +{len(cot_batches)-5} more")
    if len(ctf_batches) > 5:
        print(f"  ... +{len(ctf_batches)-5} more")
    print(f"\nWrite output files and validate with: python scripts/orchestrator.py validate")


def validate():
    """Validate quality."""
    # Count examples
    cot_base = Path("data/phase4_reasoning/deep_judge_cot")
    ctf_base = Path("data/phase4_reasoning/critique_then_fix")
    cot_examples = []
    ctf_examples = []
    for f in sorted(cot_base.glob("wave_*_accepted.jsonl")):
        data = f.read_text().strip()
        for line in data.split("\n"):
            if line.strip():
                try:
                    cot_examples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    for f in sorted(ctf_base.glob("wave_*_accepted.jsonl")):
        data = f.read_text().strip()
        for line in data.split("\n"):
            if line.strip():
                try:
                    ctf_examples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"\n{'='*60}")
    print(f"VALIDATION")
    print(f"{'='*60}")
    for stream, examples, base in [("cot", cot_examples, cot_base), ("ctf", ctf_examples, ctf_base)]:
        target = TARGETS[stream]
        current = len(examples)
        pct = current / target * 100 if target > 0 else 0
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(f"\n{stream.upper()}: [{bar}] {current}/{target} ({pct:.0f}%)")
        if current >= target:
            print(f"  STATUS: TARGET MET")
        else:
            print(f"  STATUS: {target - current} remaining")
        # Quality checks
        if stream == "cot":
            if examples:
                # Check dimension coverage
                from scripts.generate_deep_judge_cot import REQUIRED_DIMENSIONS
                missing_dims = 0
                for ex in examples:
                    da = ex.get("reasoning", {}).get("dimension_analysis", {})
                    if any(d not in da for d in REQUIRED_DIMENSIONS):
                        missing_dims += 1
                print(f"  Dimension completeness: {len(examples) - missing_dims}/{len(examples)}")

    print(f"\nWave batch files: {len(list(WAVE_DIR.glob('batch_cot_r*.json')))} CoT + {len(list(WAVE_DIR.glob('batch_ctf_r*.json')))} CtF")


# ─── Main ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", choices=["cot", "ctf", "both"], default="both")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds to prepare")
    parser.add_argument("command", choices=["spawn", "merge", "validate", "plan"], default="plan")
    args = parser.parse_args()

    if args.command == "plan":
        state = json.loads((OUTPUT_DIR / "state.json").read_text()) if (OUTPUT_DIR / "state.json").exists() else {}
        print(f"\n{'='*60}")
        print(f"PLAN")
        print(f"{'='*60}")
        for s in ("cot", "ctf"):
            current = state.get("total_accepted", {}).get(s, 0)
            remaining = max(0, TARGETS[s] - current)
            if remaining <= 0:
                print(f"  {s.upper()}: TARGET MET ({current}/{TARGETS[s]})")
                continue
            rounds = (remaining + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"  {s.upper()}: need {remaining} more -> ~{rounds} rounds")
        print(f"\n  Total estimated rounds: ~{sum((TARGETS[s] - state.get('total_accepted', {}).get(s, 0) + BATCH_SIZE - 1) // BATCH_SIZE for s in ('cot', 'ctf') if TARGETS[s] > state.get('total_accepted', {}).get(s, 0))}")
        print(f"\n  To spawn: python scripts/orchestrator.py spawn --rounds 10")

    elif args.command == "spawn":
        spawn(args.rounds, args.stream)

    elif args.command == "merge":
        merge()

    elif args.command == "validate":
        validate()


if __name__ == "__main__":
    main()
