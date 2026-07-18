#!/usr/bin/env bash
# launch_validated_smoke.sh — RVAL-05 / Phase 08.2
#
# PURPOSE: Assemble the two per-seed rl_train.py commands for the gated
#          50/250-step smoke, then BY DEFAULT only PRINT them (dry-print).
#
# HARD GUARD: This script does NOT execute training unless the operator
#             explicitly passes --i-understand-this-spends-gpu.
#             Without that flag, the script prints both commands + a
#             pointer to the runbook, then exits 0 without touching GPU/Tinker.
#
# SPEC-ONLY: Phase 08.2 running this script = a $0 dry-print.
#            Execution is a future gated Phase 9 rerun per:
#            .planning/phases/08.2-reward-validity/08.2-SMOKE-RUNBOOK.md
#
# Usage:
#   bash scripts/launch_validated_smoke.sh                     # dry-print (default)
#   bash scripts/launch_validated_smoke.sh --i-understand-this-spends-gpu  # LAUNCH (Phase 9 rerun only)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SWEEP_RESULTS="$REPO_ROOT/output/reward_validity/sweep_results.json"
RUNBOOK="$REPO_ROOT/.planning/phases/08.2-reward-validity/08.2-SMOKE-RUNBOOK.md"

# ---------------------------------------------------------------------------
# Guard: detect confirm flag
# ---------------------------------------------------------------------------
CONFIRM_FLAG=""
for arg in "$@"; do
    if [ "$arg" = "--i-understand-this-spends-gpu" ]; then
        CONFIRM_FLAG="yes"
    fi
done

# ---------------------------------------------------------------------------
# Resolve (form, weight) from sweep_results.json
# selected = null is expected (Plan 04 documented escalation).
# When null, fall back to the top oracle-valid ranked entry:
#   filter valid==true, sort by ci_lo desc, first occurrence.
# From the actual sweep_results.json this is hybrid@0.8 (ci_lo=0.3725).
# ---------------------------------------------------------------------------
PYTHON="$REPO_ROOT/.venv-tinker/bin/python"

if [ ! -f "$SWEEP_RESULTS" ]; then
    echo "WARNING: sweep_results.json not found at $SWEEP_RESULTS"
    echo "Using fallback candidate: form=hybrid, calib_weight=0.8 (SWEEP-SELECTION.md recommendation)"
    CALIB_FORM="hybrid"
    CALIB_WEIGHT="0.8"
else
    # Parse with Python — deterministic, no jq dependency
    RESOLVED=$(SWEEP_RESULTS_PATH="$SWEEP_RESULTS" "$PYTHON" - <<'PYEOF'
import json, sys, os

# Path passed via env variable (avoids __file__ = <stdin> issue in heredoc)
sweep_path = os.environ.get("SWEEP_RESULTS_PATH", "output/reward_validity/sweep_results.json")

with open(sweep_path) as f:
    data = json.load(f)

selected = data.get("selected")
if selected and selected.get("form") and selected.get("calib_weight") is not None:
    form = selected["form"]
    weight = selected["calib_weight"]
else:
    # selected is null (documented Plan 04 escalation) — fall back to top
    # oracle-valid ranked entry: valid==true, max ci_lo, first-occurrence tiebreak.
    valid_rows = [r for r in data.get("ranked", []) if r.get("valid")]
    if not valid_rows:
        print("ERROR: no valid rows in ranked table", file=sys.stderr)
        sys.exit(1)
    best = max(valid_rows, key=lambda r: r["ci_lo"])
    form = best["form"]
    weight = best["calib_weight"]

print(f"{form} {weight}")
PYEOF
)
    CALIB_FORM=$(echo "$RESOLVED" | awk '{print $1}')
    CALIB_WEIGHT=$(echo "$RESOLVED" | awk '{print $2}')
fi

# ---------------------------------------------------------------------------
# Config constants (from runbook + rl_train._parse_args)
# ---------------------------------------------------------------------------
TOTAL_STEPS=250
CHECKPOINT_EVERY=50
CODEGEN_PROBE_EVERY=50
CODEGEN_BAR=0.4616
JUDGE_MAX_NEW_TOKENS=4096

# Warm-start path — operator must supply via INIT_FROM env or edit this line.
# REQUIRED for a real RLEV-01 run (D-09-04). Default is a placeholder.
INIT_FROM="${INIT_FROM:-tinker://<RUN>:train:0/sampler_weights/wp-reasoning-v4-r32-rp30-ep3}"

# Judge endpoint — operator must supply via JUDGE_BASE_URL env or edit this line.
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8000/v1}"

# ---------------------------------------------------------------------------
# Assemble per-seed commands
# ---------------------------------------------------------------------------
CMD_SEED_A=(
    ".venv-tinker/bin/python" "scripts/rl_train.py"
    "--init-from" "$INIT_FROM"
    "--lora-seed" "12345"
    "--total-steps" "$TOTAL_STEPS"
    "--checkpoint-every" "$CHECKPOINT_EVERY"
    "--codegen-probe-every" "$CODEGEN_PROBE_EVERY"
    "--codegen-bar" "$CODEGEN_BAR"
    "--calib-form" "$CALIB_FORM"
    "--calib-weight" "$CALIB_WEIGHT"
    "--judge-base-url" "$JUDGE_BASE_URL"
    "--judge-max-new-tokens" "$JUDGE_MAX_NEW_TOKENS"
)

CMD_SEED_B=(
    ".venv-tinker/bin/python" "scripts/rl_train.py"
    "--init-from" "$INIT_FROM"
    "--lora-seed" "99999"
    "--total-steps" "$TOTAL_STEPS"
    "--checkpoint-every" "$CHECKPOINT_EVERY"
    "--codegen-probe-every" "$CODEGEN_PROBE_EVERY"
    "--codegen-bar" "$CODEGEN_BAR"
    "--calib-form" "$CALIB_FORM"
    "--calib-weight" "$CALIB_WEIGHT"
    "--judge-base-url" "$JUDGE_BASE_URL"
    "--judge-max-new-tokens" "$JUDGE_MAX_NEW_TOKENS"
)

# ---------------------------------------------------------------------------
# Print the assembled commands (always — dry-print is the default path)
# ---------------------------------------------------------------------------
echo "=== launch_validated_smoke.sh — Phase 08.2 RVAL-05 ==="
echo ""
echo "Resolved reward config from sweep_results.json:"
echo "  form        : $CALIB_FORM"
echo "  calib_weight: $CALIB_WEIGHT"
echo "  (selected=null in sweep; fallback to top oracle-valid ranked entry)"
echo ""
echo "--- Seed A (--lora-seed 12345) ---"
echo "${CMD_SEED_A[*]}"
echo ""
echo "--- Seed B (--lora-seed 99999) ---"
echo "${CMD_SEED_B[*]}"
echo ""
echo "Kill-at-50 gate: read teacher-Spearman trend + echo <= 0.30 at step 50."
echo "If validated metric is flat, KILL. Do not push to 250, never to 500."
echo ""
echo "Runbook: $RUNBOOK"
echo ""

# ---------------------------------------------------------------------------
# Guard: refuse to launch without explicit confirm flag
# ---------------------------------------------------------------------------
if [ -z "$CONFIRM_FLAG" ]; then
    echo "--- DRY-PRINT ONLY (Phase 08.2 boundary) ---"
    echo "To LAUNCH training (gated Phase 9 rerun ONLY), pass:"
    echo "  bash scripts/launch_validated_smoke.sh --i-understand-this-spends-gpu"
    echo ""
    echo "Phase 08.2 does NOT spend GPU/Tinker. This script is a $0 dry-print in this phase."
    exit 0
fi

# ---------------------------------------------------------------------------
# LAUNCH PATH — only reached with --i-understand-this-spends-gpu
# This path is for the FUTURE GATED PHASE 9 RERUN only.
# ---------------------------------------------------------------------------
echo "=== LAUNCH CONFIRMED: starting training (gated Phase 9 rerun) ==="
echo ""

# Validate required env before spending GPU
if echo "$INIT_FROM" | grep -q '<RUN>'; then
    echo "ERROR: INIT_FROM is still the placeholder value."
    echo "  Set: export INIT_FROM='tinker://<actual-run>:train:0/sampler_weights/...'"
    exit 1
fi

echo "--- Launching Seed A (--lora-seed 12345) ---"
cd "$REPO_ROOT"
PYTHONPATH=. "${CMD_SEED_A[@]}" &
PID_A=$!
echo "Seed A PID: $PID_A"

echo ""
echo "--- Launching Seed B (--lora-seed 99999) ---"
PYTHONPATH=. "${CMD_SEED_B[@]}" &
PID_B=$!
echo "Seed B PID: $PID_B"

echo ""
echo "Both seeds launched. Monitor via:"
echo "  tail -f output/rl_checkpoints/metrics/rl_metrics.jsonl"
echo ""
echo "KILL-AT-50: At step 50, perform the manual read protocol from the runbook."
echo "If teacher-Spearman trend is flat OR echo > 0.30 OR codegen < 0.4616: kill both PIDs."
echo "  kill $PID_A $PID_B"
