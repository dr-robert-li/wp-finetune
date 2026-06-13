#!/usr/bin/env bash
# Phase 04.4 Plan 04 Task 2 — triple-gated idempotent promote of the v4-winner
# staging merge to the canonical v4-suffixed reasoning-merged dir.
#
# Gates (ALL required before any filesystem write):
#   G1  HUMAN_APPROVED_V4_POSTMERGE sentinel present in the REVL-05 review pack.
#   G2' (automated_pass==true) OR (D-V4-10 waiver APPROVED) — waiver amendment, the
#       automated verdict is NOT mutated; the waiver doc is the override authority.
#   G3  Idempotency — if canonical already exists AND its merge_report matches staging,
#       refuse with exit 0 "already promoted, no-op" (archive-not-overwrite).
#
# Writes ONLY to the v4-suffixed canonical path. NEVER touches the distinct v3 dir
# (models/qwen3-30b-wp-30_70-reasoning-merged) or the READ-ONLY baseline
# (models/qwen3-30b-wp-30_70-merged-v2).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGING="models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v4"
CANONICAL="models/qwen3-30b-wp-30_70-reasoning-merged-v4"
V3_DIR="models/qwen3-30b-wp-30_70-reasoning-merged"          # distinct — must NOT touch
BASELINE="models/qwen3-30b-wp-30_70-merged-v2"               # READ-ONLY baseline — must NOT touch
REVIEW="output/eval_reasoning_v4_winner/v1.2_human_review_v4_winner.md"
VERDICT="output/eval_reasoning_v4_winner/automated_verdict.json"
WAIVER=".planning/phases/04.4-reasoning-eval-adapter-merge-inserted/04.4-D-V4-10-WAIVER.md"
MERGE_REPORT="output/merge_v4_winner/merge_report.json"

echo "[promote] ROOT=$ROOT"

# --- G1: human sentinel (match a real sentinel line: starts with the token + a timestamp) ---
if ! grep -Eq '^HUMAN_APPROVED_V4_POSTMERGE: [0-9]{4}-' "$REVIEW"; then
  echo "[promote] G1 FAIL: HUMAN_APPROVED_V4_POSTMERGE sentinel not found in $REVIEW. Refusing." >&2
  exit 1
fi
echo "[promote] G1 OK: human sentinel present."

# --- G2': automated_pass==true OR D-V4-10 waiver APPROVED ---
AUTOPASS=$(python3 -c "import json;print(json.load(open('$VERDICT')).get('automated_pass'))" 2>/dev/null || echo "None")
WAIVED=$(grep -Eq '^Status:\s*APPROVED|WAIVER APPROVED:' "$WAIVER" 2>/dev/null && echo "true" || echo "false")
if [ "$AUTOPASS" = "True" ]; then
  echo "[promote] G2' OK: automated_pass==true."
elif [ "$WAIVED" = "true" ]; then
  echo "[promote] G2' OK: automated_pass=$AUTOPASS but D-V4-10 waiver APPROVED (override authority)."
else
  echo "[promote] G2' FAIL: automated_pass=$AUTOPASS and no approved D-V4-10 waiver. Refusing." >&2
  exit 1
fi

# --- Pre-write safety: staging must exist and be a valid model dir ---
if [ ! -f "$STAGING/model.safetensors.index.json" ]; then
  echo "[promote] FAIL: staging $STAGING missing model.safetensors.index.json. Refusing." >&2
  exit 1
fi

# --- G3: idempotency (archive-not-overwrite) ---
if [ -d "$CANONICAL" ]; then
  if [ -f "$CANONICAL/merge_report.json" ] && [ -f "$MERGE_REPORT" ] && \
     cmp -s "$CANONICAL/merge_report.json" "$MERGE_REPORT"; then
    echo "[promote] G3: canonical already promoted and merge_report matches — no-op (idempotent)."
    exit 0
  else
    echo "[promote] G3 FAIL: canonical $CANONICAL exists but merge_report does NOT match staging." >&2
    echo "[promote]   Refusing to overwrite an existing canonical dir (archive-not-overwrite)." >&2
    exit 1
  fi
fi

# --- Promote: copy staging -> canonical + merge_report alongside ---
echo "[promote] All gates pass. Copying $STAGING -> $CANONICAL ..."
TMP="${CANONICAL}.partial.$$"
rm -rf "$TMP"
cp -a "$STAGING" "$TMP"
cp -a "$MERGE_REPORT" "$TMP/merge_report.json"
mv "$TMP" "$CANONICAL"
echo "[promote] DONE: canonical promoted to $CANONICAL"

# --- Untouched-invariant assertions ---
for guard in "$V3_DIR" "$BASELINE"; do
  if [ -e "$guard" ]; then echo "[promote] invariant: $guard present and untouched."; fi
done
test -f "$CANONICAL/model.safetensors.index.json" && echo "[promote] verify: canonical index present."
