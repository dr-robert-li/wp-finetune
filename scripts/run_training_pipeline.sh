#!/usr/bin/env bash
# Full training pipeline — runs inside DGX Toolbox Unsloth Studio container.
# Idempotent: safe to re-run, completed steps are skipped.
#
# Usage:
#   ./scripts/run_training_pipeline.sh           # full pipeline
#   ./scripts/run_training_pipeline.sh --dry-run  # verify config only
#
# Prerequisites:
#   - DGX Toolbox Unsloth Studio container running with project mounted
#   - Training data at data/final_dataset/openai_train.jsonl
#   - Config at config/train_config.yaml
set -euo pipefail

CONTAINER="unsloth-studio"
WORKDIR="/workspace/wp-finetune"
DCRUN="docker exec -w $WORKDIR $CONTAINER"
DRY_RUN="${1:-}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Colors ──────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

step() { echo -e "\n${GREEN}=== $1 ===${NC}"; }
warn() { echo -e "${YELLOW}WARNING: $1${NC}"; }
fail() { echo -e "${RED}ERROR: $1${NC}"; exit 1; }

# ── 0. Container check ─────────────────────────────
step "Checking container"

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    warn "Container '$CONTAINER' not running. Launching via DGX Toolbox..."

    # Use dgx_toolbox.yaml to find toolbox path
    DGX_PATH="${DGX_TOOLBOX_PATH:-$(python3 -c "
import yaml
from pathlib import Path
cfg = yaml.safe_load(open('$PROJECT_ROOT/config/dgx_toolbox.yaml'))
print(Path(cfg.get('dgx_toolbox_path', '~/dgx-toolbox')).expanduser())
" 2>/dev/null || echo "$HOME/dgx-toolbox")}"

    export EXTRA_MOUNTS="$PROJECT_ROOT:$WORKDIR"
    bash "$DGX_PATH/containers/unsloth-studio.sh"

    echo "Waiting for container setup (30s)..."
    sleep 30
fi

# Verify project is mounted
if ! $DCRUN test -f config/train_config.yaml 2>/dev/null; then
    fail "Project not mounted in container. Re-launch with:
  export EXTRA_MOUNTS=\"$PROJECT_ROOT:$WORKDIR\"
  docker stop $CONTAINER; docker rm $CONTAINER
  ~/dgx-toolbox/containers/unsloth-studio.sh"
fi
echo "Container OK ✓"

# ── 0.5 Install deps if needed ──────────────────────
step "Checking dependencies"
if ! $DCRUN python -c "import trl" 2>/dev/null; then
    echo "Installing pinned dependencies..."
    $DCRUN pip install --no-deps \
        "transformers==4.56.2" "trl==0.24.0" "datasets==4.3.0" \
        "bitsandbytes==0.48.0" "huggingface-hub==0.34.1" \
        pyyaml python-dotenv scipy wandb peft hf_transfer 2>&1 | tail -3
fi
echo "Dependencies OK ✓"

# ── 1. GPU + Memory check ──────────────────────────
step "Pre-flight"
$DCRUN nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || fail "No GPU access"
echo "Memory:"
free -h | head -2
echo ""

# ── 2. Download model ──────────────────────────────
step "Download model"
$DCRUN python -m scripts.download_model

# ── 3. Extend tokenizer ────────────────────────────
step "Extend tokenizer"
$DCRUN python -m scripts.prepare_tokenizer

# Verify
$DCRUN python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('adapters/tokenizer')
gen = tok.encode('<wp_gen>', add_special_tokens=False)
judge = tok.encode('<wp_judge>', add_special_tokens=False)
assert len(gen) == 1 and len(judge) == 1, f'Token check failed: {gen}, {judge}'
print(f'Tokenizer OK: wp_gen={gen[0]}, wp_judge={judge[0]}')
"

# ── 4. Train ────────────────────────────────────────
step "Train model"
if [ "$DRY_RUN" = "--dry-run" ]; then
    $DCRUN python -m scripts.train_model --dry-run
    echo -e "\n${YELLOW}DRY RUN — training not started. Remove --dry-run to train.${NC}"
    exit 0
fi

$DCRUN python -m scripts.train_model

# ── 5. Merge adapter ───────────────────────────────
step "Merge adapter"
$DCRUN python -m scripts.merge_adapter

# ── 6. Summary ──────────────────────────────────────
step "Training pipeline complete"
echo "Adapter:  adapters/qwen3-wp/"
echo "Merged:   models/Qwen3-30B-A3B-merged/ (if merge passed)"
echo "Tokenizer: adapters/tokenizer/"
echo ""
echo "Next: /gsd:execute-phase 4 (evaluation)"
