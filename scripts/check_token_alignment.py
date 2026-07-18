"""BASE-02 gate: align model.config eos/pad token IDs to the tokenizer's.

Qwen3.6-35B-A3B (like other recent Qwen releases) ships a model.config whose
eos/pad token IDs do NOT match the tokenizer's (QwenLM/Qwen3.6 discussion #96,
maintainer-classified "working as intended" — the model.generation_config
carries a multi-stop eos list for thinking/non-thinking flexibility, but the
plain model.config text_config is left stale/unset). This is a known,
maintainer-acknowledged quirk, not something to "fix upstream" — this gate
detects it, aligns it in place, and proves a real generation stops naturally
before trusting the fix.

Per Pitfall 1 (.planning/phases/20-base-bring-up/20-RESEARCH.md): read
tokenizer.eos_token_id as authoritative, NEVER model.config.eos_token_id.

This module exposes three pure, unit-testable pieces plus a main() that
exercises them against the real downloaded base:

  - align_and_check(model_config, tokenizer): mutate + assert the invariant
  - classify_stopped_naturally(output_ids, max_tokens, eos_token_ids): bool
  - build_receipt(...): flat-JSON gate-receipt dict (output/base20/*.json
    convention, matches output/tinker/PROMOTED_*.json)

Usage:
    python -m scripts.check_token_alignment
    python scripts/check_token_alignment.py
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Same direct-execution sys.path fix as scripts/download_model.py /
# scripts/smoke_load_base20.py — scripts/ (not repo root) lands on sys.path[0]
# under direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3.6-35B-A3B"
CONFIG_JSON_PATH = MODEL_DIR / "config.json"
CONFIG_JSON_BACKUP_PATH = MODEL_DIR / "config.json.orig"
OUTPUT_PATH = PROJECT_ROOT / "output" / "base20" / "token_alignment.json"

SMOKE_PROMPT = "Reply with exactly one word: OK"
MAX_TOKENS_BUDGET = 64


# ---------------------------------------------------------------------------
# Pure, unit-testable pieces (Wave 0 covers these — see
# tests/test_check_token_alignment.py)
# ---------------------------------------------------------------------------


def align_and_check(model_config, tokenizer) -> None:
    """Align model_config eos/pad token IDs to the tokenizer's, then assert.

    Mutates model_config in place (works on any object exposing settable
    `eos_token_id`/`pad_token_id` attributes — a real PretrainedConfig/
    text_config, or a mock in tests). Per Pitfall 1: tokenizer is
    authoritative, not the pre-existing model_config value.

    Fails LOUD (AssertionError) if the post-fix invariant does not hold —
    this is the gate, not a best-effort fix.
    """
    tokenizer_eos_id = tokenizer.eos_token_id
    tokenizer_pad_id = tokenizer.pad_token_id

    model_config.eos_token_id = tokenizer_eos_id
    model_config.pad_token_id = (
        tokenizer_pad_id if tokenizer_pad_id is not None else tokenizer_eos_id
    )

    assert model_config.eos_token_id == tokenizer_eos_id, (
        f"post-fix eos mismatch: model_config={model_config.eos_token_id} "
        f"!= tokenizer={tokenizer_eos_id}"
    )
    assert model_config.pad_token_id is not None, "post-fix pad_token_id is still None"


def classify_stopped_naturally(output_ids, max_tokens, eos_token_ids) -> bool:
    """True iff generation stopped at a natural eos boundary before max_tokens.

    False when output length equals (or exceeds) the max_tokens budget
    (run-to-length = the generation never actually stopped) OR when the
    output is empty OR when the final token is not an eos id — a matched-
    but-never-hit ID must not produce a false pass (Pitfall 1).
    """
    if not output_ids:
        return False
    if len(output_ids) >= max_tokens:
        return False

    if isinstance(eos_token_ids, (list, tuple, set)):
        eos_set = set(eos_token_ids)
    else:
        eos_set = {eos_token_ids}

    return output_ids[-1] in eos_set


def build_receipt(
    status: str,
    orig_eos_id,
    aligned_eos_id,
    orig_pad_id,
    aligned_pad_id,
    tokenizer_eos_id,
    stopped_naturally: bool,
    **extra,
) -> dict:
    """Flat-JSON gate receipt (output/base20/*.json convention)."""
    return {
        "status": status,
        "orig_eos_id": orig_eos_id,
        "aligned_eos_id": aligned_eos_id,
        "orig_pad_id": orig_pad_id,
        "aligned_pad_id": aligned_pad_id,
        "tokenizer_eos_id": tokenizer_eos_id,
        "stopped_naturally": stopped_naturally,
        **extra,
    }


def write_receipt(receipt: dict) -> dict:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(receipt, f, indent=2)
    return receipt


# ---------------------------------------------------------------------------
# Real-model plumbing (not unit-tested — GPU/CPU-load-dependent, matches the
# project convention that GPU/model-load-touching scripts are smoke-tested,
# not pytest-tested; see 20-RESEARCH.md Wave 0 Gaps)
# ---------------------------------------------------------------------------


def resolve_text_config(model_config):
    """Return the sub-config object that actually carries eos/pad token ids.

    Qwen3.6-35B-A3B resolves to a composite Qwen3_5MoeConfig (VL wrapper):
    the TOP-LEVEL config has no eos_token_id/pad_token_id attribute at all
    (verified empirically) — those live on `model_config.text_config`.
    `get_text_config()` is the official transformers API for this and
    returns a live reference (mutations propagate), not a copy — verified
    empirically this session. Fall back to model_config itself for any
    plain (non-composite) config that lacks the method.
    """
    if hasattr(model_config, "get_text_config"):
        return model_config.get_text_config()
    return model_config


def run_gate() -> dict:
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    if not CONFIG_JSON_PATH.exists():
        return write_receipt(
            build_receipt(
                status="fail",
                orig_eos_id=None,
                aligned_eos_id=None,
                orig_pad_id=None,
                aligned_pad_id=None,
                tokenizer_eos_id=None,
                stopped_naturally=False,
                failing_field="config_json_missing",
                host_notes=f"{CONFIG_JSON_PATH} does not exist",
            )
        )

    # Backup BEFORE any mutation — reproducibility (T-20-02a).
    if not CONFIG_JSON_BACKUP_PATH.exists():
        shutil.copy2(CONFIG_JSON_PATH, CONFIG_JSON_BACKUP_PATH)
        print(f"Backed up {CONFIG_JSON_PATH} -> {CONFIG_JSON_BACKUP_PATH}")
    else:
        print(f"Backup already exists at {CONFIG_JSON_BACKUP_PATH} (not overwritten)")

    print(f"Loading tokenizer from {MODEL_DIR} (trust_remote_code=True) ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    print(f"Loading model from {MODEL_DIR} (bfloat16, device_map=cpu, trust_remote_code=True) ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    text_config = resolve_text_config(model.config)
    orig_eos_id = text_config.eos_token_id
    orig_pad_id = text_config.pad_token_id
    tokenizer_eos_id = tokenizer.eos_token_id

    print(f"Pre-fix: text_config.eos_token_id={orig_eos_id!r} pad_token_id={orig_pad_id!r} "
          f"tokenizer.eos_token_id={tokenizer_eos_id!r}")

    try:
        align_and_check(text_config, tokenizer)
    except AssertionError as exc:
        return write_receipt(
            build_receipt(
                status="fail",
                orig_eos_id=orig_eos_id,
                aligned_eos_id=getattr(text_config, "eos_token_id", None),
                orig_pad_id=orig_pad_id,
                aligned_pad_id=getattr(text_config, "pad_token_id", None),
                tokenizer_eos_id=tokenizer_eos_id,
                stopped_naturally=False,
                failing_field="align_and_check",
                host_notes=str(exc),
            )
        )

    aligned_eos_id = text_config.eos_token_id
    aligned_pad_id = text_config.pad_token_id

    # Persist the fix via direct JSON read-modify-write against the ORIGINAL
    # on-disk config.json (not model.config.save_pretrained()). Verified
    # empirically this session: AutoModelForCausalLM.from_pretrained() on
    # this VL checkpoint unwraps model.config to a plain text-only config
    # object (get_text_config() returns self on it) — calling
    # save_pretrained() on THAT object silently drops vision_config /
    # architectures / image_token_id / video_token_id / vision_*_token_id,
    # corrupting the VL wrapper structure 20-04's merge path depends on.
    # JSON surgery preserves every byte of the original except the two
    # target fields.
    print(f"Persisting aligned config to {CONFIG_JSON_PATH} (JSON surgery, preserving VL wrapper structure) ...")
    with open(CONFIG_JSON_BACKUP_PATH) as f:
        raw_config = json.load(f)
    target = raw_config["text_config"] if "text_config" in raw_config else raw_config
    target["eos_token_id"] = aligned_eos_id
    target["pad_token_id"] = aligned_pad_id
    # WR-06: write to a temp file in the same directory, then os.replace() it
    # into place -- atomic, so a mid-write kill (OOM/disk-full/SIGKILL) can't
    # leave a truncated/invalid config.json.
    tmp_config_path = CONFIG_JSON_PATH.with_suffix(".json.tmp")
    tmp_config_path.write_text(json.dumps(raw_config, indent=2))
    tmp_config_path.replace(CONFIG_JSON_PATH)

    # generation_config.json: align pad_token_id and ensure tokenizer's eos
    # is present in the (possibly multi-stop) eos list, if the file exists.
    generation_config_ids = None
    gen_config_path = MODEL_DIR / "generation_config.json"
    if hasattr(model, "generation_config") and gen_config_path.exists():
        gen_cfg = model.generation_config
        existing_eos = gen_cfg.eos_token_id
        if isinstance(existing_eos, (list, tuple)):
            eos_list = list(existing_eos)
            if tokenizer_eos_id not in eos_list:
                eos_list.insert(0, tokenizer_eos_id)
            gen_cfg.eos_token_id = eos_list
        else:
            gen_cfg.eos_token_id = tokenizer_eos_id
        gen_cfg.pad_token_id = aligned_pad_id
        gen_cfg.save_pretrained(str(MODEL_DIR))
        generation_config_ids = gen_cfg.eos_token_id
        print(f"Persisted generation_config.json: eos_token_id={generation_config_ids!r} "
              f"pad_token_id={gen_cfg.pad_token_id!r}")

    # Real stop-token smoke generation (Pitfall 1: a matched-but-never-hit ID
    # could still falsely "work" — must observe an actual natural stop).
    print(f"Running stop-token smoke generation: {SMOKE_PROMPT!r} (max_new_tokens={MAX_TOKENS_BUDGET}) ...")
    inputs = tokenizer(SMOKE_PROMPT, return_tensors="pt")
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS_BUDGET,
            do_sample=False,
            eos_token_id=aligned_eos_id,
            pad_token_id=aligned_pad_id,
        )
    output_ids = gen_out[0][input_len:].tolist()
    stop_gen_len = len(output_ids)
    stopped_naturally = classify_stopped_naturally(
        output_ids, MAX_TOKENS_BUDGET, aligned_eos_id
    )
    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"Generated {stop_gen_len} tokens (stopped_naturally={stopped_naturally}): {decoded!r}")

    status = "pass" if stopped_naturally else "fail"
    receipt = build_receipt(
        status=status,
        orig_eos_id=orig_eos_id,
        aligned_eos_id=aligned_eos_id,
        orig_pad_id=orig_pad_id,
        aligned_pad_id=aligned_pad_id,
        tokenizer_eos_id=tokenizer_eos_id,
        stopped_naturally=stopped_naturally,
        generation_config_eos_ids=generation_config_ids,
        stop_gen_len=stop_gen_len,
        max_tokens_budget=MAX_TOKENS_BUDGET,
        decoded_output=decoded,
        canonical_ids={
            "eos_token_id": aligned_eos_id,
            "pad_token_id": aligned_pad_id,
        },
    )
    return write_receipt(receipt)


if __name__ == "__main__":
    try:
        result = run_gate()
    except Exception as exc:  # noqa: BLE001 — gate script: any exception is a gate failure
        write_receipt(
            build_receipt(
                status="fail",
                orig_eos_id=None,
                aligned_eos_id=None,
                orig_pad_id=None,
                aligned_pad_id=None,
                tokenizer_eos_id=None,
                stopped_naturally=False,
                failing_field="exception",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        )
        print(f"BASE-02 GATE FAILED: {exc}")
        sys.exit(1)

    if result["status"] != "pass":
        print(f"BASE-02 GATE FAILED: {result}")
        sys.exit(1)

    print("BASE-02 GATE PASSED")
    print(json.dumps(result, indent=2))
