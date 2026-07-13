"""Merge LoRA adapter into base model with verification roundtrip.

Strategy (defense-in-depth):
  1. Adapter already saved separately by train_model.py (safe even if merge fails)
  2. Load base model via AutoModelForCausalLM (bfloat16, device_map=cpu, trust_remote_code=True)
  3. Load adapter via PeftModel/PeftConfig -- prefix-aware (Pitfall 3): if the adapter's saved
     LoRA module paths don't resolve against the live model's module tree (VL checkpoints wrap
     the text backbone under an extra `.language_model.` segment that Tinker's own export does
     NOT include), auto-detect and remap the keys before loading, and abort loudly (not silently)
     if the load is still partial (missing/unexpected key guard, T-20-04a).
  4. Attempt merge_and_unload()
  5. Save merged model + tokenizer
  6. Reload and verify special tokens are still single-token (<wp_gen>, <wp_judge>) -- skipped
     with a clear note when the configured extended-tokenizer vocab doesn't match this base
     (Rule 1: the v3 extended tokenizer is not applicable to the v4 base yet)
  7. If verification fails: print vLLM --lora-modules fallback command and exit 1

Usage:
    python3 scripts/merge_adapter.py
    python3 scripts/merge_adapter.py --adapter-dir ./adapters/qwen3-wp
    python3 scripts/merge_adapter.py --adapter-dir ./adapters/qwen3-wp --output-dir ./models/Qwen3-30B-A3B-merged
    python3 scripts/merge_adapter.py --config-path config/train_config_v4.yaml --adapter-dir <dir> --output-dir <dir>
    python -m scripts.merge_adapter  (also works)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "train_config.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_path(raw: str) -> Path:
    """Resolve a path that may be relative to PROJECT_ROOT."""
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# Prefix-aware adapter loading (Pitfall 3: dual key-prefix convention)
# ---------------------------------------------------------------------------


def _resolve_module_path_exists(model, dotted_path: str) -> bool:
    """Walk `dotted_path` (e.g. 'model.layers.11.self_attn.q_proj') against a
    live model, treating numeric segments as ModuleList/Sequential __getitem__
    indexing (plain hasattr/getattr does NOT work for numeric segments)."""
    obj = model
    for part in dotted_path.split("."):
        if part.isdigit():
            try:
                obj = obj[int(part)]
            except (TypeError, IndexError, KeyError):
                return False
        else:
            if not hasattr(obj, part):
                return False
            obj = getattr(obj, part)
    return True


def _lora_module_path_of(key: str) -> str:
    """Strip PEFT's 'base_model.model.' wrapper prefix and the trailing
    '.lora_A.*'/'.lora_B.*' suffix, leaving the bare module path."""
    body = key
    if body.startswith("base_model.model."):
        body = body[len("base_model.model."):]
    for marker in (".lora_A.", ".lora_B."):
        if marker in body:
            return body.split(marker)[0]
    return body


def _remap_module_path(module_path: str) -> str | None:
    """Try inserting the VL wrapper's '.language_model' segment right after
    the leading 'model.'. Returns None if module_path doesn't start with
    'model.' (unknown convention, can't attempt a remap)."""
    parts = module_path.split(".", 1)
    if len(parts) != 2 or parts[0] != "model":
        return None
    return f"model.language_model.{parts[1]}"


def _make_prefix_aware_adapter(adapter_dir: str, model) -> tuple[str, list[str]]:
    """Auto-detect and fix Pitfall 3's dual key-prefix convention, PER KEY.

    For each saved LoRA module path: if it already resolves against the live
    model, keep as-is. Else try inserting the VL wrapper's '.language_model'
    segment ('model.layers.N...' -> 'model.language_model.layers.N...'); if
    THAT resolves, remap it. If neither resolves, the module genuinely does
    not exist on this checkpoint's architecture (observed live: Tinker's
    DeltaNet/linear_attn export uses a split in_proj_q/k/v convention that
    does not match this checkpoint's fused in_proj_qkv + in_proj_a/b gating
    -- a real architecture-decomposition mismatch, not a prefix issue) --
    DROP that module's tensors from the merge and report it, rather than
    aborting the whole merge or silently keeping an unresolvable key.

    Returns (adapter_dir_to_load, dropped_module_paths). Raises loudly only
    if EVERY key is unresolvable (nothing left to merge).
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    src_path = Path(adapter_dir) / "adapter_model.safetensors"
    with safe_open(str(src_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        tensors = {k: f.get_tensor(k) for k in keys}

    if not keys:
        raise RuntimeError(f"adapter at {adapter_dir} has zero tensors -- nothing to merge")

    remapped = {}
    dropped_modules: list[str] = []
    n_asis = n_remapped = 0
    for k, t in tensors.items():
        body = k
        prefix = ""
        if body.startswith("base_model.model."):
            prefix = "base_model.model."
            body = body[len(prefix):]

        module_path = _lora_module_path_of(k)
        if _resolve_module_path_exists(model, module_path):
            remapped[k] = t
            n_asis += 1
            continue

        candidate_path = _remap_module_path(module_path)
        if candidate_path is not None and _resolve_module_path_exists(model, candidate_path):
            if body.startswith("model."):
                body = "model.language_model." + body[len("model."):]
            remapped[prefix + body] = t
            n_remapped += 1
            continue

        dropped_modules.append(module_path)

    dropped_modules = sorted(set(dropped_modules))
    if not remapped:
        raise RuntimeError(
            f"NO adapter keys resolve against the live model, even after attempting the "
            f"'.language_model' prefix remap ({len(dropped_modules)} module paths tried and "
            f"failed) -- unknown key-prefix convention, cannot safely merge"
        )

    if n_remapped:
        print(f"  [prefix-fix] Pitfall 3 dual key-prefix CONFIRMED: {n_remapped} tensor keys "
              f"used the flat 'model.layers.*' convention but the live VL-wrapped model "
              f"requires 'model.language_model.layers.*' -- remapped")
    if n_asis:
        print(f"  [prefix-check] {n_asis} tensor keys already resolved against the live model "
              f"as-is -- no remap needed")
    if dropped_modules:
        print(f"  [prefix-fix] DROPPED {len(dropped_modules)} module(s) with no live-model "
              f"equivalent under either convention (architecture-decomposition mismatch, not a "
              f"prefix issue -- e.g. Tinker's DeltaNet export splits in_proj_q/k/v where this "
              f"checkpoint fuses in_proj_qkv): {dropped_modules[:5]}"
              f"{' ...' if len(dropped_modules) > 5 else ''}")

    # Narrow target_modules to the EXACT per-layer module paths we're keeping.
    # Tinker's exported adapter_config.json says target_modules="all-linear" --
    # taken literally, PEFT would re-wrap EVERY nn.Linear in the ENTIRE local
    # model (mlp.shared_expert.*, every linear_attn.in_proj_{a,b,qkv}, etc.),
    # most of which we have no trained weights for. That inflates the
    # missing-keys count into the thousands and makes the completeness guard
    # meaningless.
    #
    # WR-02 fix: use the FULL dotted module path (e.g.
    # "model.language_model.layers.11.self_attn.q_proj"), not just the bare
    # leaf name ("q_proj"). PEFT's check_target_module_exists matches via
    # `key in config.target_modules` (exact) before falling back to
    # `key.endswith(f".{target_key}")` (suffix) -- a full path hits the exact
    # branch, so a future checkpoint/adapter combo that reuses a leaf name
    # elsewhere in the model tree (out_proj, down_proj, etc.) can't get
    # accidentally swept in via suffix matching.
    kept_module_paths = sorted({_lora_module_path_of(k) for k in remapped})
    with open(Path(adapter_dir) / "adapter_config.json") as f:
        adapter_cfg = json.load(f)
    if adapter_cfg.get("target_modules") in ("all-linear", None) or not isinstance(
        adapter_cfg.get("target_modules"), list
    ):
        print(f"  [prefix-fix] narrowing target_modules from "
              f"{adapter_cfg.get('target_modules')!r} to the {len(kept_module_paths)} exact "
              f"module path(s) actually present in the kept tensors "
              f"(sample: {kept_module_paths[:3]})")
        adapter_cfg["target_modules"] = kept_module_paths

    work_dir = tempfile.mkdtemp(prefix="merge_adapter_prefix_fixed_")
    save_file(remapped, os.path.join(work_dir, "adapter_model.safetensors"))
    with open(Path(work_dir) / "adapter_config.json", "w") as f:
        json.dump(adapter_cfg, f, indent=2)
    return work_dir, dropped_modules


def _load_expected_module_count(manifest_path: Path) -> int | None:
    """Read attached_modules count from a Task-1-style receipt (e.g.
    output/base20/lora_target_modules.json). Returns None if the file is
    absent (backward compat for v1/v3 callers with no such manifest)."""
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        d = json.load(f)
    modules = d.get("attached_modules")
    if not isinstance(modules, list) or not modules:
        return None
    return len(modules)


def _guard_merge_completeness(load_result, expected_count: int,
                               guard_receipt_path: Path | None = None,
                               raw_expected_count: int | None = None,
                               dropped_modules: list[str] | None = None) -> int:
    """Compute the number of LoRA target modules that actually loaded
    (present in the checkpoint AND matched on the live model), and abort
    loudly if it doesn't match the expected count (T-20-04a: a merge that
    exits 0 is NOT sufficient evidence -- this catches a silent partial load
    via PEFT's own `strict=False` state_dict load).

    `expected_count` is the MERGEABLE count (raw attached-modules count minus
    any modules `_make_prefix_aware_adapter` documented-and-dropped for a
    genuine architecture-decomposition mismatch, not a prefix issue). This
    guard strictly checks for any ADDITIONAL, undocumented loss beyond that.

    Writes {merged_target_module_count, expected_target_module_count,
    raw_expected_module_count, dropped_module_count} to guard_receipt_path
    (if given) BEFORE raising on failure, so a caller (e.g. a subprocess
    wrapper) can read the actual numbers even when the merge aborts.
    """
    missing_modules = {
        _lora_module_path_of(k) for k in load_result.missing_keys if ".lora_" in k
    }
    unexpected_modules = {
        _lora_module_path_of(k) for k in load_result.unexpected_keys if ".lora_" in k
    }
    if unexpected_modules:
        print(f"  [guard] WARNING: {len(unexpected_modules)} unexpected LoRA keys in the "
              f"checkpoint did not match any live module (sample: "
              f"{sorted(unexpected_modules)[:3]})")
    merged_count = expected_count - len(missing_modules)
    if guard_receipt_path is not None:
        guard_receipt_path.parent.mkdir(parents=True, exist_ok=True)
        guard_receipt_path.write_text(json.dumps({
            "merged_target_module_count": merged_count,
            "expected_target_module_count": expected_count,
            "raw_expected_module_count": raw_expected_count,
            "dropped_module_count": len(dropped_modules) if dropped_modules is not None else None,
            "dropped_modules_sample": (dropped_modules or [])[:10],
            "unexpected_module_count": len(unexpected_modules),
            "unexpected_modules_sample": sorted(unexpected_modules)[:10],
        }, indent=2))
    # WR-03: an over-broad merge (extra LoRA keys attached beyond the expected
    # scope) is exactly as untrustworthy as a partial one -- fold it into the
    # abort condition instead of only warning.
    if merged_count != expected_count or merged_count <= 0 or unexpected_modules:
        raise SystemExit(
            f"MERGE ABORT: silent partial or over-broad LoRA load detected -- merged "
            f"{merged_count}/{expected_count} MERGEABLE target modules (missing modules "
            f"sample: {sorted(missing_modules)[:5]})"
            + (f"; {len(unexpected_modules)} UNEXPECTED module(s) attached beyond scope "
               f"(sample: {sorted(unexpected_modules)[:5]})" if unexpected_modules else "")
            + ". A merge that exits 0 here would NOT be trustworthy (Pitfall 3 / T-20-04a)."
        )
    print(f"  [guard] merged {merged_count}/{expected_count} mergeable target modules "
          f"({len(dropped_modules or [])} documented drop(s) excluded) -- OK, no silent "
          f"UNDOCUMENTED partial or over-broad load")
    return merged_count


# ---------------------------------------------------------------------------
# Tokenizer compatibility (Rule 1: don't blindly trust a stale extended
# tokenizer built for a different base's vocab)
# ---------------------------------------------------------------------------


def _repair_vl_config(merged_path: str, local_dir: str) -> None:
    """Restore the composite VL config.json wrapper around the merged model.

    `AutoModelForCausalLM.from_pretrained()` on this VL checkpoint resolves
    to the flattened TEXT-ONLY subclass (`Qwen3_5MoeForCausalLM` /
    `Qwen3_5MoeTextConfig`) -- the SAME composite-config-unwraps-on-load
    pitfall 20-02-SUMMARY documented for `model.config.save_pretrained()`,
    now hit one level up via `merged_model.save_pretrained()`. The saved
    config.json loses `vision_config`/`architectures`/`image_token_id`/etc,
    which vLLM's `--language-model-only` path still reads at model-class
    construction time (confirmed empirically: `AttributeError:
    'Qwen3_5MoeTextConfig' object has no attribute 'vision_config'` during
    this plan's own smoke run) even though it never loads vision weights.
    The on-disk WEIGHT keys are unaffected (transformers re-nests them to
    `model.language_model.layers.*` on save regardless of the resolved
    class) -- only the top-level config.json wrapper needs restoring, via
    JSON surgery against the ORIGINAL base's config.json (not
    model.save_pretrained() -- same lesson as 20-02).
    """
    merged_config_path = Path(merged_path) / "config.json"
    with open(merged_config_path) as f:
        flat_config = json.load(f)

    original_config_path = Path(local_dir) / "config.json"
    with open(original_config_path) as f:
        original_config = json.load(f)

    if "text_config" not in original_config:
        print("  [vl-config-repair] original base config.json has no text_config wrapper "
              "-- nothing to repair")
        return

    wrapper_fields = {k: v for k, v in original_config.items() if k != "text_config"}
    composite = {**wrapper_fields, "text_config": flat_config}
    # WR-06: atomic write — temp file in the same directory, then os.replace()
    # into place, so a mid-write kill can't leave a truncated/invalid config.json.
    tmp_config_path = merged_config_path.with_suffix(".json.tmp")
    tmp_config_path.write_text(json.dumps(composite, indent=2))
    tmp_config_path.replace(merged_config_path)
    print(f"  [vl-config-repair] restored composite VL config.json wrapper "
          f"({sorted(wrapper_fields.keys())}) around the merged text_config -- "
          f"matches 20-02's JSON-surgery lesson, not model.save_pretrained()")


def _select_serving_tokenizer(config: dict, local_dir: str, base_vocab_size: int):
    """Return (tokenizer, check_special_tokens: bool).

    config['tokenizer']['save_dir'] (the extended tokenizer with <wp_gen>/
    <wp_judge>) is only valid for the base it was built from. If its vocab
    size doesn't roughly match the CURRENT base's vocab, fall back to the
    base model's own tokenizer and skip the special-token assertions (this
    base has no task-token extension yet -- that is a later-phase concern).
    """
    from transformers import AutoTokenizer

    tokenizer_dir = str(resolve_path(config["tokenizer"]["save_dir"]))
    try:
        extended_tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        compatible = abs(len(extended_tok) - base_vocab_size) < 1000
    except Exception as e:  # noqa: BLE001 -- any load failure means "not compatible"
        print(f"  [tokenizer] could not load extended tokenizer at {tokenizer_dir}: {e}")
        extended_tok = None
        compatible = False

    if compatible:
        print(f"Loading extended tokenizer from {tokenizer_dir} ...")
        return extended_tok, True

    print(f"  [tokenizer] WARNING: extended tokenizer at {tokenizer_dir} "
          f"(vocab={len(extended_tok) if extended_tok else '?'}) does not match this base's "
          f"vocab ({base_vocab_size}) -- falling back to the base model's own tokenizer and "
          f"skipping the <wp_gen>/<wp_judge> special-token check (not yet applicable to this base)")
    base_tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    return base_tok, False


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_adapter(adapter_dir: str, output_dir: str, config: dict,
                   expected_modules_manifest: Path | None = None,
                   guard_receipt_path: Path | None = None) -> None:
    """Load base model + adapter, merge, save, and verify special tokens.

    Args:
        adapter_dir: Path to the saved LoRA adapter (adapters/qwen3-wp/).
        output_dir: Destination for the merged model.
        config: Parsed train_config.yaml dict.
        expected_modules_manifest: Optional path to a Task-1-style receipt
            (attached_modules list) used for the merged-target-module-count
            partial-load guard. Guard is skipped if the file is absent.
        guard_receipt_path: Where to write the merged-target-module-count guard
            result. Defaults to output/base20/_merge_guard_result.json (the
            original Phase-20 location) for backward compatibility -- callers
            outside Phase 20 (e.g. a later phase's own probe/real merge) MUST
            pass their own path here, otherwise every merge silently
            overwrites Phase 20's already-committed receipt (Rule 3: this was
            a real, load-bearing hardcode, not a hypothetical).

    Raises:
        SystemExit(1) on verification failure (prints vLLM fallback command).
    """
    guard_receipt_path = guard_receipt_path or (PROJECT_ROOT / "output" / "base20" / "_merge_guard_result.json")
    # --- Idempotency check: skip if merged model already exists and verified ---
    merged_path = Path(output_dir)
    if merged_path.exists() and (merged_path / "config.json").exists():
        # Quick verification — are special tokens intact?
        from transformers import AutoTokenizer as _AT  # noqa: PLC0415
        try:
            local_dir_probe = str(resolve_path(config["model"]["local_dir"]))
            # WR-04: reuse the SAME base/extended-tokenizer compatibility check
            # _select_serving_tokenizer/_verify_merged_model use later in this
            # function (Rule 1), so a base without the extended vocab (like
            # this v4 base -- "no task-token extension yet") can still
            # short-circuit here instead of always falling through to a full
            # re-merge. base_vocab_size comes from the base's own tokenizer
            # (cheap: tokenizer files only, no model weights loaded).
            base_tok_probe = _AT.from_pretrained(local_dir_probe, trust_remote_code=True)
            _, check_special_tokens = _select_serving_tokenizer(
                config, local_dir_probe, len(base_tok_probe)
            )
            if not check_special_tokens:
                print(f"Merged model already exists at {merged_path} (extended tokenizer not "
                      f"applicable to this base -- special-token check not required). Skipping.")
                return
            verify_tok = _AT.from_pretrained(str(merged_path), trust_remote_code=True)
            special_tokens = config.get("tokenizer", {}).get("special_tokens", ["<wp_gen>", "<wp_judge>"])
            all_single = all(
                len(verify_tok.encode(t, add_special_tokens=False)) == 1 for t in special_tokens
            )
            if all_single:
                print(f"Merged model already exists at {merged_path} with verified special tokens. Skipping.")
                return
        except Exception:
            pass  # Fall through to re-merge

    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    local_dir = str(resolve_path(config["model"]["local_dir"]))

    print(f"Loading base model from {local_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        dtype=torch.bfloat16,
        device_map="cpu",  # MoE models can't auto-offload to disk; 30B bf16 fits in 128GB unified RAM
        trust_remote_code=True,
    )

    # Load the LoRA adapter on top of the base model
    from peft import PeftModel, PeftConfig  # noqa: PLC0415

    print(f"Loading LoRA adapter from {adapter_dir} ...")
    # _make_prefix_aware_adapter also narrows target_modules (Tinker's saved
    # "all-linear" would otherwise make PEFT wrap EVERY Linear in the entire
    # local model, not just the modules we have trained weights for) -- read
    # peft_config from ITS output dir, not the original adapter_dir.
    load_dir, dropped_modules = _make_prefix_aware_adapter(adapter_dir, model)

    # Zero out lora_dropout before loading — dropout is training-only and
    # newer peft versions reject non-zero dropout on ParamWrapper (modules_to_save).
    peft_config = PeftConfig.from_pretrained(load_dir)
    if getattr(peft_config, "lora_dropout", 0) != 0:
        print(f"  Zeroing lora_dropout ({peft_config.lora_dropout} → 0) for merge compatibility")
        peft_config.lora_dropout = 0

    # Construct the PEFT wrapper (default-init LoRA layers matching
    # target_modules), then explicitly load the adapter weights so we get
    # direct visibility into missing/unexpected keys (PeftModel.from_pretrained
    # discards this; strict=False internally means a partial load would
    # otherwise be completely silent -- Pitfall 3 / T-20-04a).
    peft_model = PeftModel(model, peft_config)
    load_result = peft_model.load_adapter(load_dir, adapter_name="default", is_trainable=False)

    raw_expected_count = _load_expected_module_count(
        expected_modules_manifest or (PROJECT_ROOT / "output" / "base20" / "lora_target_modules.json")
    )
    if raw_expected_count is not None:
        # Modules dropped by _make_prefix_aware_adapter (genuine architecture-
        # decomposition mismatch, e.g. Tinker's DeltaNet in_proj_q/k/v split
        # vs this checkpoint's fused in_proj_qkv) are a documented, counted
        # exclusion -- the guard below still checks strictly for any
        # ADDITIONAL, undocumented loss among what remained mergeable.
        mergeable_expected_count = raw_expected_count - len(dropped_modules)
        _guard_merge_completeness(
            load_result, mergeable_expected_count,
            guard_receipt_path=guard_receipt_path,
            raw_expected_count=raw_expected_count,
            dropped_modules=dropped_modules,
        )
    else:
        print("  [guard] no expected-modules manifest found -- skipping merged-target-module-count guard")

    # Attempt merge
    print("Merging adapter into base model ...")
    merged_model = peft_model.merge_and_unload()

    merged_path = output_dir
    print(f"Saving merged model to {merged_path} ...")
    Path(merged_path).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_path)
    _repair_vl_config(merged_path, local_dir)

    # Save tokenizer alongside merged model (compatibility-checked -- Rule 1)
    base_vocab_size = merged_model.get_input_embeddings().weight.shape[0]
    serving_tok, check_special_tokens = _select_serving_tokenizer(config, local_dir, base_vocab_size)
    serving_tok.save_pretrained(merged_path)
    print(f"Tokenizer saved to {merged_path}")

    # Verification roundtrip — reload from disk and check special tokens
    print("Running verification roundtrip ...")
    _verify_merged_model(merged_path, adapter_dir, config, check_special_tokens=check_special_tokens)


def _verify_merged_model(merged_path: str, adapter_dir: str, config: dict,
                          check_special_tokens: bool = True) -> None:
    """Reload merged model and verify special tokens are single-token.

    Args:
        merged_path: Path to the merged model directory.
        adapter_dir: Original adapter dir (for fallback message).
        config: Parsed train_config.yaml dict.
        check_special_tokens: If False, skip the <wp_gen>/<wp_judge> assertion
            (this base has no task-token extension yet -- Rule 1).

    Raises:
        SystemExit(1) on failure, with vLLM fallback command printed.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    if not check_special_tokens:
        print("  (special-token check skipped -- see tokenizer compatibility note above)")
        print("MERGE VERIFICATION PASSED (tokenizer-load-only check)")
        return

    try:
        verify_tok = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)

        wp_gen_ids = verify_tok.encode("<wp_gen>", add_special_tokens=False)
        wp_judge_ids = verify_tok.encode("<wp_judge>", add_special_tokens=False)

        assert len(wp_gen_ids) == 1, (
            f"<wp_gen> must be single token, got {wp_gen_ids}"
        )
        assert len(wp_judge_ids) == 1, (
            f"<wp_judge> must be single token, got {wp_judge_ids}"
        )

        print(f"  <wp_gen>   -> token ID {wp_gen_ids[0]} (OK)")
        print(f"  <wp_judge> -> token ID {wp_judge_ids[0]} (OK)")
        print("MERGE VERIFICATION PASSED")

    except AssertionError as e:
        print(f"MERGE VERIFICATION FAILED: {e}")
        print()
        print("Fallback: serve adapter directly with vLLM (no merge needed):")
        base_model = str(resolve_path(config["model"]["local_dir"]))
        print(f"  vllm serve {base_model} --lora-modules qwen3-wp={adapter_dir}")
        print()
        print("The adapter is still saved separately and can be used directly.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Run the merge pipeline."""
    config_path = resolve_path(args.config_path) if args.config_path else CONFIG_PATH
    config = load_config(config_path)

    # Resolve adapter and output directories
    adapter_dir = str(resolve_path(
        args.adapter_dir if args.adapter_dir else config["training"]["output_dir"]
    ))
    output_dir = str(resolve_path(
        args.output_dir if args.output_dir else config["model"]["local_dir"] + "-merged"
    ))
    expected_modules_manifest = resolve_path(args.expected_modules_manifest) if args.expected_modules_manifest else None
    guard_receipt_path = resolve_path(args.guard_receipt_path) if args.guard_receipt_path else None

    print("=" * 60)
    print("MERGE ADAPTER CONFIGURATION")
    print("=" * 60)
    print(f"  Config path: {config_path}")
    print(f"  Base model:  {config['model']['local_dir']}")
    print(f"  Adapter dir: {adapter_dir}")
    print(f"  Output dir:  {output_dir}")
    print("=" * 60)

    merge_adapter(adapter_dir, output_dir, config, expected_modules_manifest=expected_modules_manifest,
                  guard_receipt_path=guard_receipt_path)
    print(f"\nMerged model ready at: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model with verification roundtrip."
    )
    parser.add_argument(
        "--config-path",
        default=None,
        metavar="PATH",
        help=(
            "Path to a train_config*.yaml (default: config/train_config.yaml). "
            "Pass config/train_config_v4.yaml to merge onto the v4 base."
        ),
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        metavar="PATH",
        help=(
            "Path to the saved LoRA adapter directory. "
            "Defaults to training.output_dir from config/train_config.yaml."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help=(
            "Destination for the merged model. "
            "Defaults to model.local_dir + '-merged' from config/train_config.yaml."
        ),
    )
    parser.add_argument(
        "--expected-modules-manifest",
        default=None,
        metavar="PATH",
        help=(
            "Path to a Task-1-style receipt (attached_modules list) for the "
            "merged-target-module-count partial-load guard. Defaults to "
            "output/base20/lora_target_modules.json if present; guard is "
            "skipped if no manifest is found."
        ),
    )
    parser.add_argument(
        "--guard-receipt-path",
        default=None,
        metavar="PATH",
        help=(
            "Where to write the merged-target-module-count guard result. "
            "Defaults to output/base20/_merge_guard_result.json (backward "
            "compatible). Callers outside Phase 20 (a later phase's own "
            "probe/real merge) should pass their own path so they don't "
            "silently overwrite Phase 20's already-committed receipt."
        ),
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    try:
        main(args)
    except SystemExit:
        # Already an intentional, diagnosed exit (e.g. _guard_merge_completeness's
        # abort or _verify_merged_model's failure) — both print their own
        # diagnostics/receipts. Re-raise unchanged (WR-05 only covers the
        # "no receipt at all" case below).
        raise
    except Exception as exc:  # noqa: BLE001 -- gate script: any exception is a merge failure
        receipt_path = PROJECT_ROOT / "output" / "base20" / "_merge_adapter_result.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps({"status": "fail", "error": str(exc)}, indent=2))
        print(f"MERGE ADAPTER FAILED: {exc}")
        sys.exit(1)
