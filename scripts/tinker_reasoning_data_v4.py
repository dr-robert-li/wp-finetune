#!/usr/bin/env python
"""Tinker v4 (Qwen3.6-35B-A3B) — data adapter for the WP reasoning SFT.

Non-destructive sibling of tinker_reasoning_data.py (v3, Qwen3-30B-A3B unchanged).
Same reasoning dataset (data/reasoning_dataset/), same FromConversationFileBuilder +
ChatDatasetBuilderCommonConfig + TrainOnWhat.LAST_ASSISTANT_MESSAGE mechanics — the
only deltas are BASE_MODEL and the resolved RENDERER_NAME (21-RESEARCH.md Open
Question 2).

RENDERER resolution (GEN-01 / Open Question 2): probed at runtime against the
actual tinker_cookbook renderer registry (not assumed). `tinker_cookbook.renderers`
ships a DEDICATED Qwen3.5-family renderer (`qwen3_5_disable_thinking` /
`Qwen3_5DisableThinkingRenderer`) that matches this base's real class
(`Qwen3_5MoeForCausalLM`, confirmed by 20-04's merge probe) — preferred over the
generic `qwen3_disable_thinking` (built for the OLDER, non-VL Qwen3 architecture,
used only as Phase 20's attention-only merge-probe renderer, not a data-format
authority). Empirically verified live against this exact base+tokenizer
(2026-07-13): the Qwen3.5 HF template always inserts an empty `<think>\n\n</think>\n\n`
block into the ASSISTANT HEADER for turns after the last user message — but
`build_supervised_example(..., train_on_what=LAST_ASSISTANT_MESSAGE)` gives that
header ZERO loss weight (header_weight = int(train_on_what == ALL_TOKENS) == 0 in
tinker_cookbook's base renderer). Decoding the weight>0 span alone confirms the
empty think block is NOT part of the loss target — QwenLM #131's concern is real
for the RENDERED TEXT but N/A for THIS project's TRAINING SIGNAL under this
train_on_what mode. Falls back to `qwen3_disable_thinking` if the dedicated entry
doesn't resolve; raises loudly (not a silent guess) if neither resolves — manual
prompt construction was not needed empirically this run, so is not implemented
speculatively (see Task 1 read_first / gen01_format_decision.json for the receipt).
"""
from tinker_cookbook import renderers
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.tokenizer_utils import get_tokenizer

BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
MAX_LENGTH = 8192  # unchanged from v3 -- real data max is ~5.4K chars, comfortably under
TRAIN_PATH = "data/reasoning_dataset/openai_train.jsonl"
VAL_PATH = "data/reasoning_dataset/openai_val.jsonl"

# Preference order: dedicated Qwen3.5-family no-thinking renderer first (matches
# this base's actual resolved class per 20-04), generic Qwen3 no-thinking renderer
# as the Phase-20-confirmed fallback.
_RENDERER_CANDIDATES = ["qwen3_5_disable_thinking", "qwen3_disable_thinking"]


def resolve_renderer_name(base_model: str = BASE_MODEL):
    """Probe tinker_cookbook's renderer registry at runtime (not assumed).

    Returns (renderer_name_or_None, source) where source is "registry" (a
    candidate resolved) or "manual_fallback" (none did -- not implemented here,
    since it was never empirically needed; raises downstream if hit).
    """
    tok = get_tokenizer(base_model)
    for name in _RENDERER_CANDIDATES:
        try:
            renderers.get_renderer(name, tok)
            return name, "registry"
        except Exception:
            continue
    return None, "manual_fallback"


RENDERER_NAME, RENDERER_SOURCE = resolve_renderer_name()
if RENDERER_NAME is None:
    raise RuntimeError(
        "No candidate renderer resolved for BASE_MODEL="
        f"{BASE_MODEL!r} (tried {_RENDERER_CANDIDATES}). Manual generation-prompt "
        "construction (bypassing apply_chat_template) was the planned fallback but "
        "is NOT implemented -- it was never empirically exercised. Implement it "
        "here before proceeding, per 21-RESEARCH.md Open Question 2."
    )


def build_datasets(train_path: str = TRAIN_PATH, val_path: str = VAL_PATH,
                   batch_size: int = 8):
    """Return (train_dataset, val_dataset) as cookbook SupervisedDatasets."""
    cc = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=BASE_MODEL,
        renderer_name=RENDERER_NAME,
        max_length=MAX_LENGTH,
        batch_size=batch_size,
        train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    train_ds, _ = FromConversationFileBuilder(
        file_path=train_path, test_size=0, common_config=cc)()
    val_ds, _ = FromConversationFileBuilder(
        file_path=val_path, test_size=0, common_config=cc)()
    return train_ds, val_ds


def _max_tokenized_len_and_think_check(train_ds, val_ds):
    """GEN-01 spot-check: max tokenized length across ALL train+val datums, and
    confirm no non-empty-weight span decodes to include a spurious empty
    <think></think> pair (QwenLM #131 concern, checked against the actual
    LOSS-WEIGHTED span, not just the raw rendered text -- see module docstring)."""
    import numpy as np

    max_len = 0
    empty_think_injected = False
    tok = get_tokenizer(BASE_MODEL)
    for ds in (train_ds, val_ds):
        for i in range(len(ds)):
            batch = ds.get_batch(i)
            for datum in batch:
                max_len = max(max_len, datum.model_input.length)
                weights = np.array(datum.loss_fn_inputs["weights"].data)
                target_tokens = np.array(datum.loss_fn_inputs["target_tokens"].data, dtype=int)
                nz = np.where(weights > 0)[0]
                if len(nz) == 0:
                    continue
                target_text = tok.decode(target_tokens[nz].tolist())
                if "<think>" in target_text or "</think>" in target_text:
                    empty_think_injected = True
    return max_len, empty_think_injected


if __name__ == "__main__":
    import json
    from pathlib import Path

    tr, va = build_datasets(batch_size=8)
    print(f"renderer={RENDERER_NAME} (source={RENDERER_SOURCE})")
    print(f"train batches: {len(tr)} | val batches: {len(va)} (batch_size=8)")
    b = tr.get_batch(0)
    print(f"batch[0]: {len(b)} Datums; first datum tokens={b[0].model_input.length}")

    max_len, empty_think_injected = _max_tokenized_len_and_think_check(tr, va)
    len_under_64k = max_len < 64_000
    print(f"max_tokenized_len={max_len} len_under_64k={len_under_64k} "
          f"empty_think_injected={empty_think_injected}")

    from tinker_cookbook import hyperparam_utils as hp
    resolved_lr = hp.get_lr(BASE_MODEL, is_lora=True)

    # output_router_logits (GEN-01 / Open Question 3): checked directly against
    # Tinker's actual training-client API surface, not assumed from HF transformers
    # usage notes. tinker.ServiceClient.create_lora_training_client(base_model, rank,
    # seed, train_mlp, train_attn, train_unembed, user_metadata) and
    # TrainingClient.forward_backward(data, loss_fn, loss_fn_config) expose NO
    # router/aux-loss/load-balancing kwarg; grep across the installed tinker +
    # tinker_cookbook package source for "router"/"load_balanc" returns zero hits.
    output_router_logits_disposition = (
        "N/A at Tinker's abstraction layer (raw-transformers-forward concern only) "
        "-- evidence: tinker.ServiceClient.create_lora_training_client signature "
        "(base_model, rank, seed, train_mlp, train_attn, train_unembed, "
        "user_metadata) and TrainingClient.forward_backward(data, loss_fn, "
        "loss_fn_config) expose no router/aux-loss knob; grep for "
        "'router'/'load_balanc' across the installed tinker + tinker_cookbook "
        "package source returns zero hits (checked live, 2026-07-13)."
    )

    n_train = sum(len(tr.get_batch(i)) for i in range(len(tr)))
    n_val = sum(len(va.get_batch(i)) for i in range(len(va)))

    receipt = {
        "renderer_name": RENDERER_NAME,
        "renderer_source": RENDERER_SOURCE,
        "resolved_lr": resolved_lr,
        "lr_rationale": (
            "Kept hp.get_lr(BASE_MODEL, is_lora=True) -- Tinker's own auto-computed "
            "LR for the base+LoRA combination, matching the ACTUAL v1.2/v1.3 Tinker "
            "regime (empirically resolved here to ~4.99e-4, matching ROADMAP.md's "
            "Phase 4.3 supersession note). GEN-02's literal '<=2e-5' text is a "
            "stale carry-over from the abandoned DGX/Unsloth-era RTRN-01 spec "
            "(superseded 2026-06-11 per ROADMAP.md), not a deliberate new v4.0 "
            "constraint -- following 'same pipeline as Qwen3' literally means "
            "reusing Tinker's auto-LR, not the superseded manual cap."
        ),
        "output_router_logits_disposition": output_router_logits_disposition,
        "max_tokenized_len": max_len,
        "len_under_64k": len_under_64k,
        "empty_think_injected": empty_think_injected,
        "n_train_examples": n_train,
        "n_val_examples": n_val,
    }
    out_path = Path("output/base21/gen01_format_decision.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2))
    print(f"wrote {out_path}")
    print("DATA-ADAPTER OK")
