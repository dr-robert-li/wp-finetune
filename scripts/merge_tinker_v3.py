"""Tinker-convention MoE LoRA merge for wp-reasoning-v3 -> stock Qwen3-30B-A3B.

This module implements Tinker's THIRD distinct MoE tensor convention (distinct from
PEFT strided `B[:, e::E]` and from Unsloth contiguous-block `B[:, e*R:(e+1)*R]`).

Tinker layout (VERIFIED by tensor inspection of
models/tinker_export/wp-reasoning-v3/checkpoint.tar, 2026-06-07):

  w1 (gate_proj): lora_A [1,32,2048] SHARED      ; lora_B [128,768,32] PER-EXPERT
  w2 (down_proj): lora_A [128,32,768] PER-EXPERT ; lora_B [1,2048,32]  SHARED
  w3 (up_proj):   lora_A [1,32,2048] SHARED      ; lora_B [128,768,32] PER-EXPERT
  unembed:        lora_A [32,2048]               ; lora_B [151936,32]  (STOCK vocab)

Per-expert delta math (scale = lora_alpha / r):
  delta_gate_e = B_w1[e] @ A_w1            -> [768,2048]
  delta_up_e   = B_w3[e] @ A_w3            -> [768,2048]
  gate_up[e]  += cat([delta_gate_e, delta_up_e], dim=0)  -> [1536,2048]  (gate FIRST)
  delta_down_e = B_w2 @ A_w2[e]            -> [2048,768]
  down[e]     += delta_down_e
  lm_head     += (B_unembed @ A_unembed)   -> [151936,2048]

EMPIRICALLY VERIFIED per-expert distinctness (this plan's research, 2026-06-07):
  w1.lora_B[0] vs [1] max_diff = 0.049 ; w2.lora_A[0] vs [1] max_diff = 0.049.
  => ALL of w1/w2/w3 produce PER-EXPERT-DISTINCT deltas. A test asserting
  delta_e0 == delta_e1 (the obsolete 04.4-RESEARCH "shared-A => same delta" claim)
  would PASS a broadcast bug and FAIL a correct merge -- do NOT use it.

NO model load and NO file IO happen at import: the delta builders + fidelity helpers
are pure tensor/logic functions so tests/phase4_4/test_tinker_merge_convention.py and
tests/phase4_4/test_fidelity_protocol.py import the exact production code. The CLI merge
(stock base load, per-expert apply, manual lm_head, staging save, merge_report) lives
under `if __name__ == "__main__"` and is added in Task 2.

Plan 02 import path (fidelity gate consumes these):
  from scripts.merge_tinker_v3 import sentinel_agreement, spearman_agree
"""

from __future__ import annotations

from typing import Sequence

import torch


# --------------------------------------------------------------------------- #
# Per-expert LoRA delta builders (pure functions; no IO, no model load).
# --------------------------------------------------------------------------- #

def _squeeze_shared(t: torch.Tensor) -> torch.Tensor:
    """Squeeze a leading singleton dim off a SHARED factor: [1,R,in] -> [R,in]."""
    if t.dim() == 3 and t.shape[0] == 1:
        return t.squeeze(0)
    return t


def build_gate_up_delta(
    A_w1: torch.Tensor,
    B_w1: torch.Tensor,
    A_w3: torch.Tensor,
    B_w3: torch.Tensor,
    e: int,
    scale: float = 1.0,
) -> torch.Tensor:
    """Fused gate_up delta for expert `e` -> [1536, 2048] (gate rows first, up rows second).

    A_w1/A_w3 are the SHARED gate/up lora_A ([1,32,2048] or [32,2048]).
    B_w1/B_w3 are the PER-EXPERT lora_B ([128,768,32]); expert slice is B[e] -> [768,32].
    delta_gate = B_w1[e] @ A_w1 ; delta_up = B_w3[e] @ A_w3 ; cat on dim 0, gate first.
    """
    A1 = _squeeze_shared(A_w1).float()          # [32,2048]
    A3 = _squeeze_shared(A_w3).float()          # [32,2048]
    Bg = B_w1[e].float()                        # [768,32]
    Bu = B_w3[e].float()                        # [768,32]
    delta_gate = (Bg @ A1) * scale              # [768,2048]
    delta_up = (Bu @ A3) * scale                # [768,2048]
    return torch.cat([delta_gate, delta_up], dim=0)   # [1536,2048] gate-first


def build_down_delta(
    A_w2: torch.Tensor,
    B_w2: torch.Tensor,
    e: int,
    scale: float = 1.0,
) -> torch.Tensor:
    """down_proj delta for expert `e` -> [2048, 768].

    A_w2 is the PER-EXPERT lora_A ([128,32,768]); expert slice A_w2[e] -> [32,768].
    B_w2 is the SHARED lora_B ([1,2048,32] or [2048,32]).
    delta_down = B_w2 @ A_w2[e].
    """
    B2 = _squeeze_shared(B_w2).float()          # [2048,32]
    A2_e = A_w2[e].float()                       # [32,768]
    return (B2 @ A2_e) * scale                    # [2048,768]


def build_lm_head_delta(
    A_un: torch.Tensor,
    B_un: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """unembed_tokens -> lm_head delta -> [151936, 2048], computed in float32.

    A_un [32,2048], B_un [151936,32]. delta = B_un @ A_un.
    """
    return (B_un.float() @ A_un.float()) * scale  # [151936,2048] float32


def per_expert_differ(deltas: Sequence[torch.Tensor]) -> float:
    """Min pairwise max-abs-difference across a list of per-expert deltas.

    Returns ~0 for a broadcast merge (all experts identical) and > 1e-5 for genuine
    per-expert deltas. The merge guard aborts when this is <= 1e-5. Requires >= 2
    deltas; with fewer there is no differentiation evidence so 0.0 is returned
    (which deliberately TRIPS the guard rather than vacuously passing it).
    """
    n = len(deltas)
    if n < 2:
        return 0.0
    worst = float("inf")
    for i in range(n):
        di = deltas[i].float()
        for j in range(i + 1, n):
            d = (di - deltas[j].float()).abs().max().item()
            if d < worst:
                worst = d
    return worst


# --------------------------------------------------------------------------- #
# Fidelity agreement helpers (pure logic; consumed by plan 02's fidelity gate).
# --------------------------------------------------------------------------- #

def sentinel_agreement(
    tinker_verdicts: Sequence,
    merged_verdicts: Sequence,
) -> int:
    """Count index-aligned verdict matches between Tinker-sampled and merged-served.

    Lengths MUST match (each sentinel prompt scored on both sides) -- a length
    mismatch is a wiring bug, not a partial result, so it raises instead of
    silently truncating.
    """
    if len(tinker_verdicts) != len(merged_verdicts):
        raise ValueError(
            f"verdict length mismatch: tinker={len(tinker_verdicts)} "
            f"merged={len(merged_verdicts)}"
        )
    return sum(1 for a, b in zip(tinker_verdicts, merged_verdicts) if a == b)


def _avg_rank(values: Sequence[float]) -> "list[float]":
    """Tie-aware average ranks (1-based), pure-python (no scipy dependency)."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # mean of 1-based positions i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (tie-aware), no scipy dependency."""
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    if len(x) < 2:
        return 0.0
    rx = _avg_rank(list(x))
    ry = _avg_rank(list(y))
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    denom = (vx * vy) ** 0.5
    if denom == 0.0:
        return 0.0
    return cov / denom


def spearman_agree(
    tinker_scores: Sequence[float],
    merged_scores: Sequence[float],
    thresh: float = 0.95,
) -> bool:
    """True iff Spearman(tinker, merged) >= thresh. Identical -> 1.0 -> True;
    shuffled/uncorrelated -> below thresh -> False."""
    return spearman_rho(tinker_scores, merged_scores) >= thresh


# --------------------------------------------------------------------------- #
# CLI merge (stock base load + per-expert apply + manual lm_head + staging save).
# Heavy deps (transformers/peft/safetensors) are imported lazily inside the merge
# functions so module import stays light (Task-1 < 5 s invariant). NO model load or
# file IO happens at import or at `--help`.
# --------------------------------------------------------------------------- #

NUM_LAYERS = 48
NUM_EXPERTS = 128
ATTN_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]
STOCK_VOCAB = 151936       # stock MODEL embedding/lm_head rows (padded)
STOCK_TOK_LEN = 151669     # stock Qwen3 TOKENIZER real token count (< padded model vocab).
                           # NOTE: the model embedding is padded to 151936; the tokenizer only
                           # has 151669 tokens. The extended tokenizer (151671) adds <wp_gen>/
                           # <wp_judge> as SINGLE special ids 151669/151670 -- but v3 was trained
                           # on the STOCK tokenizer (tinker_reasoning_data.py:15-16) where those
                           # markers are PLAIN TEXT. Serving the extended tokenizer would diverge
                           # from training and break task routing. So: stock tokenizer is required.
GATE_UP_OUT = 1536  # 2 * per-expert mlp dim (gate first, up second)

DEFAULT_ADAPTER_TAR = "models/tinker_export/wp-reasoning-v3/checkpoint.tar"
DEFAULT_BASE = "models/Qwen3-30B-A3B"
DEFAULT_OUTPUT_DIR = "models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v3"
DEFAULT_REPORT = "output/merge_v3/merge_report.json"
RAM_FLOOR_GIB = 70.0


def _free_ram_gib() -> float:
    """Available RAM in GiB from /proc/meminfo (MemAvailable). Used for the RAM floor."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    kb = float(line.split()[1])
                    return kb / (1024.0 * 1024.0)
    except OSError:
        pass
    return float("inf")  # unknown -> do not block (Task-3 launcher also guards)


def _untar_adapter(tar_path: str, dest: str) -> str:
    """Extract adapter_config.json + adapter_model.safetensors from checkpoint.tar."""
    import os
    import tarfile

    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            if name in ("adapter_config.json", "adapter_model.safetensors"):
                m.name = name  # flatten any leading dirs
                tf.extract(m, dest)
    return dest


def _load_adapter_tensors(adapter_dir: str) -> "dict":
    """Load all LoRA tensors from the (single) adapter_model.safetensors into CPU."""
    import os

    from safetensors import safe_open

    tensors = {}
    path = os.path.join(adapter_dir, "adapter_model.safetensors")
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def _k(layer: int, w: str, ab: str) -> str:
    """Tinker MoE expert key: base_model.model.model.layers.{L}.mlp.experts.{w}.lora_{A|B}.weight"""
    return f"base_model.model.model.layers.{layer}.mlp.experts.{w}.lora_{ab}.weight"


def _build_attention_only_adapter(adapter_tensors: dict, adapter_cfg: dict, tmp_dir: str) -> str:
    """Write a temp PEFT adapter holding ONLY self_attn q/k/v/o LoRA (2D, standard)."""
    import json
    import os

    from safetensors.torch import save_file

    os.makedirs(tmp_dir, exist_ok=True)
    attn = {k: v for k, v in adapter_tensors.items()
            if any(f".{p}." in k for p in ATTN_PROJS)}
    assert attn, "no self_attn LoRA tensors found in adapter"
    save_file(attn, os.path.join(tmp_dir, "adapter_model.safetensors"))
    cfg = dict(adapter_cfg)
    cfg["target_modules"] = ATTN_PROJS
    cfg["target_parameters"] = []
    cfg["modules_to_save"] = None
    with open(os.path.join(tmp_dir, "adapter_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    return tmp_dir


def _run_merge(args) -> int:
    import json
    import os
    import tempfile
    import time

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)

    # 1) Extract adapter + read scale = lora_alpha / r (never hardcoded at call sites).
    work = tempfile.mkdtemp(prefix="tinker_v3_adapter_")
    _untar_adapter(args.adapter_tar, work)
    with open(os.path.join(work, "adapter_config.json")) as fh:
        acfg = json.load(fh)
    r = int(acfg["r"])
    scale = float(acfg.get("lora_alpha", r)) / float(r)
    adapter = _load_adapter_tensors(work)

    # 2) Load STOCK base on CPU bf16. device_map is an explicit CPU placement, never
    #    "auto" (MoE cannot be auto-offloaded mid-merge without corrupting expert math).
    model = AutoModelForCausalLM.from_pretrained(
        args.base, device_map={"": "cpu"}, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    assert model.config.tie_word_embeddings is False, (
        "tie_word_embeddings must be False so the manual lm_head LoRA does not leak into embed_tokens"
    )
    assert model.config.vocab_size == STOCK_VOCAB, (
        f"base vocab {model.config.vocab_size} != stock {STOCK_VOCAB} (wrong base?)"
    )
    lm_head_w = model.lm_head.weight
    assert tuple(lm_head_w.shape) == (STOCK_VOCAB, model.config.hidden_size), tuple(lm_head_w.shape)

    report = {
        "status": "staging_pending_anchor",
        "merge_type": "tinker_per_expert_moe_plus_peft_attention_plus_manual_lm_head",
        "base_path": args.base,
        "adapter_tar": args.adapter_tar,
        "out_dir": args.output_dir,
        "scale": scale, "r": r, "lora_alpha": acfg.get("lora_alpha"),
        "math_convention": "tinker_shared_A_per_expert_B (w1/w3) | per_expert_A_shared_B (w2)",
    }

    # 3) MoE per-expert deltas (all 48 layers). Tinker per-expert MoE convention:
    # gate_up = cat([B_w1[e]@A_w1, B_w3[e]@A_w3]); down = B_w2@A_w2[e]; cast bf16 before add.
    gate_up_touched = down_touched = 0
    differ = {}
    for L in range(NUM_LAYERS):
        experts = model.model.layers[L].mlp.experts
        gate_up_param = experts.gate_up_proj   # (E,1536,2048)
        down_param = experts.down_proj         # (E,2048,768)
        A_w1 = adapter[_k(L, "w1", "A")]; B_w1 = adapter[_k(L, "w1", "B")]
        A_w3 = adapter[_k(L, "w3", "A")]; B_w3 = adapter[_k(L, "w3", "B")]
        A_w2 = adapter[_k(L, "w2", "A")]; B_w2 = adapter[_k(L, "w2", "B")]
        if L == 0:
            # per_expert_delta_differ guard over >=4 experts for w1/w2/w3 (gate-half=w1, up-half=w3).
            gu = [build_gate_up_delta(A_w1, B_w1, A_w3, B_w3, e, scale) for e in range(4)]
            differ["w1"] = round(per_expert_differ([g[:GATE_UP_OUT // 2] for g in gu]), 6)
            differ["w3"] = round(per_expert_differ([g[GATE_UP_OUT // 2:] for g in gu]), 6)
            differ["w2"] = round(per_expert_differ([build_down_delta(A_w2, B_w2, e, scale) for e in range(4)]), 6)
        for e in range(NUM_EXPERTS):
            gate_up_param.data[e] += build_gate_up_delta(A_w1, B_w1, A_w3, B_w3, e, scale).to(torch.bfloat16)
            down_param.data[e] += build_down_delta(A_w2, B_w2, e, scale).to(torch.bfloat16)
            gate_up_touched += 1; down_touched += 1
    report["per_expert_delta_differ_check"] = differ
    report["gate_up_touched"] = gate_up_touched
    report["down_touched"] = down_touched
    if not all(differ[w] > 1e-5 for w in ("w1", "w2", "w3")):
        report["status"] = "ABORT_broadcast_merge"
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"ABORT: per-expert deltas not distinct: {differ}", flush=True)
        return 4

    # MoE-only detection (T-04.3-01): derived from adapter key inspection, NOT a trusted flag.
    # An adapter with no self_attn.{q,k,v,o}_proj tensors is MoE-only; the attention and
    # unembed merge stages must be skipped to avoid the three v3 break points (lines 281/374/386).
    is_moe_only = not any(
        any(f".{p}." in k for p in ATTN_PROJS)
        for k in adapter
    )
    report["is_moe_only_adapter"] = is_moe_only

    # 4) Attention via attention-only temp PEFT adapter (q/k/v/o), then merge_and_unload.
    # Skipped for MoE-only adapters (no attn keys present; _build_attention_only_adapter would assert).
    if not is_moe_only:
        q_before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
        attn_dir = tempfile.mkdtemp(prefix="tinker_v3_attn_")
        _build_attention_only_adapter(adapter, acfg, attn_dir)
        model = PeftModel.from_pretrained(model, attn_dir).merge_and_unload()
        q_after = model.model.layers[0].self_attn.q_proj.weight
        report["attention_q_proj_changed"] = bool(
            (q_before - q_after).abs().max().item() > 1e-6)
        assert report["attention_q_proj_changed"], "attention merge did not change q_proj"
    else:
        report["attention_q_proj_changed"] = False
        report["attention_skipped"] = True

    # 5) Manual lm_head LoRA (PEFT skips unembed_tokens because the key is unembed_tokens, not lm_head).
    # Skipped when --exclude-lm-head (D-IT-04 attempt-1) or is_moe_only (no unembed keys in adapter).
    lm_head_excluded = getattr(args, "exclude_lm_head", False) or is_moe_only
    if not lm_head_excluded:
        A_un = adapter["base_model.model.model.unembed_tokens.lora_A.weight"]
        B_un = adapter["base_model.model.model.unembed_tokens.lora_B.weight"]
        model.lm_head.weight.data += build_lm_head_delta(A_un, B_un, scale).to(torch.bfloat16)
        assert tuple(model.lm_head.weight.shape) == (STOCK_VOCAB, model.config.hidden_size)
        report["lm_head_excluded"] = False
        report["lm_head_applied"] = True
        report["lm_head_touched"] = 1
    else:
        report["lm_head_excluded"] = True
        report["lm_head_applied"] = False
        report["lm_head_touched"] = 0
        if is_moe_only:
            report["merge_type"] = "tinker_per_expert_moe_only_NO_attn_NO_lm_head"
        else:
            report["merge_type"] = "tinker_per_expert_moe_plus_peft_attention_NO_lm_head"

    # 6) Save merged model + STOCK tokenizer (never the extended 151,938 tokenizer) to staging.
    tmp_out = args.output_dir + ".tmp_merge"
    if os.path.exists(tmp_out):
        import shutil
        shutil.rmtree(tmp_out)
    os.makedirs(tmp_out, exist_ok=True)
    model.save_pretrained(tmp_out, safe_serialization=True, max_shard_size="5GB")
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(tmp_out)
    # Integrity check (T-0441-02): the served tokenizer must tokenize the task markers the
    # SAME way v3 training did -- as PLAIN TEXT (multi-piece), not single special ids. The
    # extended tokenizer maps <wp_judge> -> one id (151670); the stock one splits it into
    # >1 BPE pieces. We assert the stock behaviour + in-range ids (no OOB vs the 151936 model).
    wp_pieces = len(tok.encode("<wp_judge>", add_special_tokens=False))
    max_id = max(tok.get_vocab().values())
    report["base_vocab"] = STOCK_VOCAB                 # model embedding rows (151936)
    report["tokenizer_vocab"] = len(tok)               # stock tokenizer real token count (151669)
    report["tokenizer_max_id"] = max_id
    report["tokenizer_wp_judge_pieces"] = wp_pieces
    report["tokenizer_is_stock_text_routing"] = wp_pieces > 1
    assert wp_pieces > 1, (
        f"extended tokenizer leaked: <wp_judge> tokenizes to a single id "
        f"({tok.encode('<wp_judge>', add_special_tokens=False)}); v3 trained on STOCK text tokenization"
    )
    assert len(tok) == STOCK_TOK_LEN, (
        f"tokenizer len {len(tok)} != stock {STOCK_TOK_LEN} (wrong/extended tokenizer)"
    )
    assert max_id < STOCK_VOCAB, (
        f"tokenizer max id {max_id} >= model vocab {STOCK_VOCAB} (out-of-range token ids)"
    )
    if os.path.exists(args.output_dir):
        import shutil
        shutil.rmtree(args.output_dir)
    os.rename(tmp_out, args.output_dir)
    import glob
    report["shard_count"] = len(glob.glob(os.path.join(args.output_dir, "model-*-of-*.safetensors")))
    report["wall_clock_sec"] = round(time.time() - t0, 1)
    report["status"] = "staging_written_pending_anchor"
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[DONE] staging merge written to {args.output_dir} in {report['wall_clock_sec']}s", flush=True)
    print(f"  differ {differ} | shards {report['shard_count']} | report {args.report}", flush=True)
    print("  STATUS: staging_written_pending_anchor — DO NOT PROMOTE until 3 anchors pass", flush=True)
    return 0


def main() -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Tinker per-expert MoE LoRA merge -> v3 staging")
    ap.add_argument("--adapter-tar", default=DEFAULT_ADAPTER_TAR,
                    help="Tinker HF PEFT LoRA checkpoint.tar (adapter_config + adapter_model.safetensors)")
    ap.add_argument("--base", default=DEFAULT_BASE, help="Stock base model dir (Qwen3-30B-A3B, vocab 151936)")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                    help="Staging output dir (must be under _staging/ unless --force-canonical)")
    ap.add_argument("--report", default=DEFAULT_REPORT, help="merge_report.json path")
    ap.add_argument("--force-canonical", action="store_true",
                    help="Allow writing outside _staging/ (DANGER: can overwrite canonical). Default off.")
    ap.add_argument("--exclude-lm-head", action="store_true", default=False,
                    help="(D-IT-04) Skip the manual lm_head LoRA stage; set lm_head_excluded=true in "
                         "merge_report.json. Default OFF — omitting this flag reproduces the v3 "
                         "manual_lm_head merge path exactly.")
    args = ap.parse_args()

    # Staging-isolation guard: refuse a non-_staging output dir unless explicitly forced.
    if "_staging/" not in args.output_dir and not args.force_canonical:
        print(f"REFUSE: --output-dir {args.output_dir!r} is not under _staging/ and "
              f"--force-canonical not set. Canonical write blocked.", file=sys.stderr)
        return 3

    # RAM floor: this CPU merge holds the ~57 GiB bf16 model + fp32 deltas; abort below 70 GiB.
    free = _free_ram_gib()
    if free < RAM_FLOOR_GIB:
        print(f"ABORT: free RAM {free:.1f} GiB < floor {RAM_FLOOR_GIB} GiB. "
              f"Close Chromium/heavy apps and retry.", file=sys.stderr)
        return 2

    return _run_merge(args)


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(main())
