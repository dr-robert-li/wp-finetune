#!/usr/bin/env python3
"""D-IT-02 attribution probe: build a SINGLE-COMPONENT merge variant.

Builds either an ATTENTION-ONLY or a MoE-ONLY merge of the wp-reasoning-v3 adapter onto the
stock base, to isolate which component carries the RC-B codegen regression (wp-bench execution
corr 0.417 -> 0.292). Both variants EXCLUDE lm_head (shown irrelevant by D-IT-04 + RC-A), so
attention-vs-MoE is the only variable.

Reuses merge_tinker_v3's exact helpers/conventions — does NOT touch the validated v3 merge path.

  --include moe   -> base + MoE per-expert deltas only      (no attention, no lm_head)
  --include attn  -> base + attention q/k/v/o PEFT merge only (no MoE, no lm_head)

Light sanity (not the full 3-anchor cert — this is a probe, not a promotion candidate):
  moe : per_expert_differ > 1e-5 (real per-expert deltas) AND q_proj UNCHANGED
  attn: q_proj CHANGED AND experts UNCHANGED
Plus the stock-tokenizer integrity asserts carried from the v3 merge.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.merge_tinker_v3 import (  # noqa: E402
    NUM_LAYERS, NUM_EXPERTS, STOCK_VOCAB, STOCK_TOK_LEN, GATE_UP_OUT,
    DEFAULT_BASE, DEFAULT_ADAPTER_TAR,
    _untar_adapter, _load_adapter_tensors, _k,
    build_gate_up_delta, build_down_delta, per_expert_differ,
    _build_attention_only_adapter,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="D-IT-02 single-component probe merge")
    ap.add_argument("--include", required=True, choices=["moe", "attn"])
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--adapter-tar", default=DEFAULT_ADAPTER_TAR)
    ap.add_argument("--output-dir", required=True, help="must be under models/_staging/")
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    if "_staging/" not in args.output_dir:
        print(f"REFUSE: --output-dir {args.output_dir!r} not under _staging/", file=sys.stderr)
        return 2

    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy
    from peft import PeftModel

    t0 = time.time()
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)

    work = tempfile.mkdtemp(prefix="dit02_probe_adapter_")
    _untar_adapter(args.adapter_tar, work)
    with open(os.path.join(work, "adapter_config.json")) as fh:
        acfg = json.load(fh)
    r = int(acfg["r"])
    scale = float(acfg.get("lora_alpha", r)) / float(r)
    adapter = _load_adapter_tensors(work)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, device_map={"": "cpu"}, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    assert model.config.vocab_size == STOCK_VOCAB

    report = {
        "status": "probe_pending", "include": args.include,
        "merge_type": f"dit02_probe_{args.include}_only_NO_lm_head",
        "base_path": args.base, "adapter_tar": args.adapter_tar, "out_dir": args.output_dir,
        "scale": scale, "r": r, "lm_head_excluded": True, "lm_head_applied": False,
    }

    q_before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()

    if args.include == "moe":
        gate_up_touched = down_touched = 0
        differ = {}
        for L in range(NUM_LAYERS):
            experts = model.model.layers[L].mlp.experts
            A_w1, B_w1 = adapter[_k(L, "w1", "A")], adapter[_k(L, "w1", "B")]
            A_w3, B_w3 = adapter[_k(L, "w3", "A")], adapter[_k(L, "w3", "B")]
            A_w2, B_w2 = adapter[_k(L, "w2", "A")], adapter[_k(L, "w2", "B")]
            if L == 0:
                gu = [build_gate_up_delta(A_w1, B_w1, A_w3, B_w3, e, scale) for e in range(4)]
                differ["w1"] = round(per_expert_differ([g[:GATE_UP_OUT // 2] for g in gu]), 6)
                differ["w3"] = round(per_expert_differ([g[GATE_UP_OUT // 2:] for g in gu]), 6)
                differ["w2"] = round(per_expert_differ(
                    [build_down_delta(A_w2, B_w2, e, scale) for e in range(4)]), 6)
            for e in range(NUM_EXPERTS):
                experts.gate_up_proj.data[e] += build_gate_up_delta(A_w1, B_w1, A_w3, B_w3, e, scale).to(torch.bfloat16)
                experts.down_proj.data[e] += build_down_delta(A_w2, B_w2, e, scale).to(torch.bfloat16)
                gate_up_touched += 1; down_touched += 1
        report["per_expert_delta_differ_check"] = differ
        report["gate_up_touched"] = gate_up_touched
        report["down_touched"] = down_touched
        report["attention_excluded"] = True
        if not all(differ[w] > 1e-5 for w in ("w1", "w2", "w3")):
            report["status"] = "ABORT_broadcast_merge"
            Path(args.report).write_text(json.dumps(report, indent=2))
            print(f"ABORT: per-expert deltas not distinct: {differ}", file=sys.stderr)
            return 4
        q_after = model.model.layers[0].self_attn.q_proj.weight
        report["attention_q_proj_changed"] = bool((q_before - q_after).abs().max().item() > 1e-6)
        assert not report["attention_q_proj_changed"], "MoE-only must NOT change q_proj"

    else:  # attn
        attn_dir = tempfile.mkdtemp(prefix="dit02_probe_attn_")
        _build_attention_only_adapter(adapter, acfg, attn_dir)
        # capture an expert weight before to assert MoE untouched
        gu_before = model.model.layers[0].mlp.experts.gate_up_proj.data[0].detach().clone()
        model = PeftModel.from_pretrained(model, attn_dir).merge_and_unload()
        q_after = model.model.layers[0].self_attn.q_proj.weight
        gu_after = model.model.layers[0].mlp.experts.gate_up_proj.data[0]
        report["attention_q_proj_changed"] = bool((q_before - q_after).abs().max().item() > 1e-6)
        report["moe_excluded"] = True
        report["experts_unchanged"] = bool((gu_before - gu_after).abs().max().item() <= 1e-6)
        assert report["attention_q_proj_changed"], "attn-only must change q_proj"
        assert report["experts_unchanged"], "attn-only must NOT change experts"

    # Save + stock tokenizer + integrity asserts (carried from v3 merge)
    tmp_out = args.output_dir + ".tmp_merge"
    if os.path.exists(tmp_out):
        shutil.rmtree(tmp_out)
    os.makedirs(tmp_out, exist_ok=True)
    model.save_pretrained(tmp_out, safe_serialization=True, max_shard_size="5GB")
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(tmp_out)
    wp_pieces = len(tok.encode("<wp_judge>", add_special_tokens=False))
    assert wp_pieces > 1, "extended tokenizer leaked"
    assert len(tok) == STOCK_TOK_LEN, f"tokenizer len {len(tok)} != {STOCK_TOK_LEN}"
    assert max(tok.get_vocab().values()) < STOCK_VOCAB, "OOB token ids"
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.rename(tmp_out, args.output_dir)

    report["shard_count"] = len(glob.glob(os.path.join(args.output_dir, "model-*-of-*.safetensors")))
    report["tokenizer_wp_judge_pieces"] = wp_pieces
    report["tokenizer_vocab"] = len(tok)
    report["wall_clock_sec"] = round(time.time() - t0, 1)
    report["status"] = "probe_built"
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"[DONE] {args.include}-only probe -> {args.output_dir} "
          f"({report['wall_clock_sec']}s, {report['shard_count']} shards)", flush=True)
    print(f"  q_proj_changed={report['attention_q_proj_changed']} "
          f"differ={report.get('per_expert_delta_differ_check')} "
          f"experts_unchanged={report.get('experts_unchanged')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
