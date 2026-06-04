"""checkpoint_parse_check.py — two-mode quality gate script.

Mode 1 (--readiness-gate):
    D-03 dataset readiness gate. Verifies data/reasoning_dataset/metadata.json
    meets the rebuilt-dataset preconditions before any training step.

Mode 2 (--checkpoint-dir):
    RTRN-04 in-process parse-failure abort hook. Loads the merged base + reasoning
    LoRA checkpoint, samples val examples, and exits non-zero if parse-fail > 5%.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Mode 1: D-03 dataset readiness gate
# ---------------------------------------------------------------------------

# Tolerance bands (D-02/D-03)
TOTAL_BAND = (650, 760)
COT_PCT_BAND = (55.0, 65.0)
REPLAY_PCT_BAND = (12.0, 18.0)
REQUIRED_REJECTION_KEYS = {"vendor_contamination", "truncated_invalid", "consistency"}


def verify_dataset_readiness(metadata_path: str | Path) -> bool:
    """Load metadata.json and assert D-03 readiness bands.

    Returns True on all-pass, exits non-zero on any failure.
    """
    path = Path(metadata_path)
    if not path.exists():
        print(f"ERROR: metadata file not found: {path}")
        print(
            "D-03 readiness gate FAILED - run the D-03 backfill "
            "(Phase 4.1 CoT generation + Phase 4.2 re-assembly) upstream first. "
            "Training will not start."
        )
        sys.exit(1)

    with path.open() as fh:
        meta = json.load(fh)

    failed = False

    # --- total_examples band ---
    total = meta.get("total_examples", 0)
    lo, hi = TOTAL_BAND
    if not (lo <= total <= hi):
        print(
            f"FAIL  total_examples={total}  expected band [{lo}, {hi}]"
        )
        failed = True

    # --- cot_percent band ---
    cot_pct = meta.get("mix", {}).get("cot_percent", 0.0)
    lo, hi = COT_PCT_BAND
    if not (lo <= cot_pct <= hi):
        print(
            f"FAIL  mix.cot_percent={cot_pct}  expected band [{lo}, {hi}]"
        )
        failed = True

    # --- replay_percent band ---
    replay_pct = meta.get("mix", {}).get("replay_percent", 0.0)
    lo, hi = REPLAY_PCT_BAND
    if not (lo <= replay_pct <= hi):
        print(
            f"FAIL  mix.replay_percent={replay_pct}  expected band [{lo}, {hi}]"
        )
        failed = True

    # --- rejection_counts key presence ---
    rejection_counts = meta.get("rejection_counts", {})
    missing_keys = REQUIRED_REJECTION_KEYS - set(rejection_counts.keys())
    if missing_keys:
        print(
            f"FAIL  rejection_counts missing keys: {sorted(missing_keys)}  "
            f"(proves vendor/truncation + consistency gates were applied)"
        )
        failed = True

    if failed:
        print(
            "D-03 readiness gate FAILED - run the D-03 backfill "
            "(Phase 4.1 CoT generation + Phase 4.2 re-assembly) upstream first. "
            "Training will not start."
        )
        sys.exit(1)

    print("D-03 readiness gate PASSED")
    return True


# ---------------------------------------------------------------------------
# Mode 2: RTRN-04 in-process parse-failure abort hook (Path C)
# ---------------------------------------------------------------------------

DEFAULT_BASE = "models/qwen3-30b-wp-30_70-merged"
DEFAULT_VAL_JSONL = "data/reasoning_dataset/openai_val.jsonl"
DEFAULT_N = 10
DEFAULT_THRESHOLD = 0.05
DEFAULT_LOAD_IN_4BIT = True
DEFAULT_MAX_MEMORY_GIB = 80
DEFAULT_MAX_NEW_TOKENS = 2048           # 04.3-03: discriminator slice (was hardcoded 1024)
DEFAULT_INCLUDE_STREAMS = "cot,ctf"     # 04.3-03: match the 04.3-02 structural slice

# Keyword sets for the 04.3-03 runtime MoE-binding probe (representation-agnostic).
_LORA_KW = ("lora", "parametriz", "adapter", "wrapper")
_EXPERT_KW = ("experts", "gate_up_proj", "down_proj")
_ATTN_KW = ("q_proj", "k_proj", "v_proj", "o_proj")


# ---------------------------------------------------------------------------
# 04.3-03 helpers (shared with capture_reasoning_responses semantics)
# ---------------------------------------------------------------------------

def _stream_of(row: dict) -> str:
    """Task type = metadata.stream (cot/ctf/replay) — mirrors capture_reasoning_responses."""
    return row.get("metadata", {}).get("stream", "")


def _user_messages(row: dict) -> list[dict]:
    """USER-ONLY prompt construction — mirrors capture_reasoning_responses._user_messages
    (line 64-65) so the Unsloth arms match the merged vLLM capture's prompt set."""
    return [m for m in row.get("messages", []) if m.get("role") == "user"]


def _structural_histogram(responses: list[str], parsed_list: list) -> dict:
    """Capture-compatible structural histogram. SAME keys as
    capture_reasoning_responses (n_total / n_with_close_tag / n_with_judge_output /
    n_parseable_scores / parseable_rate) so terse = n_total - n_with_close_tag is
    computed IDENTICALLY to 04.3-02."""
    n = len(responses)
    hist = {
        "n_total": n,
        "n_with_close_tag": sum("[/REASONING]" in (r or "") for r in responses),
        "n_with_judge_output": sum("<judge_output>" in (r or "") for r in responses),
        "n_with_corrected_code": sum("<corrected_code>" in (r or "") for r in responses),
        "n_parseable_scores": sum(p is not None and "overall_score" in p for p in parsed_list),
    }
    hist["parseable_rate"] = (hist["n_parseable_scores"] / n) if n else 0.0
    return hist


def probe_moe_binding(model, tokenizer) -> dict:
    """04.3-03 runtime MoE-binding probe (DUMP + representation-agnostic positive test).

    load_adapter on a Qwen3-MoE `target_parameters` fused-expert adapter is documented-
    UNVERIFIED: the LoRA may live as child lora_A/lora_B modules OR as a parametrization on
    the raw fused nn.Parameter (no child module). So this probe does TWO things:
      (1) DUMP the live named_modules / named_parameters whose names match lora/expert/
          parametrization keywords (eyeball evidence — written to binding_dryrun.md).
      (2) A REPRESENTATION-AGNOSTIC POSITIVE TEST (the real verdict): forward-ACTIVATION
          delta on the experts submodule — NOT a base-weight delta (a correct unmerged
          load leaves base weights unchanged, so a weight delta would false-fail). We
          capture the experts module's INPUT on one adapter-enabled forward, then re-run
          that SAME submodule on the SAME input with the adapter disabled; a nonzero output
          delta is purely the expert-LoRA contribution (input held fixed → attention cannot
          explain it). This positively isolates "expert-LoRA runtime-active".

    Returns a report dict; the caller decides BOUND vs BINDING_FAILED from `verdict`.
    Never silently passes: verdict is BINDING_FAILED unless the activation delta (preferred)
    OR an unambiguous structural expert-LoRA module positively confirms runtime binding.
    """
    import torch

    rep: dict = {}
    mod_names = [n for n, _ in model.named_modules()]
    par_names = [n for n, _ in model.named_parameters()]

    def _kw(name, kws):
        ln = name.lower()
        return any(k in ln for k in kws)

    mod_hits = [n for n in mod_names if _kw(n, _LORA_KW + _EXPERT_KW)]
    par_hits = [n for n in par_names if _kw(n, _LORA_KW + _EXPERT_KW)]
    structural_expert = sorted({
        n for n in (mod_hits + par_hits)
        if _kw(n, _EXPERT_KW) and _kw(n, _LORA_KW)
    })
    structural_attn = sorted({
        n for n in (mod_hits + par_hits)
        if _kw(n, _ATTN_KW) and _kw(n, _LORA_KW)
    })
    rep["module_hits_sample"] = mod_hits[:80]
    rep["param_hits_sample"] = par_hits[:80]
    rep["structural_expert_lora"] = structural_expert[:40]
    rep["structural_attn_lora"] = structural_attn[:40]
    rep["has_disable_adapter"] = hasattr(model, "disable_adapter")
    rep["has_disable_adapters"] = hasattr(model, "disable_adapters")

    # Locate the experts submodule (representation-agnostic: by name suffix).
    experts_name = next(
        (n for n in mod_names if n.lower().endswith("mlp.experts") or n.lower().endswith(".experts")),
        None,
    )
    rep["experts_module"] = experts_name

    # (2) forward-ACTIVATION delta on the experts submodule (the positive isolating test).
    activation_delta = None
    activation_err = None
    try:
        if experts_name is None:
            raise RuntimeError("could not locate an mlp.experts submodule by name")
        experts_mod = dict(model.named_modules())[experts_name]

        captured = {}

        def _pre_hook(_m, args, kwargs):
            # capture the experts module input (positional or kw) without altering it
            captured["args"] = tuple(a.detach().clone() if torch.is_tensor(a) else a for a in args)
            captured["kwargs"] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
            return None

        h = experts_mod.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        # one tiny forward to capture the experts input + adapter-on output
        toks = tokenizer("Evaluate this WordPress code: <?php echo 1; ?>", return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**toks)
        h.remove()

        if "args" not in captured:
            raise RuntimeError("experts forward-pre-hook did not fire")

        def _run_experts():
            with torch.no_grad():
                return experts_mod(*captured["args"], **captured["kwargs"])

        out_on = _run_experts()

        # adapter-disabled re-run on the SAME captured input
        disabled_ctx = None
        if hasattr(model, "disable_adapter"):
            disabled_ctx = model.disable_adapter
        if disabled_ctx is None:
            raise RuntimeError("no disable_adapter() context on the wrapped model")
        with disabled_ctx():
            out_off = _run_experts()

        def _first_tensor(x):
            if torch.is_tensor(x):
                return x
            if isinstance(x, (tuple, list)):
                for e in x:
                    if torch.is_tensor(e):
                        return e
            return None

        t_on, t_off = _first_tensor(out_on), _first_tensor(out_off)
        if t_on is None or t_off is None:
            raise RuntimeError("experts output is not a tensor; cannot diff activations")
        activation_delta = (t_on.float() - t_off.float()).abs().max().item()
    except Exception as e:  # noqa: BLE001 — any failure here is recorded, not fatal-by-crash
        activation_err = f"{type(e).__name__}: {e}"
    rep["activation_delta_max"] = activation_delta
    rep["activation_delta_error"] = activation_err

    # ---- verdict (never silently clean) ----
    EPS = 1e-6
    if activation_delta is not None and activation_delta > EPS:
        rep["verdict"] = "BOUND"
        rep["verdict_basis"] = (
            f"forward-activation delta on {experts_name} = {activation_delta:.3e} > {EPS:g} "
            "(expert-LoRA runtime-active; input held fixed so attention cannot explain it)"
        )
    elif activation_delta is not None and activation_delta <= EPS:
        # activation test ran and found NO expert-LoRA effect -> the experts are unaffected
        rep["verdict"] = "BINDING_FAILED"
        rep["verdict_basis"] = (
            f"forward-activation delta on {experts_name} = {activation_delta:.3e} <= {EPS:g} "
            "(adapter does NOT alter expert outputs — attention-only or unbound)"
        )
    elif structural_expert:
        # activation test could not run (no disable_adapter / no experts module diffable),
        # but a live expert-LoRA structure IS present on the model object -> positive (reduced confidence).
        rep["verdict"] = "BOUND"
        rep["verdict_basis"] = (
            "activation test unavailable (" + str(activation_err) + "); "
            f"FALLBACK: {len(structural_expert)} live expert-LoRA module/param name(s) present on the "
            "model object (e.g. " + (structural_expert[0] if structural_expert else "") + ")"
        )
    else:
        rep["verdict"] = "BINDING_FAILED"
        rep["verdict_basis"] = (
            "activation test unavailable (" + str(activation_err) + ") AND no live expert-LoRA "
            "module/param found on the model object — the fused target_parameters adapter is not "
            "positively confirmed runtime-active (documented-unverified risk realized)"
        )
    return rep


def check_close_tag_survives(tokenizer) -> dict:
    """skip_special_tokens survival: [/REASONING] is NOT a train-config special token
    (<wp_gen>/<wp_judge>), so it should survive skip_special_tokens=True as ordinary text.
    Confirm empirically via a tokenizer round-trip; if it does NOT survive, the histogram
    must switch to a skip_special_tokens=False decode."""
    probe = "reasoning prose[/REASONING]<judge_output>{}</judge_output>"
    ids = tokenizer(probe, add_special_tokens=False).input_ids
    dec_skip = tokenizer.decode(ids, skip_special_tokens=True)
    dec_keep = tokenizer.decode(ids, skip_special_tokens=False)
    return {
        "survives_skip_special_tokens": "[/REASONING]" in dec_skip,
        "survives_keep_special_tokens": "[/REASONING]" in dec_keep,
        "decoded_skip_sample": dec_skip[:160],
    }


def _load_model(base: str, checkpoint_dir: str | None, no_adapter: bool,
                load_in_4bit: bool, max_memory_gib: int):
    """from_pretrained(base) [+ load_adapter unless no_adapter] + for_inference.
    Returns (model, tokenizer). Records nothing — caller probes binding."""
    import torch
    from unsloth import FastLanguageModel

    max_memory = {0: f"{max_memory_gib}GiB", "cpu": "20GiB"}
    print(f"[parse-check] Loading base model from {base} "
          f"(load_in_4bit={load_in_4bit}, max_memory={max_memory}) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base,
        max_seq_length=8192,
        load_in_4bit=load_in_4bit,
        dtype=torch.bfloat16,
        max_memory=max_memory,
    )
    if not no_adapter and checkpoint_dir:
        print(f"[parse-check] Loading adapter from {checkpoint_dir} ...")
        model.load_adapter(checkpoint_dir)
    else:
        print("[parse-check] --no-adapter: skipping load_adapter (merged-base arm).")
    return model, tokenizer


def run_binding_dryrun(
    checkpoint_dir: str,
    base: str = "models/qwen3-30b-wp-30_70-merged-v2",
    out_md: str = "output/format_stability/discriminator/binding_dryrun.md",
    load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
    max_memory_gib: int = DEFAULT_MAX_MEMORY_GIB,
) -> dict:
    """04.3-03 Task 1: light (n<=1) dry-check. Loads base+adapter, probes the runtime
    MoE binding (BEFORE and AFTER for_inference, to also catch a silent auto-merge),
    checks [/REASONING] survival, and writes everything to binding_dryrun.md. Does NOT
    run a full capture. The verdict (BOUND / BINDING_FAILED) is written verbatim so a
    human / the next task can gate on it."""
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = _load_model(base, checkpoint_dir, no_adapter=False,
                                   load_in_4bit=load_in_4bit, max_memory_gib=max_memory_gib)

    print("[binding] probing runtime MoE binding AFTER load_adapter (before for_inference) ...")
    probe_after_load = probe_moe_binding(model, tokenizer)

    model = FastLanguageModel.for_inference(model)
    print("[binding] probing AGAIN AFTER for_inference (catch silent auto-merge) ...")
    probe_after_infer = probe_moe_binding(model, tokenizer)

    tag = check_close_tag_survives(tokenizer)

    # Auto-merge anomaly: expert-LoRA present after load but the activation effect vanished
    # after for_inference would mean the arm is no longer truly unmerged.
    auto_merge_anomaly = (
        probe_after_load.get("verdict") == "BOUND"
        and probe_after_infer.get("verdict") == "BINDING_FAILED"
    )

    final_verdict = probe_after_infer.get("verdict", "BINDING_FAILED")
    if auto_merge_anomaly:
        final_verdict = "BINDING_FAILED"

    out_path = Path(out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 04.3-03 Task 1 — runtime MoE-binding dry-check\n",
        f"- base: `{base}`",
        f"- adapter (checkpoint): `{checkpoint_dir}`",
        f"- load_in_4bit: {load_in_4bit}",
        f"\n## VERDICT: {final_verdict}\n",
        f"- after-load verdict: {probe_after_load.get('verdict')} — {probe_after_load.get('verdict_basis')}",
        f"- after-for_inference verdict: {probe_after_infer.get('verdict')} — {probe_after_infer.get('verdict_basis')}",
        f"- auto_merge_anomaly (bound-then-unbound across for_inference): {auto_merge_anomaly}",
        f"\n## [/REASONING] skip_special_tokens survival\n",
        f"- survives skip_special_tokens=True: {tag['survives_skip_special_tokens']}",
        f"- survives skip_special_tokens=False: {tag['survives_keep_special_tokens']}",
        f"- decoded sample: `{tag['decoded_skip_sample']}`",
        f"\n## Structural evidence (after for_inference)\n",
        f"- experts module located: `{probe_after_infer.get('experts_module')}`",
        f"- has disable_adapter / disable_adapters: "
        f"{probe_after_infer.get('has_disable_adapter')} / {probe_after_infer.get('has_disable_adapters')}",
        f"- activation_delta_max: {probe_after_infer.get('activation_delta_max')} "
        f"(err: {probe_after_infer.get('activation_delta_error')})",
        "- live expert-LoRA module/param names (sample):",
        "```",
        *(probe_after_infer.get("structural_expert_lora") or ["(none found)"]),
        "```",
        "- live attention-LoRA names (sample):",
        "```",
        *(probe_after_infer.get("structural_attn_lora") or ["(none found)"]),
        "```",
        "- all lora/expert/parametrization module-name hits (sample):",
        "```",
        *(probe_after_infer.get("module_hits_sample") or ["(none)"]),
        "```",
        "- all lora/expert/parametrization param-name hits (sample):",
        "```",
        *(probe_after_infer.get("param_hits_sample") or ["(none)"]),
        "```",
        "\nexpert  — `[/REASONING]` is the structural terse signal; this dry-check confirms the "
        "expert-LoRA is (or is not) runtime-bound before the full 3-arm capture.\n",
    ]
    out_path.write_text("\n".join(str(l) for l in lines))
    print(f"[binding] dry-check VERDICT={final_verdict} -> {out_path}")
    return {"verdict": final_verdict, "tag": tag,
            "after_load": probe_after_load, "after_infer": probe_after_infer}


def run_checkpoint_parse_check(
    checkpoint_dir: str | None,
    base: str = DEFAULT_BASE,
    val_jsonl: str = DEFAULT_VAL_JSONL,
    n: int = DEFAULT_N,
    threshold: float = DEFAULT_THRESHOLD,
    load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
    max_memory_gib: int = DEFAULT_MAX_MEMORY_GIB,
    include_streams: str = DEFAULT_INCLUDE_STREAMS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    no_adapter: bool = False,
    out_path: str | None = None,
) -> bool:
    """Load base [+ reasoning LoRA] in-process, sample n cot+ctf val examples, and
    measure BOTH the structural [/REASONING] terse histogram (04.3-03 discriminator
    signal) AND the RTRN-04 parse-fail rate.

    04.3-03 changes (additive): stream filter (cot+ctf) applied BEFORE the --n cap;
    user-only prompts; max_new_tokens default 2048; optional --no-adapter (merged arm);
    a structural histogram written to out_path (caller-supplied name, anti-clobber); a
    runtime MoE-binding guard re-asserted at load time when an adapter is loaded.

    When out_path is set (discriminator mode) the structural histogram is the deliverable
    and the function returns True without the RTRN-04 sys.exit abort. When out_path is
    None (legacy RTRN-04 Mode 2) the parse-fail threshold abort is preserved.
    """
    import torch
    from unsloth import FastLanguageModel
    from eval.eval_judge import parse_judge_response  # noqa: PLC0415

    model, tokenizer = _load_model(base, checkpoint_dir, no_adapter=no_adapter,
                                   load_in_4bit=load_in_4bit, max_memory_gib=max_memory_gib)

    # 04.3-03: when an adapter IS loaded, the unmerged arm MUST re-assert the runtime
    # MoE-binding guard — a silent attention-only / auto-merged bind would make a "clean"
    # reading false. Probe AFTER load (before for_inference).
    if not no_adapter and checkpoint_dir:
        guard = probe_moe_binding(model, tokenizer)
        print(f"[binding] runtime MoE-binding guard: {guard['verdict']} — {guard['verdict_basis']}")
        if guard["verdict"] != "BOUND":
            print("[binding] BINDING_FAILED — the unmerged arm is not a faithful merge-bypass. "
                  "Refusing to emit terse numbers (the discriminator would be confounded).")
            sys.exit(7)  # distinct code so the caller/Task-2 can record DISCRIMINATOR: BINDING_FAILED

    model = FastLanguageModel.for_inference(model)

    # Decide the decode mode once, from the survival check (avoid silent undercount).
    tag = check_close_tag_survives(tokenizer)
    skip_special = bool(tag["survives_skip_special_tokens"])
    if not skip_special:
        print("[parse-check] NOTE: [/REASONING] does NOT survive skip_special_tokens=True; "
              "decoding with skip_special_tokens=False so the structural split is not undercounted.")

    val_path = Path(val_jsonl)
    if not val_path.exists():
        print(f"[parse-check] ERROR: val file not found: {val_path}")
        sys.exit(1)

    # Stream filter (cot+ctf) BEFORE the --n cap (so --n 120 yields 120 cot+ctf rows).
    streams = {s.strip() for s in include_streams.split(",") if s.strip()}
    rows = []
    with val_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    samples = [r for r in rows if _stream_of(r) in streams]
    if not samples:
        print(f"[parse-check] ERROR: no rows matched include-streams={sorted(streams)}")
        sys.exit(1)
    samples = samples[:n]
    actual_n = len(samples)
    print(f"[parse-check] Evaluating {actual_n} {sorted(streams)} samples "
          f"(max_new_tokens={max_new_tokens}, threshold={threshold:.0%}) ...")

    responses: list[str] = []
    parsed_list: list = []
    parse_fail_count = 0
    for i, example in enumerate(samples):
        prompt_messages = _user_messages(example)  # USER-ONLY (match merged capture)

        inputs = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )

        generated_ids = outputs[0][inputs.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=skip_special)
        responses.append(generated_text)

        parsed = parse_judge_response(generated_text)
        parsed_list.append(parsed)
        is_fail = parsed is None or "overall_score" not in parsed
        if is_fail:
            parse_fail_count += 1
        if is_fail and parse_fail_count <= 3:
            print(f"\n[parse-check] === FAILED SAMPLE {parse_fail_count} (idx={i}) ===")
            print(generated_text[:2000])
            print(f"[parse-check] === END SAMPLE {parse_fail_count} ===\n")
        if (i + 1) % 10 == 0:
            print(f"[parse-check]   {i+1}/{actual_n} evaluated  "
                  f"parse_fail_rate={parse_fail_count/(i+1):.1%}")

    hist = _structural_histogram(responses, parsed_list)
    hist["base"] = base
    hist["adapter"] = None if no_adapter else checkpoint_dir
    hist["include_streams"] = sorted(streams)
    hist["max_new_tokens"] = max_new_tokens
    hist["load_in_4bit"] = load_in_4bit
    hist["skip_special_tokens"] = skip_special
    hist["parse_fail_count"] = parse_fail_count
    terse = hist["n_total"] - hist["n_with_close_tag"]
    print(f"[parse-check] structural histogram: {json.dumps(hist)}")
    print(f"[parse-check] terse (n_total - n_with_close_tag) = {terse}/{hist['n_total']} "
          f"= {terse/hist['n_total']:.1%}" if hist["n_total"] else "[parse-check] terse: n=0")

    if out_path:
        op = Path(out_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(hist, indent=2))
        print(f"[parse-check] structural histogram -> {op}")
        return True  # discriminator mode: histogram is the deliverable, no RTRN-04 abort

    # Legacy RTRN-04 Mode 2: preserve the parse-fail threshold abort.
    parse_fail_rate = parse_fail_count / actual_n
    print(f"[parse-check] Parse-fail rate: {parse_fail_count}/{actual_n} = {parse_fail_rate:.1%}")
    if parse_fail_rate > threshold:
        print(f"[parse-check] ABORT: parse-fail rate {parse_fail_rate:.1%} exceeds threshold "
              f"{threshold:.0%} (RTRN-04). Training run aborted.")
        sys.exit(1)
    print(f"[parse-check] PASSED: parse-fail rate {parse_fail_rate:.1%} <= threshold "
          f"{threshold:.0%} (RTRN-04).")
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="checkpoint_parse_check: D-03 readiness gate + RTRN-04 parse-failure abort hook"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--readiness-gate",
        action="store_true",
        help="Run D-03 dataset readiness gate against metadata.json",
    )
    mode.add_argument(
        "--checkpoint-dir",
        metavar="DIR",
        help="Run RTRN-04 / discriminator check against this checkpoint (load_adapter on --base)",
    )
    mode.add_argument(
        "--no-adapter",
        action="store_true",
        help="04.3-03 merged-base arm: run the capture on --base with NO load_adapter "
             "(the merge already baked the adapter in)",
    )

    # readiness-gate args
    parser.add_argument(
        "--metadata",
        default="data/reasoning_dataset/metadata.json",
        metavar="PATH",
        help="Path to metadata.json (default: data/reasoning_dataset/metadata.json)",
    )

    # checkpoint-dir args
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        metavar="DIR",
        help=f"Merged base model directory (default: {DEFAULT_BASE})",
    )
    parser.add_argument(
        "--val-jsonl",
        default=DEFAULT_VAL_JSONL,
        metavar="PATH",
        help=f"Validation JSONL file (default: {DEFAULT_VAL_JSONL})",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        metavar="INT",
        help=f"Number of val samples to evaluate (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        metavar="FLOAT",
        help=f"Parse-fail rate threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--no-4bit",
        dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit loading (default: enabled, prevents GB10 unified-mem OOM cascade)",
    )
    parser.set_defaults(load_in_4bit=DEFAULT_LOAD_IN_4BIT)
    parser.add_argument(
        "--max-memory-gib",
        type=int,
        default=DEFAULT_MAX_MEMORY_GIB,
        metavar="INT",
        help=f"Per-device max memory cap in GiB (default: {DEFAULT_MAX_MEMORY_GIB})",
    )
    # 04.3-03 discriminator args
    parser.add_argument(
        "--include-streams",
        default=DEFAULT_INCLUDE_STREAMS,
        metavar="LIST",
        help=f"Comma list of metadata.stream values to keep (default: {DEFAULT_INCLUDE_STREAMS}); "
             "applied BEFORE the --n cap so --n yields N filtered rows",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        metavar="INT",
        help=f"Generation budget (default: {DEFAULT_MAX_NEW_TOKENS}; the discriminator slice — "
             "1024 would bias toward more terse)",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="Write the structural [/REASONING] histogram here (caller-supplied name, anti-clobber); "
             "discriminator mode — suppresses the RTRN-04 parse-fail abort",
    )
    parser.add_argument(
        "--binding-dryrun",
        action="store_true",
        help="04.3-03 Task 1: load base+adapter, probe the runtime MoE binding (+ auto-merge + "
             "[/REASONING] survival), write binding_dryrun.md, and exit — NO full capture",
    )
    parser.add_argument(
        "--binding-out",
        default="output/format_stability/discriminator/binding_dryrun.md",
        metavar="PATH",
        help="Where --binding-dryrun writes its report",
    )

    args = parser.parse_args()

    if args.readiness_gate:
        verify_dataset_readiness(args.metadata)
    elif args.binding_dryrun:
        if not args.checkpoint_dir:
            parser.error("--binding-dryrun requires --checkpoint-dir (it loads the adapter to probe binding)")
        run_binding_dryrun(
            checkpoint_dir=args.checkpoint_dir,
            base=args.base,
            out_md=args.binding_out,
            load_in_4bit=args.load_in_4bit,
            max_memory_gib=args.max_memory_gib,
        )
    else:
        run_checkpoint_parse_check(
            checkpoint_dir=args.checkpoint_dir,  # None on the --no-adapter merged arm
            base=args.base,
            val_jsonl=args.val_jsonl,
            n=args.n,
            threshold=args.threshold,
            load_in_4bit=args.load_in_4bit,
            max_memory_gib=args.max_memory_gib,
            include_streams=args.include_streams,
            max_new_tokens=args.max_new_tokens,
            no_adapter=args.no_adapter,
            out_path=args.out,
        )


if __name__ == "__main__":
    main()
