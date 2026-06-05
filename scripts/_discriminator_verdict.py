#!/usr/bin/env python
"""04.3-03 Task 3: compute terse rates + Wilson 95% CIs for the three bf16 discriminator arms,
emit the canonical machine-checkable tokens, apply the PRE-REGISTERED (bf16) thresholds, and
write summary.md + 04.3-03-RESULTS.md with EXACTLY ONE canonical DISCRIMINATOR verdict.

Terse = n_total - n_with_close_tag (structural [/REASONING] split, identical to 04.3-02).
NO rubric_scorer, NO string-grep. Wilson interval, z=1.96.

Pre-registered (bf16) thresholds (committed BEFORE any capture; never call a winner on
overlapping CIs — 04.3-02 discipline):
  MERGE_ARTIFACT  : unmerged CLEAN (rate<=0.10 AND upper<=0.15) AND merged-Unsloth COLLAPSES
                    (Wilson lower>0.15) AND primary CIs DISJOINT (CI_OVERLAP FALSE) AND the two
                    MERGED arms AGREE (ENGINE_CI_OVERLAP TRUE).
  TRAINING_UNDERIMPRINT : unmerged ALSO collapses (Wilson lower>0.15) AND merged collapses
                    (and/or primary CIs OVERLAP with both arms well above 10%).
  INCONCLUSIVE    : gray-band / overlapping-near-10% / engine divergence (ENGINE_CI_OVERLAP FALSE).
  BINDING_FAILED  : the runtime MoE-binding guard never positively bound the expert-LoRA
                    (binding_dryrun.md or the ARM-3 load reported BINDING_FAILED) — the unmerged
                    arm is confounded; NO terse numbers fabricated.
"""
import argparse
import json
import math
import os

DISC = "output/format_stability/discriminator"
RESULTS = ".planning/phases/04.3-reasoning-fine-tune-inserted/04.3-03-RESULTS.md"
Z = 1.96


def wilson(k, n, z=Z):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def overlap(a, b):
    # a, b are (rate, lower, upper); intervals overlap iff l_a <= u_b AND l_b <= u_a
    return a[1] <= b[2] and b[1] <= a[2]


def load_arm(name):
    h = json.load(open(os.path.join(DISC, name)))
    n = h["n_total"]
    terse = n - h["n_with_close_tag"]
    p, lo, hi = wilson(terse, n)
    return {"name": name, "n_total": n, "n_close": h["n_with_close_tag"],
            "terse": terse, "rate": p, "lo": lo, "hi": hi}


def fmt_arm(label, src, engine, prec, a):
    return (f"- **{label}** ({src}, {engine}, {prec}): n_total={a['n_total']} "
            f"n_with_close_tag={a['n_close']} terse={a['terse']} "
            f"rate={a['rate']:.4f} Wilson95%=[{a['lo']:.4f}, {a['hi']:.4f}]")


def write_binding_failed(reason):
    body = (
        "# 04.3-03 Discriminator — RESULTS\n\n"
        "DISCRIMINATOR: BINDING_FAILED\n\n"
        f"The runtime MoE-binding guard did not positively confirm a nonzero expert-LoRA "
        f"activation delta ({reason}). `model.load_adapter(checkpoint-72)` no-ops or "
        f"auto-merges the fused-MoE expert-LoRA on this load path, so the UNMERGED arm cannot "
        f"serve as a faithful merge-bypass — the discriminator is itself confounded. NO terse "
        f"numbers are fabricated.\n\n"
        "Recommended next discriminator: a load path that DOES bind the fused-MoE LoRA at "
        "runtime (transformers+PEFT with the Unsloth-contiguous per-expert convention, or a "
        "manual per-expert delta application bypassing the custom merge), re-probed by the same "
        "activation-delta guard before any capture.\n\n"
        "Boundaries (hold on every branch): Steps 5-8 of 04.3-REOPEN-PLAN.md remain DEFERRED to "
        "the follow-up plan keyed off this verdict. ckpt-72 is NOT promoted; "
        "models/qwen3-30b-wp-30_70-merged-v2 remains the certified fallback; reasoning-merged + "
        "merged-v2 untouched.\n"
    )
    open(os.path.join(DISC, "summary.md"), "w").write(
        "# Discriminator summary\n\nDISCRIMINATOR: BINDING_FAILED\n\n" + reason + "\n")
    open(RESULTS, "w").write(body)
    print("DISCRIMINATOR: BINDING_FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binding-failed", action="store_true",
                    help="force the BINDING_FAILED branch (guard never bound the expert-LoRA)")
    ap.add_argument("--binding-md", default=os.path.join(DISC, "binding_dryrun.md"))
    args = ap.parse_args()

    os.makedirs(DISC, exist_ok=True)

    binding_failed = args.binding_failed
    reason = "explicit --binding-failed"
    if not binding_failed and os.path.exists(args.binding_md):
        import re as _re
        txt = open(args.binding_md).read()
        # Gate on the CANONICAL verdict line only — "BINDING_FAILED" also appears in
        # intermediate-probe / auto-merge-anomaly lines on a BOUND result (false-positive trap).
        if _re.search(r'^## VERDICT: BINDING_FAILED', txt, _re.M):
            binding_failed = True
            reason = "binding_dryrun.md recorded BINDING_FAILED"
    if binding_failed:
        write_binding_failed(reason)
        return

    vllm = load_arm("merged72_vllm_histogram.json")       # arm 1 merged vLLM bf16
    mu = load_arm("merged72_unsloth_histogram.json")      # arm 2 merged Unsloth bf16
    un = load_arm("unmerged72_unsloth_histogram.json")    # arm 3 unmerged Unsloth bf16

    # n_total identity across the three arms (same cot+ctf @120 slice)
    ns = {vllm["n_total"], mu["n_total"], un["n_total"]}
    slice_ok = len(ns) == 1

    eng = (vllm["rate"], vllm["lo"], vllm["hi"])
    mut = (mu["rate"], mu["lo"], mu["hi"])
    unt = (un["rate"], un["lo"], un["hi"])
    ENGINE_CI_OVERLAP = overlap(eng, mut)   # arm1 vs arm2 (two MERGED arms) — must be TRUE for MERGE_ARTIFACT
    CI_OVERLAP = overlap(mut, unt)          # arm2 vs arm3 (primary) — must be FALSE for MERGE_ARTIFACT

    unmerged_clean = (un["rate"] <= 0.10) and (un["hi"] <= 0.15)
    unmerged_collapses = un["lo"] > 0.15
    merged_collapses = mu["lo"] > 0.15

    if not slice_ok:
        verdict = "INCONCLUSIVE"
        rationale = (f"n_total differs across arms ({sorted(ns)}) — the three arms did NOT cover "
                     f"an identical slice, so a clean comparison is not supportable. "
                     f"Next: re-capture all three arms on the identical cot+ctf @120 slice.")
    elif not ENGINE_CI_OVERLAP:
        verdict = "INCONCLUSIVE"
        rationale = ("ENGINE_CI_OVERLAP: FALSE — the two MERGED arms (vLLM-bf16 vs Unsloth-bf16) "
                     "diverge materially on identical weights, so a vLLM/Unsloth engine gap (not "
                     "merge-math) could explain a merged collapse; a clean MERGE_ARTIFACT call is "
                     "not supportable. Next: reconcile the engine gap before re-judging the merge.")
    elif unmerged_clean and merged_collapses and not CI_OVERLAP:
        verdict = "MERGE_ARTIFACT"
        rationale = ("Unmerged-72 is CLEAN (rate<=0.10, Wilson upper<=0.15) while merged-72-Unsloth "
                     "COLLAPSES (Wilson lower>0.15) with DISJOINT primary CIs (CI_OVERLAP: FALSE) "
                     "and the two merged arms AGREE (ENGINE_CI_OVERLAP: TRUE — engine delta "
                     "immaterial). The custom per-expert fused-MoE merge math drops the format "
                     "imprint. Corrective branch (DEFERRED to a follow-up plan): rewrite "
                     "scripts/_p0_merge_unsloth_static_moe.py — NO GPU retrain.")
    elif unmerged_collapses and merged_collapses:
        verdict = "TRAINING_UNDERIMPRINT"
        rationale = ("Both the merged AND the unmerged (load_adapter, merge-bypassed) arms COLLAPSE "
                     "(Wilson lower>0.15) — the LoRA adapter is too weak vs the base prior even when "
                     "applied faithfully, so the collapse is not a merge artifact. Corrective branch "
                     "(DEFERRED): a NEW corrective-training plan with a stronger format signal (NOT "
                     "the refuted Path-A/Path-B).")
    else:
        verdict = "INCONCLUSIVE"
        rationale = ("Neither the clean-vs-collapse (MERGE_ARTIFACT) nor the both-collapse "
                     "(TRAINING_UNDERIMPRINT) pattern holds cleanly: unmerged in the 10-15% gray "
                     f"band or primary CIs overlap near 10% (CI_OVERLAP: {'TRUE' if CI_OVERLAP else 'FALSE'}, "
                     f"unmerged rate={un['rate']:.4f} upper={un['hi']:.4f}). Next discriminator: widen "
                     "n>=300 on an identical slice for both Unsloth arms, or a per-stream breakdown.")

    # ---- summary.md ----
    lines = [
        "# Discriminator summary — three bf16 arms (04.3-03)\n",
        fmt_arm("merged-72-vLLM-bf16 (ARM 1)", "reasoning-merged", "vLLM", "bf16", vllm),
        fmt_arm("merged-72-Unsloth-bf16 (ARM 2)", "reasoning-merged", "Unsloth", "bf16", mu),
        fmt_arm("unmerged-72-Unsloth-bf16 (ARM 3)", "merged-v2 + load_adapter(ckpt-72)", "Unsloth", "bf16", un),
        "",
        f"n_total identical across arms: {slice_ok} ({sorted(ns)})",
        "",
        "## Engine cross-check (PURE engine delta — both merged arms bf16, quantization NOT a variable)",
        f"merged-vLLM-bf16 vs merged-Unsloth-bf16 Wilson CIs "
        f"[{vllm['lo']:.4f},{vllm['hi']:.4f}] vs [{mu['lo']:.4f},{mu['hi']:.4f}]",
        f"ENGINE_CI_OVERLAP: {'TRUE' if ENGINE_CI_OVERLAP else 'FALSE'}",
        "",
        "## Primary discriminator (merge-math vs load_adapter; engine/precision/tokens/slice constant)",
        f"merged-Unsloth-bf16 vs unmerged-Unsloth-bf16 Wilson CIs "
        f"[{mu['lo']:.4f},{mu['hi']:.4f}] vs [{un['lo']:.4f},{un['hi']:.4f}]",
        f"CI_OVERLAP: {'TRUE' if CI_OVERLAP else 'FALSE'}",
        f"UNMERGED_RATE: {un['rate']:.4f}",
        f"UNMERGED_UPPER: {un['hi']:.4f}",
        "",
        f"DISCRIMINATOR: {verdict}",
        "",
        "## Rationale",
        rationale,
    ]
    open(os.path.join(DISC, "summary.md"), "w").write("\n".join(lines) + "\n")

    # ---- 04.3-03-RESULTS.md (EXACTLY ONE canonical DISCRIMINATOR line) ----
    rlines = [
        "# 04.3-03 Discriminator — RESULTS\n",
        f"DISCRIMINATOR: {verdict}\n",
        "## Per-arm terse rates (structural [/REASONING] split; bf16; Wilson 95%)",
        fmt_arm("merged-72-vLLM-bf16", "reasoning-merged", "vLLM", "bf16", vllm),
        fmt_arm("merged-72-Unsloth-bf16", "reasoning-merged", "Unsloth", "bf16", mu),
        fmt_arm("unmerged-72-Unsloth-bf16", "merged-v2+load_adapter(ckpt-72)", "Unsloth", "bf16", un),
        "",
        "## Canonical tokens",
        f"ENGINE_CI_OVERLAP: {'TRUE' if ENGINE_CI_OVERLAP else 'FALSE'}",
        f"CI_OVERLAP: {'TRUE' if CI_OVERLAP else 'FALSE'}",
        f"UNMERGED_RATE: {un['rate']:.4f}",
        f"UNMERGED_UPPER: {un['hi']:.4f}",
        "",
        "## Decision rationale (pre-registered bf16 thresholds)",
        rationale,
        "",
        "## Corrective branch + boundaries",
        "This plan SELECTS the corrective branch only; it does NOT launch the retrain or the "
        "merge-script rewrite. Steps 5-8 of 04.3-REOPEN-PLAN.md remain DEFERRED to the follow-up "
        "plan keyed off this verdict. ckpt-72 is NOT promoted; "
        "models/qwen3-30b-wp-30_70-merged-v2 remains the certified fallback; reasoning-merged + "
        "merged-v2 untouched.",
    ]
    open(RESULTS, "w").write("\n".join(rlines) + "\n")
    print(f"DISCRIMINATOR: {verdict}")
    print(f"ENGINE_CI_OVERLAP={ENGINE_CI_OVERLAP} CI_OVERLAP={CI_OVERLAP} "
          f"UNMERGED_RATE={un['rate']:.4f} UNMERGED_UPPER={un['hi']:.4f}")


if __name__ == "__main__":
    main()
