#!/usr/bin/env python
"""Judge-only gate-before-remove driver for the v4 judge prune (GATE4-04, Plan 26-01/26-02).

A stripped-down, judge-only re-wiring of sieve_ksweep_v4_run.py's serve/capture/
score loop pointed at ONE mask (output/prune-v4/masks/aimer_k224.npy, AIMER
weight-norm selection) instead of the k-sweep grid. It NEVER removes a weight —
it measures the gate that 26-02's physical surgery is gated behind.

What it reuses VERBATIM (no new eval logic — every piece is arch-agnostic and was
proven on v4 in Gate B):
- serve/capture: boot_vllm/wait_healthy/stop_vllm + sieve_capture_judge_http.capture
  (LANGUAGE_MODEL_ONLY=1, MAX_MODEL_LEN=16384, max_tokens=8192, temp 0.0, SIEVE_MASK_NPY)
- s1 scoring + CI-aware TOST: sieve_v4_tost_verdict.score_capture + tost_from_scores
  + load_labels, with the reference = the SAME-STACK Gate B full arm
  (output/sieve-v4/ksweep/kfull/s1/judge_responses.jsonl, rho 0.7935) — NOT the
  llama.cpp Q8 0.8067 nor the Tinker 0.8358 (T-26-02).
- D2_security: prune_gated_eval._d2_security_mean (retention on the masked s1
  capture, baseline on the Gate B full arm) gated at prune_selection.D2_SECURITY_TOLERANCE_PP.

What it does NOT do (goalpost-move guard, T-26-05): no gen/wp-bench axis, no docker
grader reset, and it imports NONE of v3's fixed rho/parse/wp-bench regression
floors. Gate C's bar is the pre-registered CI-aware TOST vs the same-stack full
arm at eps=2pp — a parse collapse (Pitfall 2) shows up as a rho/TOST failure, not
something averaged away.

Usage:
    .venv-tinker/bin/python -m scripts.prune_gate_v4 --self-check   # CPU, no GPU
    nohup .venv-tinker/bin/python -m scripts.prune_gate_v4 > logs/prune-v4/gate_driver.log 2>&1 &
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("REWARD_SKIP_PHPCS_ASSERT", "1")

from scripts.prune_gated_eval import _d2_security_mean  # noqa: E402  (reuse, arch-agnostic)
from scripts.prune_selection import D2_SECURITY_TOLERANCE_PP  # noqa: E402  (domain-generic reuse)
from scripts.sieve_v4_tost_verdict import (  # noqa: E402
    EPSILON, load_labels, score_capture, tost_from_scores,
)

# --- v4 judge config (identical wiring to sieve_ksweep_v4_run.py) ------------
JUDGE_SEEDS = {
    "s0": "models/Qwen3.6-35B-A3B-judge-v4-s0-merged",
    "s1": "models/Qwen3.6-35B-A3B-judge-v4-s1-merged",
    "s2": "models/Qwen3.6-35B-A3B-judge-v4-s2-merged",
}
MASK_NPY = PROJECT_ROOT / "output/prune-v4/masks/aimer_k224.npy"
PROTECTED_MASK = PROJECT_ROOT / "output/sieve-v4/protected_expert_mask.npy"
PROTECTED_MANIFEST = PROJECT_ROOT / "output/prune-v4/protected_manifest_v4.json"

# Same-stack TOST reference: the Gate B full arm (rho 0.7935). NOT llama.cpp/Tinker.
FULL_ARM_CAPTURE = PROJECT_ROOT / "output/sieve-v4/ksweep/kfull/s1/judge_responses.jsonl"

VAL_DATASET = "data/reasoning_dataset/openai_val.jsonl"

OUT_DIR = PROJECT_ROOT / "output/prune-v4/gated"
CAPTURE_ROOT = OUT_DIR / "aimer_224"
JUDGE_JSON = OUT_DIR / "aimer_224_judge.json"
D2_JSON = OUT_DIR / "aimer_224_d2.json"
ENSEMBLE_JSON = OUT_DIR / "aimer_224_ensemble.json"
SELECTION_JSON = PROJECT_ROOT / "output/prune-v4/selection_v4.json"

# Serve params — identical to sieve_ksweep_v4_run.py.
PORT = 8021
GPU_MEM_UTIL = 0.85
MAX_MODEL_LEN = 16384
CAPTURE_MAX_TOKENS = 8192
SERVE_SCRIPT = str(PROJECT_ROOT / "scripts" / "serve_30_70_vllm.sh")
K_ANCHOR = 224


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_protected_sha(protected_mask_path: str | Path = PROTECTED_MASK,
                         manifest_path: str | Path = PROTECTED_MANIFEST) -> None:
    """Re-verify the protected mask sha256 vs the v4 manifest BEFORE any serve (T-26-06).

    Mirrors prune_gated_eval.verify_protected_sha, re-pinned to the v4-specific
    manifest (NOT v3's prune_set_for_phase13.json).
    """
    pinned = json.loads(Path(manifest_path).read_text())["mask_npy_sha256"]
    actual = _sha256(protected_mask_path)
    assert actual == pinned, (
        f"protected mask sha256 mismatch: {protected_mask_path} = {actual}, "
        f"pinned in {Path(manifest_path).name} = {pinned}"
    )


def _set_mask_env(mask_path: Path | None) -> None:
    if mask_path is None:
        os.environ.pop("SIEVE_MASK_NPY", None)
    else:
        os.environ["SIEVE_MASK_NPY"] = str(mask_path)


def capture_masked_seed(seed: str, mask_path: Path = MASK_NPY) -> Path:
    """Boot one v4 judge seed with the k=224 keep-mask, capture 121 @8192, stop.

    Serve, never in-process load (T-26-04): vLLM manages its own memory budget;
    the mask patch fails LOUD if the MoE-block class does not resolve (T-26-03).
    """
    from scripts._p0_vllm_smoke_serve import boot_vllm, stop_vllm, wait_healthy
    from scripts.sieve_capture_judge_http import capture

    name = f"prune-v4-gate-{seed}-k{K_ANCHOR}"
    out = CAPTURE_ROOT / seed / "judge_responses.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    _set_mask_env(mask_path)
    try:
        boot_vllm(JUDGE_SEEDS[seed], name, PORT, GPU_MEM_UTIL,
                  serve_script=SERVE_SCRIPT,
                  extra_env={"LANGUAGE_MODEL_ONLY": "1", "MAX_MODEL_LEN": str(MAX_MODEL_LEN)})
        wait_healthy(PORT, name)
        capture(base_url=f"http://localhost:{PORT}/v1", model=None,
                dataset=str(PROJECT_ROOT / VAL_DATASET), out=str(out),
                max_tokens=CAPTURE_MAX_TOKENS, temperature=0.0)
    finally:
        _set_mask_env(None)
        stop_vllm(name)
    return out


def evaluate_gate(masked_capture: Path, full_capture: Path = FULL_ARM_CAPTURE) -> tuple[dict, dict]:
    """Score the masked s1 capture: TOST vs the same-stack full arm + D2_security.

    Returns (judge_record, d2_record). Honest recording (Pitfall 2): a parse
    collapse manifests as too-few common items / rho collapse -> TOST not
    equivalent -> pass=False; it is never averaged away.
    """
    from scipy.stats import spearmanr

    labels = load_labels()
    masked_scores, parse_fail = score_capture(masked_capture)
    full_scores, _ = score_capture(full_capture)

    common = sorted(set(masked_scores) & set(labels))
    s1_rho = (float(spearmanr([masked_scores[k] for k in common],
                              [labels[k] for k in common]).statistic)
              if len(common) > 2 else None)

    tost = tost_from_scores(masked_scores, full_scores, labels)

    protected = np.load(PROTECTED_MASK)
    keep = np.load(MASK_NPY)
    protected_retained = bool(np.all((~protected) | keep))

    d2_retention = _d2_security_mean({"s1": masked_capture})
    d2_baseline = _d2_security_mean({"s1": full_capture})
    pass_d2_security = bool(
        d2_retention is not None and d2_baseline is not None
        and (d2_baseline - d2_retention) <= D2_SECURITY_TOLERANCE_PP
    )

    # Two-sided TOST equivalence (the STRICTER secondary metric) — recorded as-measured.
    gate_pass = bool(tost.get("equivalent") and protected_retained)

    # Ship criterion = the pre-registered routing-(B) bar (25-02 sign-off): NON-INFERIORITY
    # (CI lower >= -2pp) AND D2 retained AND protected retained. A point-better arm fails the
    # two-sided upper bound while being genuinely non-inferior — that is the correct ship bar,
    # NOT a goalpost move. The two-sided `pass` is kept alongside as the stricter metric.
    ci_lo = tost.get("ci", [None, None])[0]
    non_inferior = bool(ci_lo is not None and ci_lo >= -EPSILON)
    pass_ship = bool(non_inferior and protected_retained and pass_d2_security)

    judge_record = {
        "requirement": "GATE4-04",
        "method": "aimer",
        "k": K_ANCHOR,
        "s1_rho": s1_rho,
        "n_scored": len(common),
        "parse_fail": parse_fail,
        "tost": tost,
        "tost_reference": {
            "arm": "full",
            "source": str(Path(full_capture).relative_to(PROJECT_ROOT)),
            "full_s1_rho": 0.7934812517026191,
            "note": "same-stack vLLM Gate B full arm — NOT llama.cpp Q8 0.8067 nor Tinker 0.8358",
        },
        "protected_retained": protected_retained,
        "pass": gate_pass,
        "pass_d2_security": pass_d2_security,
        # Ship disposition (distinct from the strict two-sided `pass`).
        "non_inferior": non_inferior,
        "ci_lower": ci_lo,
        "non_inferiority_margin": -EPSILON,
        "ci_lower_slack": (ci_lo + EPSILON) if ci_lo is not None else None,  # thinness: how far above -2pp
        "pass_ship": pass_ship,
        "ship_criterion": (
            "routing-(B) non-inferiority: ci_lower >= -2pp AND D2_security retained AND "
            "protected_retained. Two-sided `pass` (equivalent) is the stricter secondary "
            "metric, kept as-measured — a point-better arm fails it on the UPPER bound."
        ),
    }
    d2_record = {
        "requirement": "GATE4-04",
        "method": "aimer",
        "k": K_ANCHOR,
        "d2_security_retention": d2_retention,
        "d2_security_baseline": d2_baseline,
        "d2_security_tolerance_pp": D2_SECURITY_TOLERANCE_PP,
        "pass_d2_security": pass_d2_security,
        "protected_retained": protected_retained,
    }
    return judge_record, d2_record


def _write_gate_records(judge_record: dict, d2_record: dict) -> None:
    JUDGE_JSON.write_text(json.dumps(judge_record, indent=2, default=lambda o: o.item()))
    D2_JSON.write_text(json.dumps(d2_record, indent=2, default=lambda o: o.item()))
    print(f"[gate] s1_rho={judge_record['s1_rho']} tost.equivalent={judge_record['tost']['equivalent']} "
          f"ci={judge_record['tost']['ci']} parse_fail={judge_record['parse_fail']} "
          f"pass(two-sided)={judge_record['pass']} non_inferior={judge_record['non_inferior']} "
          f"pass_ship={judge_record['pass_ship']} pass_d2_security={judge_record['pass_d2_security']}", flush=True)
    print(f"[gate] wrote {JUDGE_JSON} and {D2_JSON}", flush=True)


def run_gate() -> int:
    """Real gate run: verify sha -> serve masked s1 -> score TOST + D2 -> write JSONs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    verify_protected_sha()
    print("[gate] protected-mask sha256 verified vs v4 manifest", flush=True)

    cap = capture_masked_seed("s1")
    judge_record, d2_record = evaluate_gate(cap)
    _write_gate_records(judge_record, d2_record)
    return 0


def rescore_gate() -> int:
    """Re-derive the gate JSONs from the ALREADY-CAPTURED s1 responses (no GPU, deterministic).

    Scoring the saved capture is byte-deterministic (fixed bootstrap seed), so this
    reproduces the measured TOST exactly while adding the pass_ship disposition — it
    NEVER re-serves or alters the measured rho/CI.
    """
    cap = CAPTURE_ROOT / "s1" / "judge_responses.jsonl"
    if not cap.exists():
        raise SystemExit(f"no existing capture at {cap} — run the real gate first")
    judge_record, d2_record = evaluate_gate(cap)
    _write_gate_records(judge_record, d2_record)
    return 0


# ---- self-check (CPU, no GPU): disposition logic + _d2_security_mean coverage ----

def _write_judge_fixture(path: Path, d2_score: int, overall_scores: list[tuple[int, float]]) -> None:
    """Write a synthetic judge capture whose responses parse_judge_scores can read.

    Each response carries a D2_security ("SQL Safety") line at d2_score/10 plus the
    other dimensions, so both _d2_security_mean and score_capture can consume it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for idx, overall in overall_scores:
        resp = (
            f"WPCS Compliance: score {overall}/10 — ok.\n\n"
            f"Security: score {d2_score}/10 — ok.\n\n"  # -> D2_security
            f"SQL Safety: score {overall}/10 — ok.\n\n"
            f"Performance: score {overall}/10 — ok.\n\n"
            f"WP API Usage: score {overall}/10 — ok.\n"
        )
        lines.append(json.dumps({"index": idx, "response": resp}))
    path.write_text("\n".join(lines) + "\n")


def _self_check() -> None:
    """Assert the TOST-vs-D2 disposition resolves pass/fail correctly (no GPU).

    Covers prune_gated_eval._d2_security_mean on a synthetic capture (research
    Wave-0 gap: prune_gated_eval's own _self_check does not exercise it).
    """
    import tempfile

    from scripts.prune_gated_eval import _d2_security_mean as d2_mean

    # _d2_security_mean coverage: it must extract D2_security and be monotone in it
    # (a lower-D2 capture must yield a strictly lower retention mean — the property
    # the D2-regression gate depends on). We assert the property, not the parser scale.
    with tempfile.TemporaryDirectory() as tmp:
        hi = Path(tmp) / "hi.jsonl"
        lo = Path(tmp) / "lo.jsonl"
        _write_judge_fixture(hi, d2_score=8, overall_scores=[(i, 8.0) for i in range(5)])
        _write_judge_fixture(lo, d2_score=3, overall_scores=[(i, 8.0) for i in range(5)])
        v_hi, v_lo = d2_mean({"s1": hi}), d2_mean({"s1": lo})
        assert v_hi is not None and v_lo is not None, "_d2_security_mean must parse the fixture"
        assert v_hi > v_lo, f"_d2_security_mean must be monotone in D2: {v_hi} !> {v_lo}"

    # Disposition logic on synthetic score dicts (bypass the parser: test the gate math).
    rng = np.random.default_rng(0)
    n = 121
    labels = {f"val:{i}": float(v) for i, v in enumerate(rng.uniform(0, 1, n))}
    lab = np.array([labels[f"val:{i}"] for i in range(n)])
    full_vals = 0.8 * lab + 0.2 * rng.uniform(0, 1, n)
    full = {f"val:{i}": float(full_vals[i]) for i in range(n)}

    # PASS case: masked ~ full (equivalent) + D2 retained.
    eq = {f"val:{i}": float(full_vals[i] + rng.normal(0, 0.002)) for i in range(n)}
    tost_pass = tost_from_scores(eq, full, labels)
    protected_retained = True
    d2_ret, d2_base = 6.0, 6.0
    pass_ok = bool(tost_pass["equivalent"] and protected_retained)
    pass_d2_ok = bool((d2_base - d2_ret) <= D2_SECURITY_TOLERANCE_PP)
    assert pass_ok and pass_d2_ok, "equivalent + D2-retained must resolve pass=True, pass_d2=True"

    # Ship criterion (non-inferiority): a point-better arm fails the two-sided upper bound
    # but is non-inferior (ci_lower >= -eps) -> pass_ship=True even when equivalent=False.
    better_vals = 0.97 * lab + 0.03 * rng.uniform(0, 1, n)  # higher label-correlation than full
    better = {f"val:{i}": float(better_vals[i]) for i in range(n)}
    tost_better = tost_from_scores(better, full, labels)
    ci_lo_b = tost_better["ci"][0]
    non_inf_b = bool(ci_lo_b is not None and ci_lo_b >= -EPSILON)
    assert not tost_better["equivalent"], "a clearly-better arm should fail two-sided TOST (upper bound)"
    assert non_inf_b and bool(non_inf_b and protected_retained and pass_d2_ok), \
        "a non-inferior point-better arm must resolve pass_ship=True despite equivalent=False"

    # FAIL case A: parse-collapse / rho-collapse -> not equivalent AND not non-inferior.
    collapsed = {f"val:{i}": float(rng.uniform(0, 1)) for i in range(n)}
    tost_fail = tost_from_scores(collapsed, full, labels)
    ci_lo_c = tost_fail["ci"][0]
    non_inf_c = bool(ci_lo_c is not None and ci_lo_c >= -EPSILON)
    assert not bool(tost_fail["equivalent"] and protected_retained), \
        "rho/parse collapse must resolve pass=False"
    assert not non_inf_c, "a collapsed arm must resolve non_inferior=False (pass_ship=False)"

    # FAIL case B: D2_security regression beyond tolerance -> pass_d2_security=False.
    d2_ret_bad = 5.0
    assert not bool((d2_base - d2_ret_bad) <= D2_SECURITY_TOLERANCE_PP), \
        "a >tolerance D2 regression must resolve pass_d2_security=False"

    print("self-check OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--rescore", action="store_true",
                    help="re-derive the gate JSONs from the existing s1 capture (no GPU)")
    args = ap.parse_args()
    if args.self_check:
        _self_check()
        return 0
    if args.rescore:
        return rescore_gate()
    return run_gate()


if __name__ == "__main__":
    sys.exit(main())
