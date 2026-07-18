"""Gate-before-remove eval driver for Phase 13 pruning (PRUNE-03).

Turns any per-expert IMPORTANCE score array (AIMER weight-norms, or REAP
saliency once 13-05 runs it) into a keep-mask via the Phase-11 k-sweep
machinery (scripts.sieve_expert_mask_inference.build_ksweep_mask -- scores
replace routing counts 1:1, same top-k-union-protected contract), then runs
the Phase-11 vLLM masking + wp-bench + 3-seed judge eval against THIS
phase's regression bars.

HARD CONSTRAINT (13-CONTEXT #2): every method x ratio is evaluated via this
GATING MASK before any weight is physically removed (PRUNE-06 happens only
for the winning variant, after PRUNE-05 selection).

Regression bars are the vLLM-measured values pinned in
output/sieve/prune_set_for_phase13.json -- NEVER the Tinker-native numbers
(0.842 ensemble rho / 0.827 s1 rho are sampler-specific and must not appear
in bar logic; see 13-RESEARCH Common Pitfall 4):
    gen wp-bench   >= 0.4284  (0.4484 vLLM-full-arm minus 2pp)
    judge ens rho  >= 0.7555  (0.8075 vLLM-full-arm minus two-SE floor)
    judge parse    >= 0.95    (121-item val set)
    judge s1 rho   >= 0.7497  (pre-authorized single-seed fallback, recorded not gated)

No real GPU serving happens in this plan (13-02): --self-check and --dry-run
are CPU-verifiable; scripts.__main__ real-run path executes in 13-04/13-05.

Usage:
    # CPU-only, no GPU/model required:
    python -m scripts.prune_gated_eval --self-check
    python -m scripts.prune_gated_eval --dry-run --ratio 25

    # Real gate run (13-04/13-05, GPU required):
    python -m scripts.prune_gated_eval --method aimer --ratio 25 --axis gen \\
        --score-npy output/prune/aimer_scores_gen.npy
    python -m scripts.prune_gated_eval --method aimer --ratio 25 --axis judge \\
        --score-npy output/prune/aimer_scores_judge.npy
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sieve_expert_mask_inference import build_ksweep_mask, load_protected_mask  # noqa: E402

N_LAYERS = 48
N_EXPERTS = 128

# ratio (% compression) -> k (experts kept per layer): 25% keeps 96/128, 50% keeps
# 64/128, 75% keeps 32/128 (13-02-PLAN Task 1 action).
RATIO_TO_K = {25: 96, 50: 64, 75: 32}

# vLLM-measured regression bars (output/sieve/prune_set_for_phase13.json
# regression_bars block). Tinker-native 0.842/0.827 deliberately absent.
GEN_WPBENCH_FLOOR = 0.4284
JUDGE_ENS_RHO_FLOOR = 0.7555
JUDGE_PARSE_FLOOR = 0.95
JUDGE_S1_FALLBACK_BAR = 0.7497  # pre-authorized fallback, recorded alongside, not gating

PRUNE_SET_JSON = PROJECT_ROOT / "output/sieve/prune_set_for_phase13.json"
DEFAULT_PROTECTED_MASK = PROJECT_ROOT / "output/profiling/reasoning-merged-v4/protected_expert_mask.npy"

# Same checkpoints/paths as scripts/sieve_ksweep_run.py (Phase 11) -- unchanged.
GEN_MODEL = "models/qwen3-30b-wp-30_70-reasoning-merged-v4"
JUDGE_SEEDS = {
    "s0": "models/_staging/qwen3-30b-wp-v1.3-s0-merged",
    "s1": "models/_staging/qwen3-30b-wp-v1.3-merged",
    "s2": "models/_staging/qwen3-30b-wp-v1.3-s2-merged",
}
VAL_DATASET = "data/reasoning_dataset/openai_val.jsonl"
VAL_LABELS = "output/relabel/val_labels_v1.json"

PORT = 8021
GPU_MEM_UTIL = 0.55

MASK_DIR = PROJECT_ROOT / "output/prune/masks"
OUT_ROOT = PROJECT_ROOT / "output/prune/gated"

# Unmasked (k=full) 3-seed judge baseline captures from Phase 11's k-sweep --
# the reference for D2_security retention comparison (no new GPU spend to
# get a baseline: these captures already exist, same val set, max_tokens=2048).
BASELINE_JUDGE_CAPTURES = {
    seed: PROJECT_ROOT / f"output/sieve/ksweep/judge_kfull/{seed}/judge_responses.jsonl"
    for seed in ("s0", "s1", "s2")
}

DEFAULT_SCORE_NPY = {
    "aimer": {"gen": "output/prune/aimer_scores_gen.npy", "judge": "output/prune/aimer_scores_judge.npy"},
    "reap": {"gen": "output/prune/reap_scores_gen.npy", "judge": "output/prune/reap_scores_judge.npy"},
}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_protected_sha(protected_mask_path: str | Path) -> None:
    """Re-verify the protected mask sha256 against the pinned value BEFORE any serve.

    T-13-03 mitigation: a tampered/regenerated protected mask must never silently
    flow into a gate run.
    """
    pinned = json.loads(PRUNE_SET_JSON.read_text())["protected_experts"]["mask_npy_sha256"]
    actual = _sha256(protected_mask_path)
    assert actual == pinned, (
        f"protected mask sha256 mismatch: {protected_mask_path} = {actual}, "
        f"pinned in {PRUNE_SET_JSON.name} = {pinned}"
    )


def build_gated_mask(scores: np.ndarray, protected: np.ndarray, ratio: int) -> np.ndarray:
    """Score array + ratio -> [n_layers, n_experts] keep-mask, never dropping a protected expert.

    Reuses build_ksweep_mask unchanged (Pattern 1: any per-expert score replaces
    routing counts 1:1 -- top-k-by-score UNION protected, per layer).
    """
    k = RATIO_TO_K[ratio]
    kept = build_ksweep_mask(scores, protected, k)
    # Explicit re-assertion (belt-and-braces on top of build_ksweep_mask's own guarantee):
    # a keep-mask must NEVER drop a protected expert.
    assert np.all(np.logical_or(np.logical_not(protected), kept)), (
        "gated mask drops a protected expert -- must never happen"
    )
    return kept


def _reset_wpbench_grader() -> None:
    """Stop+remove stale wp-env-runtime-* grader containers before a wp-bench run.

    Phase 11 fix 8c4b167: wp-bench's docker grader reuses WordPress+MySQL
    containers across invocations, letting DB state accumulate and silently
    degrade the correctness sub-score run-to-run. Reset before every gen arm.
    """
    import subprocess
    names = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"],
                            capture_output=True, text=True).stdout.splitlines()
    stale = [n for n in names if n.startswith("wp-env-runtime-")]
    if stale:
        print(f"[prune-gated-reset] removing stale grader containers: {stale}", flush=True)
        subprocess.run(["docker", "rm", "-f", *stale], stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT, check=False)


def _set_mask_env(mask_path: Path | None) -> None:
    import os
    if mask_path is None:
        os.environ.pop("SIEVE_MASK_NPY", None)
    else:
        os.environ["SIEVE_MASK_NPY"] = str(mask_path)


def run_gen_gate(mask_path: Path, out_dir: Path) -> dict:
    """Serve gen checkpoint with the gated mask, run wp-bench (Phase 11 fix 8c4b167 reset first)."""
    from scripts.run_eval_reasoning import _wpbench_with_boot

    _reset_wpbench_grader()
    _set_mask_env(mask_path)
    try:
        res = _wpbench_with_boot(GEN_MODEL, "prune-gated-gen-vllm", "prune_gated_gen",
                                  GPU_MEM_UTIL, out_dir)
    finally:
        _set_mask_env(None)
    return res


def _capture_judge_seed(seed: str, mask_path: Path, out_dir: Path) -> Path:
    from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, stop_vllm
    from scripts.sieve_capture_judge_http import capture

    name = f"prune-gated-judge-{seed}-vllm"
    out = out_dir / seed / "judge_responses.jsonl"
    _set_mask_env(mask_path)
    try:
        boot_vllm(JUDGE_SEEDS[seed], name, PORT, GPU_MEM_UTIL)
        wait_healthy(PORT, name)
        # Phase 11 fix cd36a5e: judge captures need max_tokens 2048, not the 1024 default.
        capture(base_url=f"http://localhost:{PORT}/v1", model=None,
                dataset=str(PROJECT_ROOT / VAL_DATASET), out=str(out), max_tokens=2048)
    finally:
        _set_mask_env(None)
        stop_vllm(name)
    return out


def score_judge_gate(seed_captures: dict[str, Path]) -> dict:
    """3-seed median ensemble rho + s1 fallback rho + parse-rate vs val_labels_v1."""
    from eval.eval_judge import _derive_prose_overall
    from eval.output_parsers import load_dim_map, parse_judge_scores
    from scipy.stats import spearmanr

    dm = load_dim_map()
    dw = {k: v for k, v in dm["dimension_weights"].items() if not k.startswith("_")}
    rows = [json.loads(line) for line in open(PROJECT_ROOT / VAL_DATASET) if line.strip()]
    wj_rows = [i for i, r in enumerate(rows) if next(
        (m["content"] for m in r["messages"] if m["role"] == "user"), ""
    ).startswith("<wp_judge>")]
    labels = {k: v for k, v in json.load(open(PROJECT_ROOT / VAL_LABELS)).items()
              if k.startswith("val:")}

    def load_capture(path: Path) -> dict:
        scores = {}
        for line in open(path):
            r = json.loads(line)
            if "index" not in r:
                continue
            parsed = parse_judge_scores(r["response"], "auto")
            if not parsed or not parsed.get("dimension_scores"):
                continue
            o = (float(parsed["overall"]) if "overall" in parsed
                 else _derive_prose_overall(parsed["dimension_scores"], dw))
            scores[f"val:{wj_rows[r['index']]}"] = o
        return scores

    per_seed = {seed: load_capture(p) for seed, p in seed_captures.items()}

    ensemble = {}
    for key in labels:
        vals = [per_seed[s][key] for s in per_seed if key in per_seed[s]]
        if vals:
            ensemble[key] = float(np.median(vals))
    common = sorted(set(ensemble) & set(labels))
    ens_rho = (spearmanr([ensemble[k] for k in common],
                         [labels[k] for k in common]).statistic
               if len(common) > 2 else None)

    # Per-seed rho (13-04 Task 2 done criterion: judge validated on ALL 3
    # seeds, not s1 alone -- HARD CONSTRAINT 6). None if a seed parsed <3 items.
    per_seed_rho: dict[str, float | None] = {}
    for seed in per_seed:
        common_s = sorted(set(per_seed[seed]) & set(labels))
        per_seed_rho[seed] = (
            float(spearmanr([per_seed[seed][k] for k in common_s],
                            [labels[k] for k in common_s]).statistic)
            if len(common_s) > 2 else None
        )
    s1_rho = per_seed_rho.get("s1")

    parse_rate = len(ensemble) / len(labels) if labels else None
    return {
        "judge_ensemble_rho": ens_rho,
        "judge_s1_rho": s1_rho,
        "per_seed_rho": per_seed_rho,
        "parse_rate": parse_rate,
        "n_scored": len(common),
        "n_per_seed": {s: len(v) for s, v in per_seed.items()},
    }


def _d2_security_mean(seed_captures: dict[str, Path]) -> float | None:
    """Mean D2_security dimension score (0-10 raw scale, eval/dim_map.json) across
    the val set, ensembled per item (median across seeds with a parseable score).
    None if nothing parsed. Used for prune_selection.py's d2_security_retention/
    d2_security_baseline eligibility check (13-03 forward dependency)."""
    from eval.output_parsers import parse_judge_scores

    per_seed: dict[str, dict[int, float]] = {}
    for seed, path in seed_captures.items():
        path = Path(path)
        if not path.exists():
            continue
        scores: dict[int, float] = {}
        for line in open(path):
            r = json.loads(line)
            if "index" not in r:
                continue
            parsed = parse_judge_scores(r["response"], "auto")
            d2 = (parsed or {}).get("dimension_scores", {}).get("D2_security")
            if d2 is not None:
                scores[r["index"]] = float(d2)
        per_seed[seed] = scores

    all_indices: set[int] = set()
    for s in per_seed.values():
        all_indices |= set(s)
    ensemble = []
    for idx in all_indices:
        vals = [per_seed[s][idx] for s in per_seed if idx in per_seed[s]]
        if vals:
            ensemble.append(float(np.median(vals)))
    return float(np.mean(ensemble)) if ensemble else None


def _write_result(method: str, ratio: int, axis: str, record: dict) -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"{method}_{ratio}_{axis}.json"
    # default=.item(): numpy scalars leak into records (np.bool_ from
    # `np_float >= floor` comparisons, np.float64 from spearmanr.statistic)
    # and stdlib json rejects np.bool_ (not a bool subclass).
    out_path.write_text(json.dumps(record, indent=2, default=lambda o: o.item()))
    return out_path


def run_gate(method: str, ratio: int, axis: str, score_npy: str | Path,
             protected_mask_npy: str | Path) -> dict:
    """Build the gated mask, verify it, serve+eval, write result record. GPU required."""
    verify_protected_sha(protected_mask_npy)
    protected = load_protected_mask(protected_mask_npy)
    scores = np.load(score_npy)
    kept = build_gated_mask(scores, protected, ratio)

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = MASK_DIR / f"{method}_{axis}_k{RATIO_TO_K[ratio]}.npy"
    np.save(mask_path, kept)

    out_dir = OUT_ROOT / f"{method}_{ratio}_{axis}"
    protected_retained = bool(np.all((~protected) | kept))

    if axis == "gen":
        gen_res = run_gen_gate(mask_path, out_dir)
        wp_bench = gen_res.get("wpbench_score")
        record = {
            "method": method, "ratio": ratio, "axis": axis,
            "wp_bench": wp_bench,
            "wp_bench_detail": {k: gen_res.get(k) for k in ("scores", "ran", "error")},
            "protected_retained": protected_retained,
            "pass": bool(wp_bench is not None and wp_bench >= GEN_WPBENCH_FLOOR),
            "pass_gen_wp_bench": bool(wp_bench is not None and wp_bench >= GEN_WPBENCH_FLOOR),
            "bars_used": {"gen_wp_bench_floor": GEN_WPBENCH_FLOOR},
        }
    else:
        seed_captures = {seed: _capture_judge_seed(seed, mask_path, out_dir)
                         for seed in JUDGE_SEEDS}
        judge_res = score_judge_gate(seed_captures)
        ens_rho = judge_res["judge_ensemble_rho"]
        parse_rate = judge_res["parse_rate"]
        pass_rho = bool(ens_rho is not None and ens_rho >= JUDGE_ENS_RHO_FLOOR)
        pass_parse = bool(parse_rate is not None and parse_rate >= JUDGE_PARSE_FLOOR)
        record = {
            "method": method, "ratio": ratio, "axis": axis,
            "judge_ensemble_rho": ens_rho,
            "judge_s1_rho": judge_res["judge_s1_rho"],
            "per_seed_rho": judge_res["per_seed_rho"],
            "parse_rate": parse_rate,
            "n_scored": judge_res["n_scored"], "n_per_seed": judge_res["n_per_seed"],
            "protected_retained": protected_retained,
            "pass": pass_rho and pass_parse,
            "pass_judge_ensemble_rho": pass_rho,
            "pass_judge_parse_rate": pass_parse,
            "s1_fallback_bar": JUDGE_S1_FALLBACK_BAR,
            "bars_used": {"judge_ensemble_rho_floor": JUDGE_ENS_RHO_FLOOR,
                          "judge_parse_floor": JUDGE_PARSE_FLOOR},
        }

        # 13-03 forward dependency: prune_selection.load_variant_records() only
        # reads d2_security_retention/baseline from the "gen" or "d2" axis file,
        # never "judge" -- write a separate {method}_{ratio}_d2.json (per its
        # documented merge convention) so 13-06 selection can find it.
        d2_record = {
            "method": method, "ratio": ratio, "axis": "d2",
            "d2_security_retention": _d2_security_mean(seed_captures),
            "d2_security_baseline": _d2_security_mean(BASELINE_JUDGE_CAPTURES),
            "protected_retained": protected_retained,
        }
        d2_path = _write_result(method, ratio, "d2", d2_record)
        print(f"Wrote {d2_path}")

    out_path = _write_result(method, ratio, axis, record)
    print(f"Wrote {out_path}")
    return record


def _dry_run(ratio: int, method: str, score_npy_gen: str | Path, score_npy_judge: str | Path,
             protected_mask_npy: str | Path) -> None:
    """Build+validate both axis masks and print planned arms WITHOUT serving (CPU-only)."""
    verify_protected_sha(protected_mask_npy)
    protected = load_protected_mask(protected_mask_npy)
    k = RATIO_TO_K[ratio]
    print(f"=== DRY RUN: method={method} ratio={ratio}% (k={k}) ===")
    for axis, score_path in (("gen", score_npy_gen), ("judge", score_npy_judge)):
        score_path = Path(score_path)
        if not score_path.exists():
            print(f"  [{axis}] score array not found: {score_path} (skipping)")
            continue
        scores = np.load(score_path)
        kept = build_gated_mask(scores, protected, ratio)
        per_layer = kept.sum(axis=1)
        protected_retained = bool(np.all((~protected) | kept))
        print(f"  [{axis}] planned arm: serve masked ({'gen checkpoint' if axis == 'gen' else '3 judge seeds'}), "
              f"kept {int(kept.sum())} total experts, per-layer min/max={int(per_layer.min())}/{int(per_layer.max())}, "
              f"protected_retained={protected_retained}")
    print("=== DRY RUN complete (no GPU serving performed) ===")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["aimer", "reap"], default="aimer")
    ap.add_argument("--ratio", type=int, choices=[25, 50, 75], default=25)
    ap.add_argument("--axis", choices=["gen", "judge"], default=None,
                     help="Required for a real (non-dry-run) gate.")
    ap.add_argument("--score-npy", default=None,
                     help="Score array for --axis. Default: method's canonical output/prune path.")
    ap.add_argument("--score-npy-gen", default=None, help="Override gen score array (dry-run only).")
    ap.add_argument("--score-npy-judge", default=None, help="Override judge score array (dry-run only).")
    ap.add_argument("--protected-mask", default=str(DEFAULT_PROTECTED_MASK))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        gen_npy = args.score_npy_gen or DEFAULT_SCORE_NPY[args.method]["gen"]
        judge_npy = args.score_npy_judge or DEFAULT_SCORE_NPY[args.method]["judge"]
        _dry_run(args.ratio, args.method, gen_npy, judge_npy, args.protected_mask)
        return 0

    if not args.axis:
        ap.error("--axis {gen,judge} is required for a real gate run (or pass --dry-run)")
    score_npy = args.score_npy or DEFAULT_SCORE_NPY[args.method][args.axis]
    record = run_gate(args.method, args.ratio, args.axis, score_npy, args.protected_mask)
    # Infra failure (boot/eval error -> null primary metric) is NOT a measured
    # fail: exit nonzero so a chained driver stops instead of running further
    # arms against a gate record that never actually measured anything.
    primary = record.get("wp_bench") if args.axis == "gen" else record.get("judge_ensemble_rho")
    if primary is None:
        print(f"ERROR: {args.axis} gate produced null primary metric (infra failure)", flush=True)
        return 1
    return 0


def _self_check() -> None:
    """Assert-based self-check on a synthetic score array (no GPU, no real checkpoint)."""
    n_layers, n_experts = 4, N_EXPERTS
    rng = np.random.default_rng(0)
    scores = rng.random((n_layers, n_experts))
    protected = np.zeros((n_layers, n_experts), dtype=bool)
    # Plant a protected expert that would NOT survive top-k on score alone.
    scores[2, 5] = 0.0  # coldest possible score
    protected[2, 5] = True

    for ratio, expected_k in RATIO_TO_K.items():
        kept = build_gated_mask(scores, protected, ratio)
        assert kept.shape == (n_layers, n_experts)
        assert np.all(np.logical_or(np.logical_not(protected), kept)), (
            f"ratio={ratio}: protected expert dropped"
        )
        assert kept[2, 5] == True  # noqa: E712 -- the planted cold-but-protected expert must survive
        assert kept[0].sum() >= expected_k  # at least the k budget kept (more if protected overlap)

    # sha256 re-verification path: a tampered protected mask must be caught.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        real_mask_path = tmp_path / "protected.npy"
        np.save(real_mask_path, protected)
        fake_prune_set = tmp_path / "prune_set_for_phase13.json"
        fake_prune_set.write_text(json.dumps({
            "protected_experts": {"mask_npy_sha256": _sha256(real_mask_path)}
        }))
        global PRUNE_SET_JSON
        orig = PRUNE_SET_JSON
        PRUNE_SET_JSON = fake_prune_set
        try:
            verify_protected_sha(real_mask_path)  # must pass: sha matches
            # Now tamper the mask on disk and confirm the mismatch is caught.
            tampered = protected.copy()
            tampered[0, 0] = not tampered[0, 0]
            np.save(real_mask_path, tampered)
            raised = False
            try:
                verify_protected_sha(real_mask_path)
            except AssertionError:
                raised = True
            assert raised, "tampered protected mask must fail sha256 verification"
        finally:
            PRUNE_SET_JSON = orig

    # Regression-bar constants must be the vLLM-measured values, never Tinker-native.
    assert GEN_WPBENCH_FLOOR == 0.4284
    assert JUDGE_ENS_RHO_FLOOR == 0.7555
    assert JUDGE_PARSE_FLOOR == 0.95
    for tinker_native in (0.842, 0.827):
        assert tinker_native not in (GEN_WPBENCH_FLOOR, JUDGE_ENS_RHO_FLOOR, JUDGE_PARSE_FLOOR)

    print("self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
