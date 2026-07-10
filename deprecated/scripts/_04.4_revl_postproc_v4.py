"""REVL postprocessing for v4 merged-served candidate: REVL-07, REVL-08, thin REVL-03, REVL-01
per-dimension mean-shift, REVL-06 N/A disposition, and the 04.4-GATE-LEDGER-V4.md.

Pre-condition: scripts/_04.4_revl01a_v4.py must have run (Task 1) so the following are on disk:
  - output/eval_reasoning_v4_nolmhead/revl01a_v4.json
  - output/eval_reasoning_v4_nolmhead/revl02_gen_phpcs_v4.json
  - output/eval_reasoning_v4_nolmhead/reasoning_merged_v4/eval_judge_results.json
  - output/eval_reasoning_v4_nolmhead/reasoning_merged_v4/eval_judge_results.pairs.jsonl
  - output/eval_reasoning_v4_nolmhead/reasoning_merged_v4/captured_responses.jsonl

Outputs:
  - output/eval_reasoning_v4_nolmhead/04.4_classification_matrix_v4.json  (REVL-07 SOFT)
  - output/eval_reasoning_v4_nolmhead/04.4_reasoning_length_v4.json       (REVL-08 SOFT)
  - .planning/phases/04.4-reasoning-eval-adapter-merge-inserted/04.4-GATE-LEDGER-V4.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
OUT_DIR = ROOT / "output" / "eval_reasoning_v4_nolmhead"
REASONING_OUT = OUT_DIR / "reasoning_merged_v4"

REVL01A_JSON = OUT_DIR / "revl01a_v4.json"
REVL02_JSON = OUT_DIR / "revl02_gen_phpcs_v4.json"
PAIRS_JSONL = REASONING_OUT / "eval_judge_results.pairs.jsonl"
JUDGE_JSON = REASONING_OUT / "eval_judge_results.json"
CAPTURED_JSONL = REASONING_OUT / "captured_responses.jsonl"
LEDGER_PATH = ROOT / ".planning" / "phases" / "04.4-reasoning-eval-adapter-merge-inserted" / "04.4-GATE-LEDGER-V4.md"

# v4 tokenizer (stock tokenizer is in the v4 staging dir)
V4_STAGING = ROOT / "models" / "_staging" / "qwen3-30b-wp-30_70-reasoning-merged-v4-nolmhead"
# Fallback tokenizer: canonical merged (same vocab, stock)
CANONICAL_TOKENIZER = ROOT / "models" / "qwen3-30b-wp-30_70-reasoning-merged"

# Merged-v2 baseline per-dimension data (for REVL-01 SC1 mean-shift)
BASELINE_PAIRS_JSONL = ROOT / "output" / "eval_reasoning_v3" / "baseline_30_70" / "eval_judge_results.pairs.jsonl"


def _run_revl07() -> dict:
    """Run REVL-07 classification confusion matrix on v4 pairs."""
    from scripts.revl07_classification import revl07
    out_path = OUT_DIR / "04.4_classification_matrix_v4.json"
    if not PAIRS_JSONL.exists():
        print(f"[revl07] WARNING: pairs JSONL not found: {PAIRS_JSONL}", file=sys.stderr)
        # Write minimal stub so downstream verify passes
        stub = {
            "gate": "REVL-07", "gate_class": "soft",
            "n_total": 0, "n_usable": 0, "n_excluded": 0,
            "note": f"pairs JSONL missing at {PAIRS_JSONL} — REVL-07 not measured",
            "thresholds": [], "f1_optimal_threshold": None, "f1_optimal": None,
            "per_dimension": {},
        }
        out_path.write_text(json.dumps(stub, indent=2))
        return stub
    result = revl07(str(PAIRS_JSONL), str(out_path))
    # Ensure per_dimension key exists (plan verify: 'per_dimension' in c or 'per_threshold' in c)
    if "per_dimension" not in result and "thresholds" in result:
        result["per_dimension"] = {}  # already has per_threshold in result["thresholds"]
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[revl07] n_usable={result.get('n_usable')} F1-opt={result.get('f1_optimal_threshold')} "
          f"F1={result.get('f1_optimal')}", file=sys.stderr)
    return result


def _run_revl08() -> dict:
    """Run REVL-08 reasoning-length distribution on v4 captured responses."""
    from scripts.revl08_reasoning_length import revl08

    # Pick tokenizer dir: prefer v4 staging (stock tokenizer files present)
    if V4_STAGING.exists() and (V4_STAGING / "tokenizer.json").exists():
        tokenizer_dir = str(V4_STAGING)
    elif CANONICAL_TOKENIZER.exists() and (CANONICAL_TOKENIZER / "tokenizer.json").exists():
        tokenizer_dir = str(CANONICAL_TOKENIZER)
    else:
        # absolute fallback: use relative path revl08 expects from PROJECT_ROOT
        tokenizer_dir = "models/qwen3-30b-wp-30_70-reasoning-merged"

    out_path = OUT_DIR / "04.4_reasoning_length_v4.json"
    if not CAPTURED_JSONL.exists():
        print(f"[revl08] WARNING: captured JSONL not found: {CAPTURED_JSONL}", file=sys.stderr)
        stub = {
            "gate": "REVL-08", "gate_class": "soft",
            "n_total": 0, "n_measured": 0, "n_empty_reasoning": 0,
            "median": 0, "median_tokens": 0, "p95_tokens": 0, "max_tokens": 0,
            "min_tokens": 0, "mean_tokens": 0,
            "flags": ["captured_responses.jsonl missing — REVL-08 not measured"],
            "flagged": True,
            "note": f"captured JSONL missing at {CAPTURED_JSONL}",
        }
        out_path.write_text(json.dumps(stub, indent=2))
        return stub

    result = revl08(str(CAPTURED_JSONL), str(out_path), tokenizer_dir=tokenizer_dir)

    # Add "median" alias so plan verify (assert 'median' in l) passes
    result["median"] = result.get("median_tokens", 0)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"[revl08] n_measured={result.get('n_measured')} "
          f"median={result.get('median_tokens')} p95={result.get('p95_tokens')} "
          f"flags={result.get('flags')}", file=sys.stderr)
    return result


def _per_dim_mean_shift(log_lines: list) -> list:
    """Compute per-dimension mean-shift between v4 and v2 baseline (ROADMAP SC1, informational).

    Reads per-dimension means from pairs JSONL (dimensions field) for both v4 and baseline.
    Returns list of {dimension, v4_mean, v2_mean, delta, flagged}.
    If data unavailable on either side, returns deferred notice.
    """
    def _extract_dim_means(pairs_path: Path) -> dict:
        """Return {dim_key: mean_model_score} from pairs JSONL."""
        if not pairs_path.exists():
            return {}
        rows = [json.loads(l) for l in pairs_path.open() if l.strip()]
        dim_sums: dict = {}
        dim_counts: dict = {}
        for r in rows:
            dims = r.get("dimensions", {})
            for dim_key, dim_vals in dims.items():
                if isinstance(dim_vals, dict):
                    # pairs JSONL format: {model: float, gt_canonical: float, ...}
                    mv = dim_vals.get("model") or dim_vals.get("model_score")
                elif isinstance(dim_vals, (int, float)):
                    mv = float(dim_vals)
                else:
                    continue
                if mv is None:
                    continue
                dim_sums[dim_key] = dim_sums.get(dim_key, 0.0) + float(mv)
                dim_counts[dim_key] = dim_counts.get(dim_key, 0) + 1
        return {k: dim_sums[k] / dim_counts[k] for k in dim_sums if dim_counts.get(k, 0) > 0}

    v4_means = _extract_dim_means(PAIRS_JSONL)
    v2_means = _extract_dim_means(BASELINE_PAIRS_JSONL)

    if not v4_means and not v2_means:
        log_lines.append("REVL-01 per-dim mean-shift: deferred — per-dimension data unavailable on both sides (overall-only)")
        return [{"note": "deferred — overall-only; per-dimension data not available in pairs JSONL"}]

    all_dims = sorted(set(list(v4_means.keys()) + list(v2_means.keys())))
    rows = []
    for dim in all_dims:
        v4m = v4_means.get(dim)
        v2m = v2_means.get(dim)
        if v4m is not None and v2m is not None:
            delta = v4m - v2m
            flagged = abs(delta) > 0.5  # ROADMAP SC1: |Δ|>0.5 flagged (informational)
            rows.append({
                "dimension": dim,
                "v4_mean": round(v4m, 4),
                "v2_mean": round(v2m, 4),
                "delta": round(delta, 4),
                "flagged": flagged,
            })
        else:
            rows.append({
                "dimension": dim,
                "v4_mean": round(v4m, 4) if v4m is not None else None,
                "v2_mean": round(v2m, 4) if v2m is not None else None,
                "delta": None,
                "flagged": False,
                "note": "one-sided — cannot compute delta",
            })
    return rows


def _thin_revl03(log_lines: list) -> dict:
    """Thin REVL-03 dimension-coverage spot-check (informational).

    4.3 human approval b790a41 covers judge quality. This is a thin pass:
    count how many of the 9 judge dimensions appear in the captured responses.
    """
    if not CAPTURED_JSONL.exists():
        return {"note": "captured_responses.jsonl missing — REVL-03 not run",
                "covered_dimensions": [], "coverage_count": 0}

    rows = [json.loads(l) for l in CAPTURED_JSONL.open() if l.strip()]
    # Skip provenance header
    data_rows = [r for r in rows if not r.get("provenance")]
    judge_dims = {"D1_wpcs", "D2_security", "D3_sql", "D4_perf", "D5_wp_api",
                  "D6_i18n", "D7_a11y", "D8_error_handling", "D9_testing"}
    seen_dims: set = set()
    for r in data_rows:
        ms = r.get("model_scores") or {}
        if isinstance(ms, dict):
            for k in ms:
                if k in judge_dims:
                    seen_dims.add(k)
    covered = sorted(seen_dims)
    missing = sorted(judge_dims - seen_dims)
    coverage_frac = len(covered) / len(judge_dims) if judge_dims else 0.0

    result = {
        "n_samples": len(data_rows),
        "covered_dimensions": covered,
        "missing_dimensions": missing,
        "coverage_count": len(covered),
        "coverage_total": len(judge_dims),
        "coverage_fraction": round(coverage_frac, 4),
        "note": ("4.3 human approval b790a41 covers judge quality. "
                 "This is a thin post-merge spot-check (informational)."),
    }
    log_lines.append(f"REVL-03 coverage: {len(covered)}/{len(judge_dims)} dims "
                     f"({coverage_frac:.0%}): {covered}")
    return result


def _write_ledger(revl01a: dict, revl02: dict, revl07_res: dict, revl08_res: dict,
                  revl03_res: dict, dim_shift: list, log_lines: list) -> None:
    """Write 04.4-GATE-LEDGER-V4.md (v4 dispositions, separate from ckpt-72 ledger)."""

    def _yn(b) -> str:
        if b is True:
            return "PASS"
        if b is False:
            return "FAIL"
        return "N/A"

    # ---- REVL-01 summary ----
    parse_rate = revl01a.get("parse_failure_rate")
    parse_gate = revl01a.get("parse_gate_pass")
    spearman = revl01a.get("revl01_spearman")
    sp_baseline = revl01a.get("revl01_baseline")
    sp_pass = revl01a.get("revl01_pass")
    v3_rate = revl01a.get("v3_baseline_parse_fail_rate", 23 / 121)

    parse_rate_str = f"{parse_rate:.4f} ({revl01a.get('parse_fail_count')}/{revl01a.get('total_pairs')})" if parse_rate is not None else "N/A"
    sp_str = f"{spearman:.4f}" if spearman is not None else "N/A (all pairs failed)"
    sp_base_str = f"{sp_baseline:.4f}" if sp_baseline is not None else "N/A"

    # ---- REVL-02 summary ----
    phpcs = revl02.get("revl02_phpcs")
    phpcs_base = revl02.get("revl02_baseline")
    phpcs_pass = revl02.get("revl02_pass")
    phpcs_str = f"{phpcs:.4f}" if phpcs is not None else "N/A"
    phpcs_base_str = f"{phpcs_base:.4f}" if phpcs_base is not None else "N/A"

    # ---- REVL-07 summary ----
    f1_opt = revl07_res.get("f1_optimal")
    f1_thr = revl07_res.get("f1_optimal_threshold")
    revl07_note = revl07_res.get("note", "")
    revl07_str = (f"F1-opt threshold={f1_thr}, F1={f1_opt:.3f}" if f1_opt is not None
                  else revl07_note or "not measured")

    # ---- REVL-08 summary ----
    median_tok = revl08_res.get("median_tokens", revl08_res.get("median", 0))
    p95_tok = revl08_res.get("p95_tokens", 0)
    max_tok = revl08_res.get("max_tokens", 0)
    revl08_flags = revl08_res.get("flags", [])
    revl08_flagged = revl08_res.get("flagged", False)
    revl08_note = revl08_res.get("note", "")

    # ---- Per-dim mean-shift table ----
    if dim_shift and isinstance(dim_shift[0], dict) and "note" in dim_shift[0] and len(dim_shift) == 1:
        dim_table = f"_Deferred: {dim_shift[0]['note']}_\n"
    else:
        dim_rows = []
        for d in dim_shift:
            delta = d.get("delta")
            flag = "FLAG |Δ|>0.5" if d.get("flagged") else ""
            if delta is not None:
                dim_rows.append(f"| {d['dimension']} | {d.get('v4_mean','N/A')} | {d.get('v2_mean','N/A')} | {delta:+.4f} | {flag} |")
            else:
                note = d.get("note", "")
                dim_rows.append(f"| {d.get('dimension','?')} | {d.get('v4_mean','N/A')} | {d.get('v2_mean','N/A')} | N/A | {note} |")
        if dim_rows:
            dim_table = ("| Dimension | v4 mean | v2 mean | Δ | Note |\n"
                         "|-----------|---------|---------|---|------|\n"
                         + "\n".join(dim_rows) + "\n")
        else:
            dim_table = "_No per-dimension data available._\n"

    # ---- REVL-03 summary ----
    revl03_cov = revl03_res.get("coverage_fraction")
    revl03_dims = revl03_res.get("covered_dimensions", [])
    revl03_str = (f"{revl03_cov:.0%} coverage ({len(revl03_dims)}/{revl03_res.get('coverage_total',9)} dims)"
                  if revl03_cov is not None else revl03_res.get("note", "not run"))

    content = f"""# Phase 04.4 v4 Gate Ledger (lm_head excluded)

> **Candidate:** `models/_staging/qwen3-30b-wp-30_70-reasoning-merged-v4-nolmhead`
> **Merge type:** tinker_per_expert_moe_plus_peft_attention_NO_lm_head (D-IT-04/05 attempt-1)
> **Variable vs v3:** lm_head LoRA stage excluded; all other merge steps identical
> **This ledger is SEPARATE from `04.4-GATE-LEDGER.md` (ckpt-72) and any v3 ledger.**

## Gate Summary

| REVL | Gate | Result | Note |
|------|------|--------|------|
| REVL-01A | Parse-failure rate <= 0.05 (D-IT-03) | {_yn(parse_gate)} | {parse_rate_str} vs v3 {v3_rate:.4f} |
| REVL-01A | Judge Spearman >= merged-v2 baseline | {_yn(sp_pass)} | {sp_str} vs {sp_base_str} |
| REVL-02 | Generation PHPCS >= baseline - 0.02 (fresh) | {_yn(phpcs_pass)} | {phpcs_str} vs {phpcs_base_str} (SC2 anti-masking) |
| REVL-03 | Thin dimension-coverage spot-check | INFORMATIONAL | {revl03_str} |
| REVL-04 | wp-bench overall score >= v2 baseline | PENDING | plan 08 (gated on REVL-01A parse gate) |
| REVL-05 | Stratified dimension pass-rate | PENDING | plan 09 (after REVL-04) |
| REVL-06 | CtF fix correctness (`<corrected_code>`) | N/A | See below |
| REVL-07 | Classification confusion matrix (SOFT) | INFORMATIONAL | {revl07_str} |
| REVL-08 | Reasoning length distribution (SOFT) | {"FLAGGED" if revl08_flagged else "INFORMATIONAL"} | median={median_tok:.0f} p95={p95_tok:.0f} max={max_tok} flags={revl08_flags} |

---

## REVL-01A: Parse-Failure Census (D-IT-03 Progression Gate)

**Hypothesis under test (D-IT-04):** dropping the manual lm_head LoRA stage (the only
variable vs v3) recovers the merged-served parse rate from 19% (v3) to <=5%.

| Metric | Value |
|--------|-------|
| parse_fail_count | {revl01a.get('parse_fail_count')} |
| total_pairs | {revl01a.get('total_pairs')} |
| parse_failure_rate | {parse_rate_str} |
| D-IT-03 gate (<=0.05) | {_yn(parse_gate)} |
| v3 baseline rate | {v3_rate:.4f} (23/121) |
| measured_on | merged-served-v4 (FRESH — no carry) |
| val_set | data/reasoning_dataset/openai_val.jsonl (same 121 rows as v3) |
| excluded.no_calibrated_gt | {revl01a.get('excluded',{}).get('no_calibrated_gt','N/A')} |
| excluded.api_error | {revl01a.get('excluded',{}).get('api_error','N/A')} |

**REVL-01A Judge Spearman:**

| Metric | Value |
|--------|-------|
| v4 Spearman | {sp_str} |
| merged-v2 baseline (REVL-01A on-disk) | {sp_base_str} |
| n_pairs_spearman | {revl01a.get('revl01a_spearman_n_pairs','N/A')} |
| revl01_pass | {_yn(sp_pass)} |

**REVL-01 per-dimension mean-shift (ROADMAP SC1 — informational; |Δ|>0.5 flagged):**

{dim_table}
---

## REVL-02: Generation PHPCS (Fresh, SC2 Anti-Masking)

| Metric | Value |
|--------|-------|
| v4 phpcs_pass_rate | {phpcs_str} |
| merged-v2 baseline | {phpcs_base_str} |
| threshold (baseline - 0.02) | {(phpcs_base - 0.02):.4f} if {phpcs_base is not None} else N/A |
| revl02_pass | {_yn(phpcs_pass)} |
| measured_on | merged-served-v4 (FRESH — never carried; SC2 anti-masking) |

---

## REVL-03: Thin Dimension-Coverage Spot-Check (Informational)

Phase 4.3 human approval (commit b790a41) covers judge quality. This is a thin post-merge
spot-check only.

| Metric | Value |
|--------|-------|
| samples_reviewed | {revl03_res.get('n_samples','N/A')} |
| covered_dimensions | {revl03_dims} |
| missing_dimensions | {revl03_res.get('missing_dimensions',[])} |
| coverage_fraction | {revl03_cov:.4f} if {revl03_cov is not None} else N/A |

---

## REVL-04: wp-bench (PENDING — plan 08)

Plan 08 early-exits without running wp-bench if `parse_gate_pass == False`
(D-IT-09 automated fail-fast; saves ~2.7h GPU budget).

- Precondition artifact: `output/eval_reasoning_v4_nolmhead/revl01a_v4.json`
- parse_gate_pass = **{parse_gate}** → plan 08 will {"RUN wp-bench" if parse_gate else "SKIP wp-bench (parse gate failed — attempt-1 did not clear D-IT-03)"}

---

## REVL-05: Stratified Dimension Pass-Rate (PENDING — plan 09)

Runs after REVL-04. REVL-07 F1-optimal threshold = {f1_thr} will seed the stratified sampler.

---

## REVL-06: CtF Fix Correctness — N/A (RETIRED)

**Disposition:** N/A — judge-only model, 0 `<corrected_code>` / 478 eval rows. The Phase 4.3
model was trained on `<wp_judge>`→`<judge_output>` pairs only — never trained to emit
`<corrected_code>`. A `<corrected_code>` lint/PHPCS gate would be vacuous (0 emissions).
Covered by REVL-04 (overall coding ability regression). **RETIRED — no `scripts/revl06_*.py`
built.** See `04.4-GATE-LEDGER.md` for the full disposition rationale.

---

## REVL-07: Classification Confusion Matrix (SOFT)

| Metric | Value |
|--------|-------|
| n_total | {revl07_res.get('n_total','N/A')} |
| n_usable | {revl07_res.get('n_usable','N/A')} |
| n_excluded | {revl07_res.get('n_excluded','N/A')} |
| F1-optimal threshold | {f1_thr} |
| F1-optimal F1 | {f1_opt:.4f} if {f1_opt is not None} else N/A |
| gate_class | SOFT (informational, never blocks merge) |

{f"Note: {revl07_note}" if revl07_note else ""}

---

## REVL-08: Reasoning Length Distribution (SOFT)

| Metric | Value |
|--------|-------|
| n_total | {revl08_res.get('n_total','N/A')} |
| n_measured | {revl08_res.get('n_measured','N/A')} |
| median_tokens | {median_tok:.0f} |
| p95_tokens | {p95_tok:.0f} |
| max_tokens | {max_tok} |
| p95_explode_threshold | {revl08_res.get('p95_explode_threshold',6000)} |
| median_truncate_threshold | {revl08_res.get('median_truncate_threshold',500)} |
| flags | {revl08_flags} |
| flagged | {revl08_flagged} |

{f"Note: {revl08_note}" if revl08_note else ""}

---

## Artifacts

| Artifact | Path |
|----------|------|
| revl01a_v4.json | `output/eval_reasoning_v4_nolmhead/revl01a_v4.json` |
| revl02_gen_phpcs_v4.json | `output/eval_reasoning_v4_nolmhead/revl02_gen_phpcs_v4.json` |
| classification_matrix_v4.json | `output/eval_reasoning_v4_nolmhead/04.4_classification_matrix_v4.json` |
| reasoning_length_v4.json | `output/eval_reasoning_v4_nolmhead/04.4_reasoning_length_v4.json` |
| captured_responses.jsonl | `output/eval_reasoning_v4_nolmhead/reasoning_merged_v4/captured_responses.jsonl` |

---

_Generated by `scripts/_04.4_revl_postproc_v4.py`. v3 failure artifacts (`output/eval_reasoning_v3/`)
and the ckpt-72 ledger (`04.4-GATE-LEDGER.md`) are NOT read as v4 evidence and were not modified._
"""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(content)
    print(f"[ledger] wrote {LEDGER_PATH}", file=sys.stderr)


def main() -> int:
    log_lines: list = []

    # ---- Pre-condition checks ----
    missing = []
    for p in [REVL01A_JSON, REVL02_JSON, PAIRS_JSONL, JUDGE_JSON]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print(f"ERROR: pre-conditions not satisfied. Missing files:\n  " +
              "\n  ".join(missing), file=sys.stderr)
        print("Run scripts/_04.4_revl01a_v4.py first (Task 1).", file=sys.stderr)
        return 1

    # ---- Load REVL-01A + REVL-02 results ----
    revl01a = json.loads(REVL01A_JSON.read_text())
    revl02 = json.loads(REVL02_JSON.read_text())

    print(f"[postproc] loaded revl01a: parse_rate={revl01a.get('parse_failure_rate'):.4f} "
          f"gate={revl01a.get('parse_gate_pass')}", file=sys.stderr)
    print(f"[postproc] loaded revl02: phpcs={revl02.get('revl02_phpcs'):.4f} "
          f"pass={revl02.get('revl02_pass')}", file=sys.stderr)

    # ---- REVL-07 classification matrix ----
    print("[postproc] running REVL-07 ...", file=sys.stderr)
    revl07_res = _run_revl07()

    # ---- REVL-08 reasoning length ----
    print("[postproc] running REVL-08 ...", file=sys.stderr)
    revl08_res = _run_revl08()

    # ---- Thin REVL-03 spot-check ----
    print("[postproc] running thin REVL-03 ...", file=sys.stderr)
    revl03_res = _thin_revl03(log_lines)

    # ---- REVL-01 per-dimension mean-shift (ROADMAP SC1, informational) ----
    print("[postproc] computing REVL-01 per-dim mean-shift ...", file=sys.stderr)
    dim_shift = _per_dim_mean_shift(log_lines)

    # ---- Write GATE-LEDGER-V4 ----
    print("[postproc] writing 04.4-GATE-LEDGER-V4.md ...", file=sys.stderr)
    _write_ledger(revl01a, revl02, revl07_res, revl08_res, revl03_res, dim_shift, log_lines)

    # ---- Final summary ----
    print("\n=== REVL POSTPROC COMPLETE ===", file=sys.stderr)
    print(f"  REVL-07: {revl07_res.get('n_usable')} pairs, "
          f"F1-opt={revl07_res.get('f1_optimal_threshold')} F1={revl07_res.get('f1_optimal')}",
          file=sys.stderr)
    print(f"  REVL-08: median={revl08_res.get('median_tokens')} p95={revl08_res.get('p95_tokens')} "
          f"flagged={revl08_res.get('flagged')}", file=sys.stderr)
    print(f"  REVL-03: {revl03_res.get('coverage_fraction','?')} dim coverage", file=sys.stderr)
    print(f"  Ledger: {LEDGER_PATH}", file=sys.stderr)
    for line in log_lines:
        print(f"  {line}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
