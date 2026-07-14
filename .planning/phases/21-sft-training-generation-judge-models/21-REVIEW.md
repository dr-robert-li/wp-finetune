---
phase: 21-sft-training-generation-judge-models
reviewed: 2026-07-14T12:11:00Z
depth: standard
files_reviewed: 14
files_reviewed_list:
  - scripts/build_base21_moe_probe_adapter.py
  - scripts/build_gen03_merge.py
  - scripts/build_gen03_wpbench.py
  - scripts/build_judge03_capture_rho.py
  - scripts/build_judge03_merge_serve.py
  - scripts/capture_judge_responses_tinker.py
  - scripts/merge_adapter.py
  - scripts/serve_base20_vllm.sh
  - scripts/smoke_judge_format_base21.py
  - scripts/tinker_reasoning_data_v4.py
  - scripts/tinker_reasoning_sft_v4.py
  - scripts/verify_moe_merge_ground_truth.py
  - tests/test_tinker_reasoning_data_v4.py
  - tests/test_merge_adapter_moe_routing.py
findings:
  critical: 3
  warning: 6
  info: 1
  total: 10
status: issues_found
---

# Phase 21: Code Review Report

**Reviewed:** 2026-07-14T12:11:00Z
**Depth:** standard
**Files Reviewed:** 14
**Status:** issues_found

## Summary

Reviewed the Phase 21 SFT/merge/eval pipeline (GEN-03/JUDGE-03 merge+serve+bench
scripts, the routed-MoE-expert merge path in `merge_adapter.py`, the v4 data/SFT
drivers, and the two Wave-0 unit tests). The routed-expert merge itself
(`_merge_via_tinker_cookbook`, its completeness guard, and the vendor mapping
pinned by `tests/test_merge_adapter_moe_routing.py`) is sound and matches the
ground-truth verification script's own claims.

Three BLOCKER-level issues were found, all in the "second serve" / merge-diff
helper pattern that is copy-pasted across `build_base21_moe_probe_adapter.py`,
`build_gen03_merge.py`, and `build_judge03_merge_serve.py`:

1. `boot_vllm()` is called outside the `try/finally` that owns `stop_vllm()` in
   several helpers, risking an orphaned GPU-resident container if boot itself
   fails partway (violates the "sole-GB10-residency" discipline these same
   files repeatedly assert as a design invariant).
2. The base-vs-merged "differs" check treats an **empty** merged-model output
   as trivially "different" from the base output, so a broken/corrupted merge
   that generates nothing can vacuously pass the merge-verification gate.
3. `merge_adapter.py`'s idempotency short-circuit does not verify that an
   on-disk merged model actually corresponds to the *current* adapter being
   merged — re-running a Phase 21 script after a new checkpoint promotion (all
   of which write to fixed, non-adapter-scoped `MERGED_MODEL_DIR` paths) can
   silently skip re-merging and report metrics that don't match the
   `promoted_sampler_path` recorded in the same receipt.

Additional warnings cover CI reproducibility (unseeded bootstrap RNG), a
hardcoded fail-receipt path that collides across phases, silent conflation of
infra errors with genuine judge-format parse failures, and a point-estimate
(non-CI-aware) seed-promotion criterion ahead of real Tinker spend.

## Critical Issues

### CR-01: `boot_vllm()` called outside the `try/finally` that tears down the container

**File:** `scripts/build_base21_moe_probe_adapter.py:264-276`
**Also present in:**
- `scripts/build_gen03_merge.py:140-152`
- `scripts/build_judge03_merge_serve.py:146-159` (`_run_base_vs_merged_diff`)
- `scripts/build_judge03_merge_serve.py:182-199` (`_capture_and_score`)
- `scripts/verify_moe_merge_ground_truth.py:116-138` (`run_vllm_side`)

**Issue:** In each of these helpers the pattern is:
```python
boot_vllm(model_dir, container, port, 0.80, serve_script=..., extra_env=...)
try:
    served = wait_healthy(...)
    ...
finally:
    stop_vllm(container)
```
`boot_vllm()` sits *before* the `try:` block. If `boot_vllm()` raises after it
has already started the docker container (e.g. `docker run -d` succeeds but an
internal readiness/timeout check inside `boot_vllm` then raises), `stop_vllm()`
is never reached and the container is left running on the GPU with no
teardown — exactly the residency leak these files' own comments warn against
("sole-GB10-residency discipline: each serve fully tears down before the next
boots"). Contrast with the correct pattern already used elsewhere in this same
phase: `scripts/build_gen03_wpbench.py:71-84` (`_run_wpbench_on`) puts
`boot_vllm()` *inside* the `try`, and `scripts/smoke_judge_format_base21.py`
achieves the same effect by wrapping the whole `run_smoke()` call (which
itself calls `boot_vllm`) in `main()`'s `try/finally`.

**Fix:** Move `boot_vllm(...)` inside the `try:` block in every listed
location, matching `build_gen03_wpbench.py`'s pattern:
```python
def _serve(model_dir: str, container: str, allow_empty: bool) -> str:
    try:
        boot_vllm(model_dir, container, port, 0.80,
                  serve_script=serve_script, extra_env={"LANGUAGE_MODEL_ONLY": "1"})
        served = wait_healthy(port, container, timeout=1200)
        out = generate(port, served, [...], max_tokens=64)
        text = (out[0] or "").strip()
        if not text and not allow_empty:
            raise RuntimeError(...)
        return text
    finally:
        stop_vllm(container)
```

---

### CR-02: `base_vs_merged_differs` is vacuously True when the merged model returns empty output

**File:** `scripts/build_base21_moe_probe_adapter.py:263-286`
**Also present in:**
- `scripts/build_gen03_merge.py:139-162`
- `scripts/build_judge03_merge_serve.py:142-169`

**Issue:** Each of these `_run_base_vs_merged_diff`/`run_base_vs_merged_diff`
functions serves the merged model with `allow_empty=True` and the base model
with `allow_empty=False`, then returns `merged_out != base_out`:
```python
merged_out = _serve(MERGED_MODEL_DIR, "...-merged", allow_empty=True)
base_out = _serve("models/Qwen3.6-35B-A3B", "...-base", allow_empty=False)
return merged_out != base_out
```
If the merged model is broken (e.g. a corrupted merge, a config mismatch, or
a load failure that manifests as immediate-EOS generation) it can legitimately
produce `""`. Since `allow_empty=True` prevents that case from raising, and
`"" != base_out` is `True` for any non-empty base output, a completely broken
merge **passes** the "differs from base" check purely because empty text is
trivially "different." This check feeds directly into fail-closed gates
downstream — e.g. `build_gen03_wpbench.py:162-165` refuses to bench unless
`merge_receipt.get("base_vs_merged_differs")` is truthy, and
`build_judge03_merge_serve.py:243-246` raises only if
`not (merge_ok and base_vs_merged_differs)` — so a broken merge with empty
output slips past both gates into expensive downstream wp-bench/judge-capture
runs, corrupting the very measurement-integrity guarantee these scripts are
designed to provide.

**Fix:** Treat empty merged output as an explicit failure signal rather than
folding it into the equality comparison, e.g.:
```python
merged_out = _serve(MERGED_MODEL_DIR, "...-merged", allow_empty=True)
base_out = _serve("models/Qwen3.6-35B-A3B", "...-base", allow_empty=False)
if not merged_out:
    return False  # empty output is never valid evidence the merge "worked"
return merged_out != base_out
```

---

### CR-03: `merge_adapter.py`'s idempotency short-circuit can silently serve a stale merge for a new promoted checkpoint

**File:** `scripts/merge_adapter.py:517-549`

**Issue:** `merge_adapter()` begins:
```python
merged_path = Path(output_dir)
if merged_path.exists() and (merged_path / "config.json").exists():
    ...
    if not check_special_tokens:
        print(f"Merged model already exists at {merged_path} ... Skipping.")
        return
    ...
```
This check only inspects whether *a* merged model with a valid config/tokenizer
already exists at `output_dir` — it never checks whether that merged model
corresponds to the *adapter_dir* being passed in for *this* invocation. Every
Phase 21 caller uses a **fixed, non-adapter-scoped** `output_dir`
(`models/Qwen3.6-35B-A3B-gen-v4-merged` in `build_gen03_merge.py:47`,
`models/Qwen3.6-35B-A3B-judge-v4-s{seed}-merged` in
`build_judge03_merge_serve.py:69`, `models/Qwen3.6-35B-A3B-base21-moe-probe-merged`
in `build_base21_moe_probe_adapter.py:44`). If any of these scripts is re-run
after a new checkpoint is promoted (e.g. GEN-02 retrains and promotes a new
`wp-gen-v4-*` sampler checkpoint), and a merged model from a *previous* run
still sits at that fixed path with a valid `config.json`, `merge_adapter.py`
will print "Skipping" and exit 0 **without re-merging**. The calling script's
`_run_merge()` sees returncode 0 and proceeds to read
`GUARD_RECEIPT_PATH.read_text()` — a receipt that is either (a) stale, from
the earlier merge, or (b) missing entirely (crashing with
`FileNotFoundError` if it wasn't written this run). In case (a), the produced
`gen03_merge.json`/`judge03_rho.json` receipt would report the **new**
`promoted_sampler_path` (fetched fresh at the top of `main()`) alongside
**stale** `merged_target_module_count`/`merged_dir` data from the old adapter
— a genuine silent measurement-integrity violation: the receipt claims to
describe the newly-promoted checkpoint but the on-disk model and its guard
numbers belong to a different, earlier adapter. This risk is compounded by
`run_merge()`'s 3600s subprocess timeout (`build_gen03_merge.py:116`,
`build_judge03_merge_serve.py:128`): a killed-mid-merge process can leave a
directory with a valid `config.json` (raw-copied early by `build_hf_model`)
but incomplete/corrupt shards, which this same short-circuit would then treat
as "already exists" on the next run.

**Fix:** Scope the idempotency check to the adapter actually being merged —
e.g. write a content marker (adapter checkpoint's `sampler_path` or a hash of
`adapter_model.safetensors`) into the merged directory's own metadata (or into
`guard_receipt_path`) at merge time, and compare it against the *current*
call's adapter before short-circuiting:
```python
marker_path = merged_path / ".merged_from.json"
if merged_path.exists() and marker_path.exists():
    prior = json.loads(marker_path.read_text())
    if prior.get("adapter_dir") == str(adapter_dir):
        ...existing tokenizer-check-then-skip logic...
```
At minimum, Phase 21 callers should pass a unique, adapter-scoped
`--output-dir` per promoted checkpoint (e.g. suffix with the sampler
checkpoint name) instead of a fixed path, so a stale directory can never be
mistaken for a fresh one.

## Warnings

### WR-01: Bootstrap CI in `build_gen03_wpbench.py` uses an unseeded RNG

**File:** `scripts/build_gen03_wpbench.py:128`
**Issue:** `_bootstrap_ci_lower` creates `rng = np.random.default_rng()` with
no seed, while every other measurement parameter in this script (`seed: 1337`
for generation) is explicitly recorded for reproducibility. Given identical
`results_json`, `ci_lower`/`ci_upper` will differ slightly on every re-run of
this function, undermining the "reproducibility" goal this phase's receipts
are meant to satisfy — particularly relevant since the floor-pass decision
(`pass_ = ci["ci_lower"] >= floor`) is a borderline binary gate that a few
bootstrap-resample percentile points could flip.
**Fix:** Seed the generator, e.g. `rng = np.random.default_rng(1337)`, and
record the seed in the output receipt.

### WR-02: Quality dimension held fixed (not resampled) inside the bootstrap

**File:** `scripts/build_gen03_wpbench.py:118-136`
**Issue:** `quality_mean` is computed once from the real data and then reused
unchanged for every one of the 1000 bootstrap resamples (`quality_mean,  #
quality held fixed`). If wp-bench ever populates per-test `quality` scores for
this run, the resulting CI understates the true sampling uncertainty of the
weighted "overall" statistic, since one of its three weighted components
(0.3 weight) never varies across resamples.
**Fix:** Resample `quality` the same way `knowledge`/`correctness` are
resampled (`rng.choice(quality, size=quality.size, replace=True)`) whenever
`quality.size > 0`, so all three CI-contributing dimensions share the same
resampling discipline.

### WR-03: Generic-exception fail receipt in `merge_adapter.py` ignores the caller-specific receipt path

**File:** `scripts/merge_adapter.py:792-807`
**Issue:** The `__main__` exception handler always writes:
```python
receipt_path = PROJECT_ROOT / "output" / "base20" / "_merge_adapter_result.json"
```
regardless of which phase invoked the script or what `--guard-receipt-path`
was passed. Every Phase 21 caller (`build_gen03_merge.py`,
`build_judge03_merge_serve.py`, `build_base21_moe_probe_adapter.py`) passes
its own `--guard-receipt-path` under `output/base21/`, but an unhandled
exception (e.g. `tinker_cookbook.weights` import error, disk-full during
`build_hf_model`) still gets its `status: fail` receipt written to the
Phase-20-specific path — silently overwriting whatever Phase 20 last wrote
there, and leaving no `output/base21/` artifact recording the failure at all.
This is the same class of hardcoded-path hazard the file's own docstring
warns about for `guard_receipt_path` (`merge_adapter.py:508-512`).
**Fix:** Derive the failure receipt location from `args.guard_receipt_path`
when provided (e.g. `args.guard_receipt_path.parent /
"_merge_adapter_result.json"`), falling back to the current hardcoded path
only when no explicit receipt path was given.

### WR-04: Generation/sampling errors are silently folded into "parse fail" / rho measurements

**File:** `scripts/smoke_judge_format_base21.py:99-101` (`judge_generate`)
**Also:** `scripts/capture_judge_responses_tinker.py:124-126`
**Issue:** Both loops catch `Exception` broadly and substitute `""` for the
failed generation:
```python
except Exception as e:  # noqa: BLE001
    print(f"... gen error idx {i}: {e}")
    outs.append("")
```
Downstream, `smoke_judge_format_base21.py` counts `""` as a parse failure
(`parse_judge_scores("") ` won't have `dimension_scores`) identically to a
genuine judge-format non-compliance, inflating `parse_fail_rate` against the
`COMMUNITY_ANCHOR_RATE` comparison with no way to tell infra failures apart
from model output failures in the receipt. In
`capture_judge_responses_tinker.py`, the same empty-string substitution feeds
into `eval_relabel.py`'s rho computation via `judge03_capture_rho.json`, so a
handful of transient sampling-API errors could measurably move the reported
rho without being distinguishable from real capability differences.
**Fix:** Track infra-error count separately from parse-fail count in both
receipts (e.g. `n_infra_error` alongside `n_parse_fail`), so a reader can tell
whether a bad rate reflects the model or the serving/sampling infrastructure.

### WR-05: Judge seed promotion (`best_single_seed`) uses raw point-estimate rho, not a CI-aware comparison

**File:** `scripts/build_judge03_capture_rho.py:112`
**Issue:** `best = max(per_seed, key=lambda s: s["rho"])` selects the seed to
promote into the (real, costly) JUDGE-03 merge+serve step purely by whichever
seed's point-estimate rho happens to be highest, even though each seed's `rho`
already has a `ci_lower`/`ci_upper` computed just a few lines earlier and
available on the same dict. With only 3 seeds and per-seed sample sizes on the
order of the `<wp_judge>` validation split, this is exactly the setup where a
noisy seed can look best by chance (winner's-curse), and that seed then
consumes real Tinker download/merge/serve spend in
`build_judge03_merge_serve.py`.
**Fix:** At minimum, log the CI overlap between the top seeds in the receipt
so a human reviewing `judge03_capture_rho.json` before the merge+serve step
can see whether the "best" seed's advantage is inside the noise band of the
runner-up.

### WR-06: `merge_adapter.py`'s expected-modules-manifest fallback defaults to a Phase-20 (attention-only) file for both merge paths

**File:** `scripts/merge_adapter.py:292-294, 607-609`
**Issue:** Both `_merge_via_tinker_cookbook` and the older PEFT `merge_adapter()`
path fall back to `PROJECT_ROOT / "output" / "base20" / "lora_target_modules.json"`
when `--expected-modules-manifest` isn't supplied. That manifest describes
Phase 20's attention-only probe (a small module count), not the ~240-module
routed-MoE-expert architecture this phase's merges use. Every current Phase 21
caller does pass its own manifest, so this isn't exploitable today, but a
future MoE caller that forgets the flag would get a confusing "module count
mismatch" abort against numbers from a completely different architecture,
rather than a clear "no manifest provided for this architecture" message.
**Fix:** Either require `--expected-modules-manifest` when
`_adapter_has_routed_expert_params()` is true (fail fast with a clear error
instead of falling back to an unrelated default), or pick an architecture-
neutral fallback name that clearly signals "no manifest" rather than silently
resolving to Phase 20's file.

## Info

### IN-01: Duplicated `BASE_MODEL` string literal instead of importing the canonical constant

**File:** `scripts/build_judge03_capture_rho.py:39`
**Issue:** `V4_BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"` is a hardcoded literal
duplicating `tinker_reasoning_data_v4.BASE_MODEL`. Every other script in this
review imports `BASE_MODEL` from `tinker_reasoning_data_v4` directly. If the
v4 base model name ever changes, this file would silently continue to pass
the stale name to `capture_judge_responses_tinker.py --base-model`.
**Fix:** `from tinker_reasoning_data_v4 import BASE_MODEL as V4_BASE_MODEL`.

---

_Reviewed: 2026-07-14T12:11:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
