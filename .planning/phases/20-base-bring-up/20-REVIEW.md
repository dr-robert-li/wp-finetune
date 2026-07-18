---
phase: 20-base-bring-up
reviewed: 2026-07-13T13:50:00Z
depth: standard
files_reviewed: 13
files_reviewed_list:
  - config/train_config_v4.yaml
  - recipes/qwen3.6-35b-a3b-vllm.yaml
  - scripts/_p0_vllm_smoke_serve.py
  - scripts/build_base20_probe_adapter.py
  - scripts/check_token_alignment.py
  - scripts/download_model.py
  - scripts/merge_adapter.py
  - scripts/serve_base20_vllm.sh
  - scripts/smoke_deltanet_base20.py
  - scripts/smoke_load_base20.py
  - scripts/smoke_vl_merge_base20.py
  - tests/test_check_token_alignment.py
  - tests/test_download_model_v4.py
findings:
  critical: 1
  warning: 10
  info: 5
  total: 16
status: issues_found
fix_status: all_fixed
fix_report: .planning/phases/20-base-bring-up/20-REVIEW-FIX.md
fixed_at: 2026-07-13T04:02:51Z
---

# Phase 20: Code Review Report

**Reviewed:** 2026-07-13T13:50:00Z
**Depth:** standard
**Files Reviewed:** 13
**Status:** issues_found

## Summary

Reviewed the Phase 20 base-bring-up scripts, config, and tests for Qwen3.6-35B-A3B
(DGX Spark GB10). The gate scripts (`check_token_alignment.py`,
`smoke_load_base20.py`, `smoke_deltanet_base20.py`, `smoke_vl_merge_base20.py`)
are consistently written to fail closed: every one wraps `main()` in a blanket
`except Exception`, writes a `status=fail` JSON receipt, and returns a non-zero
exit code, and the vLLM lifecycle helpers (`_p0_vllm_smoke_serve.py`) clean up
detached containers with `finally` blocks at two nesting levels (belt-and-
suspenders). The prefix-aware merge path and its module-count guard
(`merge_adapter.py`) were traced end to end against the real receipts already
produced in `output/base20/` (`lora_target_modules.json`,
`_merge_guard_result.json`), and the arithmetic is internally consistent
(190 raw − 90 documented-dropped = 100 merged = 100 expected).

That said, one config-file defect was found and empirically confirmed against
the real downloaded checkpoint's `model.safetensors.index.json`: the LoRA
`target_modules` in `config/train_config_v4.yaml` do not match this
architecture's real module names, and will silently under-train or no-op parts
of the MLP path once this config drives an actual training run. Ten further
warnings cover a broken idempotency check in `download_model.py`, a
completeness guard that only checks for *missing* modules and not *extra*
ones, non-atomic config writes, discarded subprocess diagnostics, and other
robustness gaps. Five info-level items cover dead code and minor style.

**Fix status (2026-07-13, iteration 1):** All 11 in-scope findings (CR-01,
WR-01 through WR-10) fixed and committed. See
[20-REVIEW-FIX.md](20-REVIEW-FIX.md) for the full per-finding report. Info
findings (IN-01 through IN-05) are out of scope and untouched.

## Critical Issues

### CR-01: `train_config_v4.yaml` LoRA `target_modules` do not match the real Qwen3.6-35B-A3B module tree — silent no-op / under-scoped MLP LoRA

**File:** `config/train_config_v4.yaml:24-30`
**Issue:** The `lora.target_modules` list is:
```yaml
target_modules:
- q_proj
- k_proj
- v_proj
- o_proj
- gate_up_proj
- down_proj
```
This list was carried over from the v3.x (dense-FFN Qwen3-30B) config. Verified
directly against the real downloaded checkpoint
(`models/Qwen3.6-35B-A3B/model.safetensors.index.json`):

- `q_proj`/`k_proj`/`v_proj`/`o_proj` correctly match `self_attn.{q,k,v,o}_proj`
  on the 11/48 self-attention layers — fine.
- `gate_up_proj` matches **zero** real `nn.Module`s. On this checkpoint the
  routed-MoE gate/up projection is a single **fused raw `nn.Parameter` tensor**
  named `mlp.experts.gate_up_proj` (no `.weight` suffix, not a `nn.Linear`
  submodule at all) — it is invisible to PEFT's standard
  `target_modules` mechanism, which only walks `model.named_modules()`. PEFT
  silently skips target-module names that never match anything; `get_peft_model()`
  will not raise, it will just attach zero LoRA layers for this name.
- `down_proj` only matches `mlp.shared_expert.down_proj` (one `nn.Linear` per
  layer). It does **not** reach the routed experts' `mlp.experts.down_proj`
  (same fused-Parameter problem as above), which hold the vast majority of the
  MLP capacity in this MoE model.

The net effect: a real training run against this config produces a LoRA that
silently trains almost none of the intended MLP capacity (only the small
`shared_expert`, and nothing on `gate_up_proj` at all), with no error,
warning, or module-count guard to catch it — the exact "silent partial
adapter" failure mode this phase's own `merge_adapter.py` guard (T-20-04a) was
built to prevent, just one layer earlier in the pipeline (train-time config,
not merge-time).

**Fix:** Either target the real per-layer `nn.Linear` paths
(`shared_expert.gate_proj`, `shared_expert.up_proj`, `shared_expert.down_proj`)
and drop `gate_up_proj`, or — if routed-expert LoRA coverage is actually
required — use PEFT's `target_parameters` mechanism (added for exactly this
fused-expert-tensor case) against `mlp.experts.gate_up_proj` /
`mlp.experts.down_proj`, e.g.:
```yaml
lora:
  target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj      # shared_expert only, or supply full paths
  - up_proj
  - down_proj
  # routed-expert coverage needs target_parameters, not target_modules —
  # see peft.tuners.tuners_utils.check_target_module_exists
```
At minimum, add a startup assertion in the training script comparable to
`merge_adapter.py`'s module-count guard, so a target_modules/architecture
mismatch fails loudly instead of silently training a near-empty adapter.

**Fix status:** fixed (commit `1703cc3`) — `target_modules` corrected to
`shared_expert.{gate_proj,up_proj,down_proj}` + `target_parameters` for
`mlp.experts.{gate_up_proj,down_proj}`, matching the working precedent in
`config/train_config_reasoning.yaml`. Requires human verification: the
actual LoRA attach count can only be confirmed by a GPU/CPU dry-run load
(`train_model.py --dry-run` → `assert_router_frozen_and_report`); this fix
was verified against `model.safetensors.index.json` only, no weights loaded.

## Warnings

### WR-01: `download_model.py` idempotency check treats any partial download as complete

**File:** `scripts/download_model.py:67-70`
**Issue:**
```python
existing_shards = count_safetensors(local_dir)
if existing_shards > 0:
    print(f"Model already present at {local_dir} ({existing_shards} safetensors shards). Skipping download.")
    return local_dir
```
A single `.safetensors` shard is enough to skip `snapshot_download()` entirely
— there is no check against the expected total shard count (available from
`model.safetensors.index.json`). If a previous download was interrupted after
downloading 1 of ~26 shards, a re-run silently reports "already present" and
never resumes the rest, defeating the "resume support" the docstring
advertises. Downstream consumers (`AutoModelForCausalLM.from_pretrained` in
`smoke_load_base20.py`/`merge_adapter.py`) will eventually fail loudly on the
missing shard, but only after a confusing detour.
**Fix:** Compare `existing_shards` against the shard count declared in
`model.safetensors.index.json` (if present) before skipping, e.g.:
```python
index_path = local_dir / "model.safetensors.index.json"
if index_path.exists():
    expected = len(set(json.loads(index_path.read_text())["weight_map"].values()))
    if existing_shards < expected:
        print(f"Incomplete download ({existing_shards}/{expected} shards) — resuming.")
    elif existing_shards >= expected:
        return local_dir
```

**Fix status:** fixed (commit `6947148`) — added `expected_shard_count()`;
`download_model()` only skips when `existing_shards >= expected_shards`.
Two new tests added; all 9 tests in `tests/test_download_model_v4.py` pass.

### WR-02: `merge_adapter.py` narrows `target_modules` to bare leaf names, not scoped per-layer paths

**File:** `scripts/merge_adapter.py:181-198`
**Issue:** `kept_leaf_names` collapses every kept tensor's module path down to
its last path component (e.g. `"q_proj"`, `"out_proj"`), then assigns this
list wholesale to `adapter_cfg["target_modules"]`. PEFT's actual matching
(`peft.tuners.tuners_utils.check_target_module_exists`) does
`key.endswith(f".{target_key}")` against **every** module in the live model,
not just the ones the adapter had weights for. This currently produces
correct results only because (a) `AutoModelForCausalLM.from_pretrained()`
resolves this VL checkpoint to a text-only class with no vision tower loaded,
and (b) these particular leaf names happen to be architecturally unique
within the text backbone. Neither of those facts is asserted or guarded by
this code — a future checkpoint/adapter combination where a leaf name
(`out_proj`, `down_proj`, etc.) is reused elsewhere in the model would cause
PEFT to silently attach extra zero-init (or, worse, weight-bearing if the
checkpoint happens to carry an unrelated tensor with that leaf suffix) LoRA
layers outside the intended scope, with no guarantee the count guard would
even notice (see WR-03).
**Fix:** Use the full remapped per-layer dotted paths (already computed as
dict keys in `remapped`) as `target_modules` instead of just their leaf
names, or pass `layers_to_transform`/exact paths so PEFT's exact-key branch
(`key in config.target_modules`) is used instead of the suffix-match branch.

**Fix status:** fixed (commit `558d783`) — `kept_leaf_names` replaced with
`kept_module_paths` (full dotted per-layer paths), routing PEFT through the
exact-key match branch.

### WR-03: Module-count guard only checks for missing modules, not unexpected/extra ones

**File:** `scripts/merge_adapter.py:244-250`
**Issue:**
```python
unexpected_modules = {...}
if unexpected_modules:
    print(f"  [guard] WARNING: {len(unexpected_modules)} unexpected LoRA keys ...")
```
`unexpected_modules` is only printed, never folded into the pass/fail
decision. `_guard_merge_completeness` aborts on missing modules but will
happily accept a merge that attached LoRA to more modules than
`lora_target_modules.json` says were trained. Given the stated goal ("a merge
that exits 0 here would NOT be trustworthy"), an over-broad merge is exactly
as untrustworthy as a partial one and should also raise.
**Fix:** `if unexpected_modules: raise SystemExit(...)` alongside the
existing missing-modules check, or fold `len(unexpected_modules)` into the
guard condition.

**Fix status:** fixed (commit `fa60ea2`) — `unexpected_modules` now folded
into the abort condition, plus added to the written guard receipt.

### WR-04: Merge idempotency short-circuit is effectively dead for the v4 base

**File:** `scripts/merge_adapter.py:373-388`
**Issue:** The "already merged, skip" fast path unconditionally checks that
`<wp_gen>`/`<wp_judge>` encode to single tokens on the *base* tokenizer of the
already-merged model:
```python
special_tokens = config.get("tokenizer", {}).get("special_tokens", ["<wp_gen>", "<wp_judge>"])
all_single = all(len(verify_tok.encode(t, add_special_tokens=False)) == 1 for t in special_tokens)
if all_single:
    ...
    return
```
It does not apply the same base/extended-tokenizer vocab-compatibility
fallback that `_select_serving_tokenizer`/`_verify_merged_model` use later in
the same function (`check_special_tokens=False` path, Rule 1). For the v4
base — which per this file's own comments has "no task-token extension yet"
— `<wp_gen>`/`<wp_judge>` will not be single tokens, `all_single` is always
`False`, and the function silently falls through to a full re-merge on every
invocation, regardless of whether a correct merge already exists on disk.
**Fix:** Reuse `_select_serving_tokenizer`'s compatibility check (or its
`check_special_tokens` flag) in the idempotency probe so it can actually
short-circuit for bases without the extended vocab.

**Fix status:** fixed (commit `660130b`) — idempotency probe now calls
`_select_serving_tokenizer` (base tokenizer only, no model weights loaded)
and short-circuits correctly when `check_special_tokens` is False.

### WR-05: `merge_adapter.py` has no top-level failure handling / receipt, unlike sibling gate scripts

**File:** `scripts/merge_adapter.py:598-600`
**Issue:** Every other Phase 20 gate/smoke script (`check_token_alignment.py`,
`smoke_load_base20.py`, `smoke_deltanet_base20.py`,
`smoke_vl_merge_base20.py`) wraps its `main()`/`run_*()` call in
`try/except Exception` and writes a `status=fail` JSON receipt before exiting
non-zero. `merge_adapter.py`'s `__main__` block has no such wrapper:
```python
if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(args)
```
Any exception raised before `_guard_merge_completeness` (e.g. base model load
failure, adapter file corruption, malformed manifest — see WR-06) propagates
as a raw traceback with no structured receipt. When invoked through
`smoke_vl_merge_base20.py`'s subprocess wrapper this is masked by the
caller's own receipt-on-failure logic, but the script is also documented as
independently runnable (`python3 scripts/merge_adapter.py ...`), and in that
mode a failure leaves no `output/base20/*.json` trail at all.
**Fix:** Wrap `main(args)` in the same try/except + fail-receipt pattern used
by the sibling gate scripts, or explicitly document that this script relies
on its caller for receipts.

**Fix status:** fixed (commit `d15c2e0`) — `__main__` now wraps `main(args)`
in try/except, writing `output/base20/_merge_adapter_result.json` on
unhandled exceptions; `SystemExit` from the script's own diagnosed abort
paths is re-raised unchanged.

### WR-06: Non-atomic config.json rewrites risk leaving a corrupted config on crash

**File:** `scripts/check_token_alignment.py:240-241`, `scripts/merge_adapter.py:314-315`
**Issue:** Both the BASE-02 token-alignment JSON surgery and the BASE-04
VL-config repair write directly to the target `config.json` in place:
```python
with open(CONFIG_JSON_PATH, "w") as f:
    json.dump(raw_config, f, indent=2)
```
```python
with open(merged_config_path, "w") as f:
    json.dump(composite, f, indent=2)
```
Neither uses a temp-file-then-`os.replace()` pattern. If the process is
killed (OOM, disk full, SIGKILL) mid-write, `config.json` is left truncated/
invalid. `check_token_alignment.py` at least keeps a pristine
`config.json.orig` backup for manual recovery, but the corrupted file would
still break every subsequent `AutoConfig.from_pretrained()` call on that
directory until someone notices and manually restores it — exactly the kind
of config-corruption risk this phase is designed to guard against.
**Fix:** Write to a temp file in the same directory and `os.replace()` it
into place:
```python
tmp = CONFIG_JSON_PATH.with_suffix(".json.tmp")
tmp.write_text(json.dumps(raw_config, indent=2))
tmp.replace(CONFIG_JSON_PATH)
```

**Fix status:** fixed (commit `3ec4b80`) — both write sites now use a
`.json.tmp` sibling + `Path.replace()` (atomic).

### WR-07: `boot_vllm()` discards the launch subprocess's stdout/stderr

**File:** `scripts/_p0_vllm_smoke_serve.py:68-69`
**Issue:**
```python
subprocess.run(["bash", serve_script], env=full_env, check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
```
`serve_base20_vllm.sh` prints useful diagnostics on failure before exiting
non-zero (e.g. `ERROR: model dir not found: $MODEL_DIR`). Because both
streams are redirected to `DEVNULL`, `check=True`'s `CalledProcessError`
carries no output, and there is no container yet for `docker logs` to
recover it from (that recovery path only exists once a container has
actually started). A launch-time failure (bad `MODEL_DIR`, bad `PORT`,
missing `docker`) surfaces as a bare "command returned non-zero exit status"
with no indication why.
**Fix:** Capture output (`capture_output=True, text=True`) and include it in
a raised error, or at least print/log it before re-raising.

**Fix status:** fixed (commit `91c818b`) — captures stdout/stderr and raises
a `RuntimeError` including them on non-zero exit.

### WR-08: Container-liveness check uses substring containment, not exact match

**File:** `scripts/_p0_vllm_smoke_serve.py:83-90`
**Issue:**
```python
alive = subprocess.run(
    ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True,
).stdout
if name not in alive:
    ...
```
`alive` is a raw newline-joined string of container names; `name not in
alive` is a substring check, not a per-line exact match. It happens to be
safe today because none of this phase's container names
(`base20-deltanet-smoke`, `base20-merge-smoke-merged`,
`base20-merge-smoke-base`) are substrings of one another, but it is a latent
false-negative risk (e.g. a container literally named `base20-vllm` would
register as "alive" merely because `base20-vllm-2` is running).
**Fix:** `if name not in alive.splitlines():`.

**Fix status:** fixed (commit `91c818b`, same commit as WR-07 — same file,
adjacent lines) — exact per-line match now used.

### WR-09: `recipes/qwen3.6-35b-a3b-vllm.yaml` documents settings that don't match the script actually used for these smoke tests

**File:** `recipes/qwen3.6-35b-a3b-vllm.yaml:24-25` vs `scripts/serve_base20_vllm.sh:34`
**Issue:** The recipe's `metadata.description` explicitly says it documents
the server used for "Phase 20 base bring-up smoke tests (BASE-03 ...,
BASE-04 ...)" and declares `max_model_len: 16384` /
`max_num_batched_tokens: 16384`. The script those smoke tests actually invoke
(`serve_base20_vllm.sh`, via `SERVE_SCRIPT` in `smoke_deltanet_base20.py` and
`smoke_vl_merge_base20.py`) defaults `MAX_MODEL_LEN` to `8192`, and neither
smoke script overrides it. The recipe is not executed by any code in this
repo (no loader renders its `command:` template), so this is a
documentation-only artifact, but it inaccurately describes what these gates
actually exercised — someone reproducing BASE-03/BASE-04 by hand from the
recipe would use a different `max-model-len` than the passing gate run did.
**Fix:** Either update the recipe's `max_model_len`/`max_num_batched_tokens`
to `8192` to match `serve_base20_vllm.sh`'s actual default, or make
`serve_base20_vllm.sh`'s default `16384` to match the recipe.

**Fix status:** fixed (commit `364c8bc`) — updated the recipe's
`max_model_len`/`max_num_batched_tokens` to `8192` (matches the actual
passing BASE-03/BASE-04 gate runs; did not change the script's default,
avoiding an unvalidated memory/timing profile).

### WR-10: Tar extraction in the Tinker probe adapter download doesn't validate member type

**File:** `scripts/build_base20_probe_adapter.py:163-168`
**Issue:**
```python
with tarfile.open(tar_path, "r:*") as tf:
    for m in tf.getmembers():
        name = os.path.basename(m.name)
        if name in ("adapter_config.json", "adapter_model.safetensors"):
            m.name = name
            tf.extract(m, ADAPTER_DIR)
```
Overriding `m.name` to a bare basename correctly neutralizes the classic
path-traversal vector (`../../etc/...`), but `m.isfile()`/`m.issym()` is
never checked. A tar entry named `adapter_config.json` that is actually a
symlink (or device/hardlink) would be extracted as such, and the immediately
following `_attached_modules_from_adapter()` would read through it. The
archive comes from an authenticated Tinker API response, so this is
low-likelihood, but it's a straightforward defense-in-depth gap for code
that already goes out of its way to prevent path traversal.
**Fix:** `if name in (...) and m.isfile(): ...` before extracting.

**Fix status:** fixed (commit `0161ebd`) — added `and m.isfile()` to the
extraction condition.

## Info

### IN-01: Dead import in `download_model.py`

**File:** `scripts/download_model.py:31`
**Issue:** `from scripts.dgx_toolbox import get_toolbox  # noqa: F401,E402 — establishes DGX pattern`
imports `get_toolbox` but never calls it; it has no import-time side effect
(confirmed — `get_toolbox` is a plain function, no module-level
initialization runs on import). The `noqa` comment ("establishes DGX
pattern") suggests intent but the import currently does nothing functional.
**Fix:** Remove it, or actually call `get_toolbox()` if the pattern is meant
to be enforced.

### IN-02: Unused variable in `smoke_vl_merge_base20.py`

**File:** `scripts/smoke_vl_merge_base20.py:122`
**Issue:** `expected_count = len(probe_receipt["attached_modules"])` is
computed but never referenced again — the actual guard comparison later uses
`guard["expected_target_module_count"]` from the merge subprocess's receipt
instead.
**Fix:** Delete the unused assignment.

### IN-03: Deprecated `snapshot_download(resume_download=True)` kwarg

**File:** `scripts/download_model.py:80`
**Issue:** The installed `huggingface_hub` (verified: 1.23.0 in the project's
default env, 1.18.0 in `.venv-tinker`) has removed `resume_download` from
`snapshot_download`'s public behavior; passing it now emits
`UserWarning: The 'resume_download' argument is deprecated and ignored ...
Downloads always resume whenever possible.` It's harmless (the call still
succeeds), just stale/noisy.
**Fix:** Drop the `resume_download=True` kwarg.

### IN-04: Magic number in tokenizer-compatibility heuristic

**File:** `scripts/merge_adapter.py:335`
**Issue:** `compatible = abs(len(extended_tok) - base_vocab_size) < 1000` —
the `1000` threshold has no named constant or comment explaining why that
value was chosen.
**Fix:** Extract to a named constant, e.g. `VOCAB_COMPAT_TOLERANCE = 1000`,
with a one-line rationale.

### IN-05: Pointless f-string

**File:** `scripts/smoke_deltanet_base20.py:77`
**Issue:** `print(f"[A1] docker + nvidia-smi liveness check OK")` has no
interpolated values.
**Fix:** Drop the `f` prefix.

---

_Reviewed: 2026-07-13T13:50:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
