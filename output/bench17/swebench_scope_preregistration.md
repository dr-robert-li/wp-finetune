# SWE-bench Scope Pre-Registration — BENCH-02 (Phase 17)

**Date:** 2026-07-11
**Status:** LOCKED before any SWE-bench eval results exist.

> **NO SWE-bench eval results have been read at the time of this commit.**
> The only SWE-bench harness runs performed so far are (a) two GOLD-patch
> arm64 validation runs (Task 1 — infrastructure sanity, gold patches by
> definition carry no information about this model's performance) and (b) a
> throughput probe that generated 8 patches but never evaluated any of them
> (Task 2 — no patch was applied, no test was run, no resolved/unresolved
> signal exists anywhere). The eval report (plan 17-03) does not exist yet;
> git history proves this commit predates it.

## Pre-Registered Scope

| Role | Dataset | Split | N (measured) | Retrieval | Arch path |
|------|---------|-------|--------------|-----------|-----------|
| **PRIMARY** | SWE-bench Lite (`princeton-nlp/SWE-bench_Lite_oracle` prompts, `SWE-bench/SWE-bench_Lite` eval) | test | **300** | oracle | native arm64, local Docker |
| **SECONDARY** | SWE-bench-Multilingual PHP subset (4 repos: phpoffice/phpspreadsheet, laravel/framework, php-cs-fixer/php-cs-fixer, briannesbitt/carbon) | test | **43** | oracle-equivalent (same prompt construction from `problem_statement` + gold file context) | native arm64, local Docker |

Both generation-mode (non-agentic: one prompt in, one unified-diff patch out),
per the LOCKED CONTEXT.md protocol. No mid-run truncation: if the run
overruns, the overrun is disclosed in the eval receipt, not silently cut.

**Dataset counts were re-measured this session** via
`len(load_swebench_dataset(...))` (the harness's own loader), replacing
training-knowledge figures:

- `SWE-bench/SWE-bench_Lite` test = **300** (matches assumed)
- `SWE-bench/SWE-bench_Multilingual` test = **300**, of which PHP-repo subset = **43** (matches assumed)
- `princeton-nlp/SWE-bench_Lite_oracle` test = **300** (prompt source, `text` field, measured 1.4k–110k tokens with this model's own tokenizer, median ~12k)

## Decision Rule (stated before results)

**Budget: generation + local Docker evaluation ≤ 20 hours wall-clock.**
Select the largest candidate scope whose measured-input projection fits;
scopes are cumulative (PHP subset is near-free and rides along with any
primary choice).

## Measured Inputs (Task 1 + Task 2, committed receipts)

From `output/bench17/swebench_throughput_probe.json` (8 real Lite-oracle
instances, 10.1k–18.4k prompt tokens, vLLM bf16, max_model_len=24576,
concurrency=2, temp 0.0, seed 0, real-generation warm-up gate):

- avg prefill **3562.8 tok/s**, avg decode **17.0 tok/s**
- avg **31.8 s/instance** generation wall-clock at concurrency=2

From `output/bench17/arm64_probe/gold.arm64_probe{1,2}.json` (Task 1):

- Native arm64 eval works end-to-end: PHP env image builds natively (~1 min,
  one-time per repo config), gold patch resolves on both probe instances,
  **~35 s/instance** Docker eval overhead (measured, PHP).
- amd64 requests on this host **fail fast** with `exec format error` (no
  QEMU/binfmt) — no silent-emulation risk.
- Python per-instance Docker overhead was **not** measured this session
  (Task 1's live validation is PHP-only per plan); projection uses a
  conservative 180 s/instance literature-based estimate, flagged as such.

## Decision Rule Applied (arithmetic)

| Candidate scope | Projected wall-clock | ≤ 20h budget? |
|-----------------|---------------------|----------------|
| PHP-Multilingual 43 | **0.61 h** | yes |
| Lite 300 | **16.32 h** | yes |
| **Lite 300 + PHP 43** | **16.93 h** | **yes — largest fitting scope → SELECTED** |
| Verified 500 (+ PHP 43) | 27.21 h (+0.61) | **no — over budget → excluded** |
| Full 2294 | not projected in detail; scales ≈ 2294/300 × Lite ≈ 125 h | no — far over budget |

Verified-500 is excluded by arithmetic, not preference: 27.21 h > 20 h even
before adding the PHP subset. If the Python Docker-overhead estimate (the one
unmeasured input) proved 2× too pessimistic, Verified would still project
~14.7 h generation+eval — but the rule consumes the projection as computed
from the receipt, and upgrading scope after seeing partial results is exactly
the anti-pattern this document forbids.

**Pre-registered scope: SWE-bench Lite 300 (primary) + SWE-bench-Multilingual
PHP 43 (secondary), oracle retrieval, generation-mode, native arm64 local
Docker evaluation.**

## Why These Choices

- **Lite-300 primary:** canonical, comparable to published generation-mode
  numbers; classic Python, explicitly out-of-domain for this PHP/WordPress
  model (the MODEL_CARD caveat is pre-agreed, BENCH-03). Largest classic
  variant fitting the budget.
- **PHP-43 secondary:** in-language bonus signal at near-zero marginal cost
  (0.61 h); the 4 repos are PHP frameworks/libraries, not WordPress —
  materially closer to the model's domain than Django/sympy/astropy, but still
  not in-domain. Both numbers reported side-by-side.
- **Oracle retrieval:** the `*_oracle` dataset variants bundle the exact
  prompt format the original SWE-bench paper used for generation-mode
  ("You will be provided with a partial code base and an issue statement…"),
  keeping the Lite number comparable to published oracle-retrieval results.
  BM25 would add retrieval-quality confounds with no comparability gain here.
- **Native arm64 local eval (no fallback needed):** Task 1 validated the
  `make_test_spec(arch="arm64")` wrapper end-to-end on 2 PHP instances (gold
  patches resolved). sb-cli/cloud is NOT invoked — the everything-local,
  no-external-service preference holds. The Epoch AI prebuilt registry is
  likewise unused; all images build locally.
- **Serving stack for the real run:** vLLM bf16 (same stack as the 0.4484 /
  0.4365 wp-bench receipts), max_model_len=24576, concurrency=2, temperature
  0.0, seed 0, max_tokens 2048, enable_thinking=false — identical to the
  committed throughput-probe config.

## Handling of Over-Length Prompts (pre-registered)

Measured this session: Lite-oracle `text` ranges 1.4k–110k tokens (median
~12k); some instances exceed max_model_len=24576. Pre-registered handling:
prompts that do not fit `max_model_len − max_tokens` are submitted anyway and
scored as **unresolved** if generation fails or the patch is empty/invalid —
counted against the model, disclosed in the receipt with an over-length count.
No silent exclusion, no post-hoc re-scoping.
