# Phase 17: Benchmark Expansion — Research

**Researched:** 2026-07-11
**Domain:** LLM eval harness operation (wp-bench execution) + SWE-bench generation-mode patch eval on aarch64/GB10
**Confidence:** MEDIUM — wp-bench path is HIGH (measured on this repo); SWE-bench feasibility is MEDIUM (verified against installed package source, not yet a live end-to-end run); throughput numbers are LOW (not measured on this repo for either task) and must be probed in Wave 0.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Benchmarks**
- Full wp-bench run (unlimited, not WPBENCH_LIMIT subset) on v1.2 gen model, shipping stack, compared against the 0.4484 Gate-1 receipt (LOCKED — user goal)
- SWE-bench generation eval as full as feasible: "full SWE-bench gen eval" per user; scope constrained only by what the DGX Spark (GB10, aarch64, 121 GB unified) toolchain can honestly EVALUATE, and the chosen scope + constraints must be pre-registered before results are read (LOCKED — BENCH-02)
- Generation-mode (non-agentic patch generation) is the SWE-bench protocol — this is a completion model with task tokens, not an agent scaffold (LOCKED)

**Execution**
- All heavy inference runs on local serving (vLLM or llama.cpp per PIPELINE.md patterns); no paid API for generation (LOCKED — project rule)
- Serving lessons from Phase 15 apply: real-generation warm-up gate before capture (not /health), context window sized as parallel × per-slot need (LOCKED)
- Results recorded as JSON artifacts under output/ with config + seed, same convention as prior eval receipts (LOCKED)

**Documentation**
- MODEL_CARD.md gains a Benchmarks section with both results + explicit out-of-domain caveat for SWE-bench (LOCKED — BENCH-03)
- JOURNAL.md entry (semi-formal voice), STATE/CHANGELOG updates, commit+push as dr-robert-li, no AI co-author trailer (LOCKED — user goal)

### Claude's Discretion
- SWE-bench variant choice (full test split vs Verified vs Lite) — decide from measured feasibility: aarch64 Docker evaluation support, disk, and wall-clock; document the choice and why. If local aarch64 evaluation of generated patches is not feasible, an alternative honest path (e.g. sb-cli cloud evaluation, or patch-generation + apply/lint-only validation with the limitation stated) may be selected, provided the limitation is pre-registered.
- Retrieval context style for generation-mode SWE-bench (oracle vs BM25) — pick the standard that keeps results comparable to published numbers; document.
- Prompt/template adaptation for the fine-tuned task-token model vs base Qwen template.

### Deferred Ideas (OUT OF SCOPE)
- Agentic SWE-bench (SWE-agent / mini-SWE-agent scaffold) — out of scope; generation-mode only this phase
- Publishing benchmark numbers to HF — Phase 18
- Any re-training or prompt-tuning to chase SWE-bench score — never in scope for this milestone
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BENCH-01 | Full (unlimited) wp-bench run on the v1.2 gen model via the shipping stack; score + config + seed recorded and compared to the 0.4484 Gate-1 receipt. | Confirmed exact entrypoint (`wp-bench run --config config/wp-bench.yaml`), confirmed shipping stack = **vLLM, bf16** (gen model was never GGUF-quantized — only the judge was), confirmed test count (344 = 24 execution + 320 knowledge) and empirical wall-clock (~1h) from `wp-finetune:run-evaluation` skill's own timing table. |
| BENCH-02 | SWE-bench generation-mode (non-agentic patch generation) eval at the largest scope the aarch64 toolchain can honestly evaluate; scope and harness constraints pre-registered before results are read. | Verified `swebench` v4.1.0 pip package already installed locally; read its harness source directly (not from training memory) to find the exact aarch64 gap: `TestSpec.platform` supports `arm64` natively but `run_evaluation.py`'s public CLI has no `--arch` flag and hardcodes `arch="x86_64"` in `make_test_spec` — a real, sourced blocker requiring either QEMU emulation or a small wrapper script. Also found the PHP-language SWE-bench-Multilingual subset (43 instances, 4 repos) shipped inside the same installed package — a much cheaper, much more topically-relevant variant than classic Python SWE-bench that CONTEXT.md's framing did not anticipate. |
| BENCH-03 | MODEL_CARD.md Benchmarks section updated with both results + out-of-domain caveat for SWE-bench. | MODEL_CARD.md structure and existing tone read directly; template for the new section follows the existing "Evaluation" section pattern. |
</phase_requirements>

## Summary

BENCH-01 is low-risk and fast: wp-bench's full suite is only 344 tests (24 Docker-graded execution tasks + 320 pure-MCQ knowledge questions, no WordPress runtime needed), the shipping config (`config/wp-bench.yaml`) is already unlimited (`limit: null`), and the project's own `wp-finetune:run-evaluation` skill records empirical wall-clock of **~1 hour** for a full wp-bench pass via vLLM regardless of eval-suite `--limit`. The gen model (`models/qwen3-30b-wp-30_70-reasoning-merged-v4`, 57 GB bf16) was **never quantized** — only the judge got the Q8 GGUF treatment in Phase 15 — so "shipping stack" for BENCH-01 unambiguously means vLLM + bf16, the same stack that produced the 0.4484 Gate-1 number. This is a straightforward re-run, not new engineering.

BENCH-02 is the harder half. The environment check confirms the DGX Spark is aarch64 with Docker 29.2.1 present, ~1.1 TB free disk, and 121 GB unified memory — none of these block SWE-bench outright. The `swebench` pip package (v4.1.0) is *already installed* in this environment, and reading its source directly (not training memory) surfaced a specific, citable gap: the `TestSpec` dataclass has full native support for `arch="arm64"` (`platform` property returns `linux/arm64/v8`), but the public CLI entrypoint (`python -m swebench.harness.run_evaluation`) never exposes an `--arch` flag and hardcodes `x86_64` when building test specs from a raw dataset. Community reporting (GitHub issues, a blog benchmark) confirms this is a known rough edge: on ARM hosts the CLI either needs `--namespace ''` to force local builds (which still request `x86_64` platform and fall through to slow QEMU emulation unless patched) or a small custom script that calls `make_test_spec(..., arch="arm64")` directly. A community-maintained prebuilt arm64 image registry exists (Epoch AI, GHCR) covering 1819/2294 x86_64-parity instances "best-effort, untested" — not a drop-in guarantee.

The most consequential research finding is that the installed `swebench` package ships a **SWE-bench-Multilingual PHP subset** (`swebench.harness.constants.php`: `phpoffice/phpspreadsheet`, `laravel/framework`, `php-cs-fixer/php-cs-fixer`, `briannesbitt/carbon` — 43 total PHP instances across 4 repos, per the public Multilingual leaderboard). This is dramatically cheaper to validate on aarch64 than the 2294/500/300-instance Python sets (4 env images to fix vs. dozens), and while none of the 4 repos are WordPress, "PHP framework/library code" is a materially better domain match for a PHP-specialized model than Django/sklearn/astropy/matplotlib (the classic SWE-bench repos are 100% Python). CONTEXT.md's framing ("SWE-bench is Python-repo patch generation... the number will be low and that is fine") was written before this was known and should be revisited at plan time: running **both** — classic SWE-bench Lite (300, canonical/comparable-to-published-numbers, expected low/out-of-domain per the original framing) and the PHP-Multilingual subset (43, in-language bonus signal, near-zero extra infra cost once one PHP env image builds) — gives the strongest honest story with minimal extra work.

No token-throughput numbers exist anywhere in this repo's receipts for either engine on this model (`gate1_bf16_baseline.json` explicitly states "TTFT/throughput not re-benched"). The one empirical timing data point available (Phase 15 judge captures: 121 items in ~20 minutes via llama.cpp, single concurrent slot, short-to-2048-token outputs) is not directly transferable to SWE-bench's 10-20k-token-input / 1-2k-token-output profile. A short throughput probe (5-10 real SWE-bench-shaped prompts) must be Wave 0 of the plan, before committing to a variant/scope — this is required by CONTEXT.md's own "pre-register scope from MEASURED feasibility" rule, not optional.

**Primary recommendation:** Run BENCH-01 as a straight vLLM full wp-bench re-run (expect ~1h, reuse `config/wp-bench.yaml` unmodified). For BENCH-02, pre-register scope as **SWE-bench Lite (300, oracle retrieval, classic Python, out-of-domain, canonical/comparable) as primary + SWE-bench-Multilingual PHP subset (43, oracle-equivalent, in-language bonus) as secondary**, both generation-mode, both gated by a Wave-0 throughput probe that must show projected wall-clock ≤ ~20h before committing to the Lite count; if the probe shows infeasibility even at Lite scale, drop to a documented smaller pre-registered N (e.g. 50-100) rather than silently truncating mid-run. Local Docker-based SWE-bench evaluation (not sb-cli) is preferred since Docker is present and disk is ample; sb-cli remains the pre-registered fallback only if local aarch64 image builds fail outright for the chosen repos.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| wp-bench code generation + grading | GB10 host (vLLM serving container) + WordPress runtime (Docker, wp-env) | wp-bench harness (Python, host-side orchestration) | Model inference happens in the vLLM container; the 24 execution tests are graded by a second Docker service (WP-CLI sandbox); the 320 knowledge tests never touch Docker. |
| wp-bench knowledge MCQ | GB10 host (vLLM serving container) | wp-bench harness (litellm client) | Pure text completion, scored by the harness locally, no WordPress runtime needed. |
| SWE-bench patch generation | GB10 host (vLLM or llama.cpp serving container) | Eval script (host-side, builds prompts from dataset, calls model, writes predictions.jsonl) | Same local-serving pattern as wp-bench/PIPELINE.md; no agent scaffold — one prompt in, one patch out. |
| SWE-bench patch evaluation (apply + test) | Docker (per-instance repo container, arch-specific) | swebench harness (host-side orchestration, report generation) | Evaluation requires building/pulling an environment image per repo/version and running the repo's real test suite inside a container — this is the aarch64-sensitive step, not generation. |
| Predictions / results artifacts | Filesystem (`output/`) | — | Convention already established (`output/packaging/*.json`) — JSON with config+seed, matching PIPELINE.md's receipt style. |
| MODEL_CARD.md Benchmarks section | Documentation (repo root / `output/packaging/`) | — | Pure documentation update, no runtime component. |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `wp-bench` (this repo's `wp-bench/` submodule/tree) | pinned to repo tree, suite `wp-core-v1` v1.1.0 | WordPress benchmark harness (knowledge + execution) | Already the project's canonical benchmark since Phase 3; `wp-bench.yaml`/`config/wp-bench.yaml` conventions are locked. `[VERIFIED: read from wp-bench/README.md + wp-bench/AGENTS.md + config/wp-bench.yaml in this repo]` |
| `swebench` | 4.1.0 (already installed; `pip index versions swebench` confirms current PyPI latest is 4.1.0, published from PyPI) | SWE-bench Python harness: dataset loading, docker image build, evaluation, report generation | The official package from the SWE-bench team (`swebench.com`); this is the only harness that implements SWE-bench's FAIL_TO_PASS/PASS_TO_PASS semantics correctly. `[ASSUMED — package identity/authorship from training knowledge + PyPI metadata (no official-docs cross-check performed this session); registry existence + local install confirmed by tool]` |
| `datasets` (HuggingFace) | 4.3.0 (installed) | Load `SWE-bench/SWE-bench_Lite`, `SWE-bench/SWE-bench_Verified`, `SWE-bench/SWE-bench_Multilingual` splits | Standard loader used internally by `swebench.harness.utils.load_swebench_dataset`. `[VERIFIED: import + version check ran successfully in this environment]` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `sb-cli` | 0.1.5 (PyPI, not yet installed) | Cloud SWE-bench evaluation (submit predictions, get scored remotely) | Fallback only, if local aarch64 image build for the chosen repo set fails outright. Requires `sb-cli gen-api-key <email>` + email verification; cost/pricing is undocumented in the public README — must be confirmed with the user before use since it is an external network dependency (not "no paid API for generation" territory, but still an external service call worth flagging). `[CITED: github.com/swe-bench/sb-cli README, fetched this session]` |
| `llama.cpp` (`llama-server`) | repo already built at `scripts/run_packaging_recipe.md` path, CUDA GB10 build | Alternative local serving engine if vLLM context/parallel budget doesn't fit SWE-bench's larger per-instance context need | Already used for judge Q8 eval in Phase 15; same binary, same `--parallel N --ctx-size` knobs apply to gen model serving if chosen over vLLM. `[VERIFIED: scripts/run_packaging_recipe.md, this repo]` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Local Docker SWE-bench eval | `sb-cli` cloud eval | No aarch64 build headaches, but external network dependency, undocumented cost, and loses "everything local" auditability that the rest of this pipeline has. Pre-registered fallback only. |
| Classic Python SWE-bench Lite (300) | SWE-bench-Multilingual PHP subset (43) | PHP subset is much smaller/cheaper and closer to the model's domain, but is NOT the benchmark most readers recognize as "SWE-bench" — running it alone (without classic Lite/Verified) would understate the "positioned against a public benchmark" ask. Recommend running both. |
| vLLM serving for gen | llama.cpp GGUF serving for gen | Gen model has never been GGUF-converted (only judge was); building a gen GGUF adds new engineering with no measured quality validation — vLLM bf16 is the only stack with a receipt (0.4484) to compare against. Do not switch engines for this phase unless vLLM context/parallel sizing makes SWE-bench infeasible. |

**Installation:**
```bash
pip install --upgrade swebench   # already installed at 4.1.0, this is a no-op check
pip install sb-cli               # only if the fallback path is invoked
```

**Version verification:**
```bash
pip index versions swebench   # confirmed: 4.1.0 latest, installed
pip index versions sb-cli     # confirmed: 0.1.5 latest, not installed
python3 -c "import swebench; print(swebench.__version__)"   # confirmed: 4.1.0
```

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| swebench | pypi | published 2025-09-11 | unknown (lookup returned null) | none returned by lookup (official repo is github.com/SWE-bench/SWE-bench) | [SUS] (reason: unknown-downloads, no-repository field populated by the seam) | Flagged — already installed and independently confirmed to be the real SWE-bench harness by reading its source (matches public docs: `swebench.com`, `SWE-bench/SWE-bench_Lite` dataset refs, `sb-cli` cross-reference in its own error messages). Planner should still add a `checkpoint:human-verify` before any *new* environment installs it, per protocol, even though this session's install is already trusted-by-inspection. |
| sb-cli | pypi | published 2025-05-14 | unknown | github.com/swe-bench/sb-cli (confirmed via WebFetch of the repo README) | [SUS] (reason: unknown-downloads) | Flagged — only relevant if the local-Docker fallback path is invoked. Planner must add `checkpoint:human-verify` before install, and confirm pricing/cost with the user before any submission (the CLI requires an API key and the README does not state cost).

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** swebench, sb-cli — both are the correct, real, official packages (verified by direct source inspection / README fetch this session), but the legitimacy seam's automated signals (no download-count data source) can't independently confirm that from metadata alone. Treat the SUS tag as "seam couldn't verify," not "evidence of typosquatting" — the planner should still gate any *fresh* install behind a human-verify checkpoint per protocol, but should not block on it given the source-level verification already done here.

## Architecture Patterns

### System Architecture Diagram

```
                     ┌─────────────────────────────────────────────┐
                     │  BENCH-01: wp-bench full run                │
                     │                                               │
  config/wp-bench.yaml │ (limit: null, concurrency: 4)               │
        │              │                                               │
        ▼              │                                               │
  wp-bench CLI ──litellm──▶ vLLM container :8020 (gen model bf16) ──▶ generated PHP/answers
        │              │                                               │
        ├─320 knowledge─▶ scored locally (no Docker)                   │
        │              │                                               │
        └─24 execution──▶ WordPress runtime (Docker, wp-env) ──▶ static+runtime assertions
                     │                                               │
                     └──▶ output/wp-bench-results.json (score, compare to 0.4484)


                     ┌─────────────────────────────────────────────┐
                     │  BENCH-02: SWE-bench generation-mode         │
                     │                                               │
  Wave 0: throughput probe (5-10 real SWE-bench prompts, ~15-20k in / ~1-2k out)
        │              │  -> measured tok/s -> wall-clock projection -> LOCK scope
        ▼              │
  dataset load (HF datasets: Lite/Verified/full or Multilingual-PHP)
        │              │
        ▼              │
  prompt build (oracle retrieval: instance text field already bundles
                the "gold" file context per the *_oracle dataset variant)
        │              │
        ▼              │
  gen model served locally (vLLM or llama.cpp, same pattern as wp-bench)──▶ model_patch (diff)
        │              │
        ▼              │
  predictions.jsonl {instance_id, model_name_or_path, model_patch}
        │              │
        ▼              │
  swebench.harness.run_evaluation  ──▶ per-instance Docker container (arch=arm64 REQUIRES
        │              │                a wrapper around make_test_spec; CLI defaults to x86_64)
        │              │                    │
        │              │                    ▼
        │              │              apply patch, run FAIL_TO_PASS / PASS_TO_PASS tests
        ▼              │
  output/*_swebench_results.json (resolved/unresolved counts, config, pre-registered scope note)
                     └─────────────────────────────────────────────┘
```

### Recommended Project Structure
```
output/
├── bench17/
│   ├── wpbench_full_gate_rerun.json        # BENCH-01: full wp-bench run, vLLM bf16, vs 0.4484
│   ├── swebench_scope_preregistration.md   # BENCH-02: locked BEFORE results are read
│   ├── swebench_throughput_probe.json      # Wave 0: measured tok/s + wall-clock projection
│   ├── swebench_predictions.jsonl          # {instance_id, model_name_or_path, model_patch}
│   ├── swebench_predictions_php.jsonl      # optional: Multilingual-PHP subset predictions
│   └── swebench_eval_report.json           # swebench harness output (resolved/unresolved)
```

### Pattern 1: Reuse the existing vLLM-serve-then-eval pattern (do not invent a new one)

**What:** `dgx_toolbox.py` starts vLLM in a container, eval script/harness talks to `http://localhost:8020/v1` via an OpenAI-compatible client (litellm for wp-bench; a small custom script for SWE-bench predictions), then the container is stopped.
**When to use:** Both BENCH-01 and BENCH-02 generation steps. Do not build a new serving path.
**Example:**
```python
# Source: scripts/run_eval_reasoning.py (this repo), adapted pattern
# wp-bench routes via litellm which needs an explicit provider prefix, and
# the config must set enable_thinking=False for this model (see run_eval_reasoning.py's
# comment on unterminated <think> blocks) before either eval reads the output.
```

### Pattern 2: Real-generation warm-up gate, not `/health`

**What:** Before capturing any scored output, send one real (non-trivial) generation request and confirm a non-empty, well-formed response — not just a 200 from `/health`.
**When to use:** Both BENCH-01 (already implicit in wp-bench's own retry logic) and BENCH-02 (must be added explicitly to any custom SWE-bench prediction-generation script). This is a LOCKED decision from CONTEXT.md, sourced from the Phase 15 lesson: `/health` returning 200 while the server was still loading produced empty captures.
**Example:**
```python
# Source: output/packaging/ens8192_run.log (this repo) — "[eval] model warm (N*2s)" pattern:
# poll with a real short generation request every 2s until it returns non-empty content,
# not until /health returns 200.
```

### Pattern 3: Context budget = parallel slots × per-slot need (llama.cpp), or vLLM `max_model_len` (vLLM)

**What:** llama.cpp's `--parallel N` splits `n_ctx` N ways (a Phase 15 lesson explicitly recorded in STATE.md). vLLM sizes similarly via `--max-model-len` and `--max-num-seqs`.
**When to use:** SWE-bench prompts are 10-20k tokens (much larger than wp-bench's short prompts or the judge's 2048-8192 token captures) — this materially changes the parallel/context tradeoff versus the Phase 15 judge recipe (`--parallel 4, per-slot ctx 11264` won't fit a 20k-token SWE-bench prompt in a 4-way split of a reasonable total context). Plan for **fewer parallel slots, larger per-slot context** for SWE-bench than was used for the judge eval.
**Example:**
```
# Phase 15 (judge, short-ish prompts): --parallel 4 --ctx-size 45056  (11264/slot)
# SWE-bench (10-20k token prompts): need >= 24576 per slot minimum for headroom;
# with model max_position_embeddings=40960, --parallel 1 or 2 is the realistic ceiling
# unless the chosen instances' problem_statement + oracle context stays well under 20k.
```

### Anti-Patterns to Avoid
- **Assuming the gen model has a GGUF checkpoint:** it does not. Only the judge was quantized in Phase 15. Do not reference a `models/_gguf/*gen*` path — it doesn't exist.
- **Running the SWE-bench CLI unmodified on this aarch64 host and trusting the result:** `run_evaluation.py`'s CLI hardcodes `arch="x86_64"` with no flag to override it — running it as-is will attempt x86_64 image operations on an aarch64 host, which either fails outright or silently falls through to slow QEMU emulation (if binfmt is registered) with no warning. Verify emulation status or use a small wrapper calling `make_test_spec(..., arch="arm64")` before trusting any run.
- **Reading SWE-bench results before pre-registering scope:** CONTEXT.md's LOCKED decision requires the scope + constraints to be written down BEFORE results are read. This is a process discipline item, not a code pattern — call it out explicitly as a plan task/checkpoint, not an implicit step.
- **Silently truncating a running eval mid-way if wall-clock overruns:** if the Wave-0 probe under-projects, don't cut short an in-flight run without recording the truncation as an explicit, disclosed limitation in the same JSON receipt.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SWE-bench patch apply + test execution semantics | A custom "apply diff, run tests, compare" script | `swebench.harness.run_evaluation` (or its internal functions called directly for aarch64 patching) | FAIL_TO_PASS/PASS_TO_PASS bookkeeping, log parsing per test framework, and per-repo environment setup scripts are all already encoded in the package's `constants/*.py` and `test_spec/create_scripts.py` — re-deriving this is a multi-week project with a much higher bug surface than patching one `arch` parameter through. |
| SWE-bench dataset retrieval-context construction (oracle/BM25) | A custom "find relevant files" retriever | The pre-built `*_oracle` / `*_bm25_27K` HuggingFace dataset variants (e.g. `princeton-nlp/SWE-bench_Lite_oracle`) | These already bundle the `text` field with the exact prompt format the original paper used — this is what keeps a from-scratch generation-mode number comparable to published numbers. |
| wp-bench grading (PHPCS, security, runtime assertions) | Any new WordPress code-quality checker | wp-bench's existing Docker WP runtime (`wp-bench/runtime/`) | Already built, versioned (WP 6.9), and is the exact grader that produced the 0.4484 comparison point — a new grader would not be comparable. |

**Key insight:** Every "don't hand-roll" item above already exists installed or vendored in this repo/environment. The entire phase should be an integration/orchestration exercise (wire local serving to two existing harnesses), not new benchmark-building — the one genuinely new piece of code is the SWE-bench aarch64 `arch` wrapper (a few lines calling `make_test_spec` directly instead of the CLI) plus a prediction-generation script that calls the served model per-instance.

## Common Pitfalls

### Pitfall 1: Comparing bf16-vLLM wp-bench numbers against a different serving stack by accident
**What goes wrong:** A future engineer might reach for the already-built Q8 GGUF/llama.cpp path (since it's the freshest, most recently-touched serving code in the repo from Phase 15) and run wp-bench against it, producing a number that isn't comparable to the 0.4484 Gate-1 receipt.
**Why it happens:** Phase 15's llama.cpp/GGUF work is the most recent, most-documented serving recipe in the repo, but it was built for the **judge**, not the gen model.
**How to avoid:** Explicitly serve the existing bf16 checkpoint (`models/qwen3-30b-wp-30_70-reasoning-merged-v4`) via vLLM for BENCH-01, matching `output/packaging/gate1_bf16_baseline.json`'s stack exactly.
**Warning signs:** Any mention of a GGUF path for the gen model, or a wp-bench score that differs from 0.4484 by more than a few points without an explanation — check the serving stack first before concluding regression.

### Pitfall 2: Trusting the SWE-bench CLI's default arch on an aarch64 host without checking
**What goes wrong:** `python -m swebench.harness.run_evaluation` silently builds/pulls `linux/x86_64` platform images on an aarch64 Docker host. If QEMU/binfmt emulation is registered, this "works" but runs 5-10x+ slower than native (per community benchmarking); if not registered, image builds fail with cryptic exec-format errors.
**Why it happens:** `arch` defaults to `"x86_64"` inside `make_test_spec()` and is never threaded through as a CLI flag in `run_evaluation.py`'s `__main__` block — confirmed by reading the installed package source directly this session (swebench 4.1.0).
**How to avoid:** Before running any real instances, run `docker run --rm --platform linux/arm64/v8 hello-world` to confirm native arm64 execution works, and write a thin wrapper that calls `get_test_specs_from_dataset` + `make_test_spec(..., arch="arm64")` directly rather than invoking the CLI script as-is.
**Warning signs:** Docker image names appearing in `docker images` output with an unexpected arch suffix, or evaluation runs that take dramatically longer than the Wave-0 probe projected.

### Pitfall 3: Underestimating SWE-bench wall-clock because wp-bench felt fast
**What goes wrong:** wp-bench's full run takes ~1 hour because most of its 344 tests are short MCQs. SWE-bench prompts are 10-20k input tokens with 1-2k token patch outputs, and each instance ALSO requires a Docker build/test-run step after generation — an entirely different cost profile. Naively assuming "eval is fast, we measured 1h for wp-bench" would badly under-provision time for even Lite (300 instances).
**Why it happens:** No throughput numbers exist in this repo for either engine at SWE-bench-scale context lengths (`gate1_bf16_baseline.json` explicitly states throughput was never re-benched).
**How to avoid:** Run the Wave-0 throughput probe (5-10 real prompts, full context length) before committing to any variant scope, and multiply the generation-only estimate by a Docker-build/test-run overhead factor (estimate a few minutes/instance for the eval-side Docker step, based on the "6x faster" native-vs-emulated finding and typical repo test-suite durations reported in SWE-bench literature) before locking the pre-registration document.
**Warning signs:** A pre-registration scope written without a Wave-0 probe result cited in it.

### Pitfall 4: Treating classic SWE-bench as the only valid variant given the model's PHP specialization
**What goes wrong:** CONTEXT.md's framing ("this model is WordPress/PHP-specialized... the number will be low and that is fine") pre-dates this research's discovery of the PHP-Multilingual subset shipped inside the installed `swebench` package. Sticking rigidly to the original framing would forgo a much more informative, nearly-free additional data point.
**Why it happens:** The PHP subset is not widely known/publicized relative to classic SWE-bench Lite/Verified; it wasn't in scope of the original CONTEXT.md discussion.
**How to avoid:** Surface this to the user/planner explicitly (as done in this document) and recommend running both variants rather than picking one exclusively.
**Warning signs:** A plan that runs only classic Python SWE-bench without at least considering the PHP subset as a cheap addendum.

## Code Examples

### Building predictions.jsonl in the exact schema swebench expects
```python
# Source: swebench.harness.constants (installed package, this session)
# Confirmed field names via: from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
# KEY_INSTANCE_ID = "instance_id"
# KEY_MODEL       = "model_name_or_path"
# KEY_PREDICTION  = "model_patch"
import json

def write_predictions(predictions: list[dict], out_path: str):
    """Each item: {"instance_id": str, "model_name_or_path": str, "model_patch": str (unified diff)}"""
    with open(out_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
```

### Loading a dataset variant (local JSON/JSONL or HuggingFace)
```python
# Source: swebench.harness.utils.load_swebench_dataset (installed package, read directly)
from swebench.harness.utils import load_swebench_dataset

# Classic:
lite = load_swebench_dataset("SWE-bench/SWE-bench_Lite", split="test")
verified = load_swebench_dataset("SWE-bench/SWE-bench_Verified", split="test")
full = load_swebench_dataset("SWE-bench/SWE-bench", split="test")
# Multilingual (PHP + 7 other languages, filter repo in the constants list):
multilingual = load_swebench_dataset("SWE-bench/SWE-bench_Multilingual", split="test")
php_only = [i for i in multilingual if i["repo"] in {
    "phpoffice/phpspreadsheet", "laravel/framework",
    "php-cs-fixer/php-cs-fixer", "briannesbitt/carbon",
}]
```

### Forcing native arm64 test specs (bypassing the CLI's hardcoded x86_64)
```python
# Source: swebench.harness.test_spec.test_spec.make_test_spec (installed package, read directly)
# The public run_evaluation.py CLI never exposes --arch; call make_test_spec directly instead.
from swebench.harness.test_spec.test_spec import make_test_spec

test_specs = [make_test_spec(instance, arch="arm64") for instance in php_only]
# Then feed test_specs into swebench.harness.run_evaluation.run_instances(...) directly,
# or into docker_build.build_env_images / build_instance_images, rather than the CLI script.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| SWE-bench = Python-only, agentic-scaffold evaluation | SWE-bench-Multilingual adds 8 languages including PHP (9 languages total, 300 instances, 42 repos) | Multilingual variant is a documented, separately-branded extension of the original SWE-bench project | For this project specifically: makes an in-language (though not in-domain-WordPress) SWE-bench variant available for the first time, changing the "the number will be low, that's fine" framing from CONTEXT.md into "we have a choice of how out-of-domain to be." |
| SWE-bench harness assumed x86_64-only | `swebench` >=4.x has a native `arch="arm64"` code path inside `TestSpec`, just not wired to the CLI | Unclear exact version this landed (traced to the installed 4.1.0's source this session); GitHub issues #375/#520 (community-reported gaps) are still open at the time of this research, suggesting the arm64 path is present-but-incomplete rather than fully productionized | Local aarch64 evaluation is possible with a small wrapper, not "impossible," which is a stronger position than "must use sb-cli" — but still requires that wrapper to be written and tested. |

**Deprecated/outdated:**
- Nothing in this repo's own tooling is deprecated by this research; wp-bench and the vLLM serving pattern are unchanged from Phase 3-15 usage.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `swebench` PyPI package is authored/maintained by the official SWE-bench team (not a lookalike) | Standard Stack, Package Legitimacy Audit | If wrong, evaluation results would be produced by an unofficial/incorrect harness; mitigated in practice this session by reading the installed package's source and confirming it matches all publicly-documented SWE-bench conventions (KEY_INSTANCE_ID/KEY_MODEL/KEY_PREDICTION field names, dataset names, sb-cli cross-reference in its own error message) — but the legitimacy seam itself could not independently confirm via download-count/repo-metadata signals this session. |
| A2 | Projected SWE-bench wall-clock (10-20k input / 1-2k output tokens per instance) will land in the "hours per hundred instances" range on this hardware, not "minutes per hundred" or "days per hundred" | Summary, Common Pitfalls #3 | If throughput is much lower than assumed, even Lite (300) could exceed 24h and the pre-registered scope would need to shrink further; if much higher, a more ambitious scope (Verified, 500) becomes feasible and the phase would be needlessly conservative. This is exactly why a Wave-0 measured probe is mandatory before locking scope — treat this whole estimate as a placeholder to be replaced by measurement, not a plan input. |
| A3 | sb-cli has no meaningful cost/is viable as a fallback | Standard Stack (Supporting), Package Legitimacy Audit | The public README does not document pricing; if it turns out to require paid credits, this would violate CONTEXT.md's "no paid API for generation" spirit (though sb-cli is evaluation, not generation, so it may be a gray area) — confirm with the user before invoking, never assume free. |
| A4 | Standard SWE-bench dataset counts (full test 2294, Verified 500, Lite 300) are current | Summary, Standard Stack | These are well-known, stable, widely-cited figures from training knowledge; not independently re-counted via `len(load_swebench_dataset(...))` this session (would require a network-connected `datasets` load, not run to conserve time/avoid an unplanned multi-GB download during research). Verify with a quick `len()` check at Wave 0 before finalizing the pre-registration doc. |

**If this table is empty:** N/A — see rows above.

## Open Questions

1. **Does this Docker install have QEMU/binfmt registered for cross-arch emulation?**
   - What we know: Docker 29.2.1 is present; `docker info` was checked but binfmt registration was not explicitly probed this session.
   - What's unclear: Whether an accidental x86_64 image request would fail fast (safe) or silently emulate (slow, misleading if not measured).
   - Recommendation: Wave 0 task — `docker run --rm --platform linux/amd64 hello-world` and `docker run --rm --platform linux/arm64/v8 hello-world`; record which succeed and how long the amd64 one takes relative to native, before any real SWE-bench build.

2. **What exact throughput (tok/s) does this GB10 host achieve for this MoE model at 15-20k-token prompts?**
   - What we know: No measurement exists in this repo for this profile; the closest data point (judge captures, ~10s/item at short-to-2048-token outputs) doesn't transfer.
   - What's unclear: Prefill and decode tok/s at the SWE-bench-relevant context length, and whether vLLM or llama.cpp is faster for this profile on this model.
   - Recommendation: Wave-0 throughput probe is a hard prerequisite to locking BENCH-02 scope, per CONTEXT.md's own "pre-register from measured feasibility" rule.

3. **Does the user want to run the PHP-Multilingual subset in addition to (or instead of) classic SWE-bench?**
   - What we know: The PHP subset is cheap and closer to the model's domain, but isn't "the" SWE-bench benchmark most readers recognize, and none of its 4 repos are WordPress.
   - What's unclear: Whether the "positioning against a public coding benchmark" goal is best served by canonical comparability (classic Lite) or domain relevance (PHP subset) or both.
   - Recommendation: Surface this choice explicitly to the user at plan/discuss time; this document recommends running both given the PHP subset's low marginal cost.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| aarch64 host | BENCH-02 (SWE-bench Docker eval) | Confirmed (`uname -m` = aarch64) | — | — |
| Docker | BENCH-02 (env/instance images, WordPress runtime for wp-bench execution tests) | Confirmed (29.2.1, 9 containers, 54 images present) | 29.2.1 | — |
| Disk space | BENCH-02 (image builds), BENCH-01 (none significant) | Confirmed (~1.1 TB free on `/`) | — | — |
| Unified memory (121 GB) | Serving the 57 GB bf16 gen model + KV cache | Confirmed present, matches PIPELINE.md's documented host | — | — |
| `swebench` (pip) | BENCH-02 | Confirmed installed (4.1.0, matches PyPI latest) | 4.1.0 | — |
| `datasets` (pip) | BENCH-02 (dataset loading) | Confirmed installed | 4.3.0 | — |
| `sb-cli` (pip) | BENCH-02 fallback path only | Not installed (available on PyPI at 0.1.5) | — | `pip install sb-cli` if local Docker eval proves infeasible |
| Native arm64 SWE-bench evaluation images | BENCH-02 | NOT verified this session (requires the `arch="arm64"` wrapper + a real build attempt) | — | Epoch AI's best-effort prebuilt arm64 registry (1819/2294 x86_64-parity images, untested per their own disclaimer), or QEMU emulation of x86_64 images (slow), or sb-cli cloud fallback |
| vLLM serving container | BENCH-01, BENCH-02 generation | Confirmed used successfully throughout Phases 3-15 on this exact host | — | llama.cpp (already provisioned from Phase 15) |
| WordPress runtime (`wp-env`/Docker) for wp-bench execution tests | BENCH-01 (24 of 344 tests) | Not explicitly re-verified this session, but is the same runtime used to produce prior wp-bench receipts on this host | — | — |

**Missing dependencies with no fallback:** none identified — every dependency either exists or has a documented fallback path above.

**Missing dependencies with fallback:**
- Native arm64 SWE-bench evaluation images (fallback: emulation, community prebuilt registry, or sb-cli — pick the cheapest that a Wave-0 test confirms works)

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | None dedicated to this phase — this is an eval-execution phase, not application code. Validation is the benchmark harnesses themselves (wp-bench, swebench) plus small glue scripts. |
| Config file | `config/wp-bench.yaml` (existing); a new `output/bench17/swebench_scope_preregistration.md` acts as the pre-registration "config" for BENCH-02 |
| Quick run command | `wp-bench run --config config/wp-bench.yaml --limit 5` (smoke, NOT the BENCH-01 deliverable — BENCH-01 itself must be unlimited per CONTEXT.md) |
| Full suite command | `wp-bench run --config config/wp-bench.yaml` (BENCH-01); custom generation script + `swebench.harness.run_evaluation` (BENCH-02) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BENCH-01 | Full wp-bench run produces a score comparable to 0.4484 | integration (external harness) | `wp-bench run --config config/wp-bench.yaml` (via vLLM, gen model served) | ✅ (harness exists, config exists) |
| BENCH-02 | SWE-bench generation-mode eval at pre-registered scope | integration (external harness + custom glue) | Wave-0 throughput probe script (new) → prediction-generation script (new) → `swebench.harness.run_evaluation` (existing, wrapped for arm64) | ❌ Wave 0 — glue scripts don't exist yet |
| BENCH-03 | MODEL_CARD.md Benchmarks section reflects both results | manual/doc check | diff review of MODEL_CARD.md against BENCH-01/02 output JSON | ✅ (MODEL_CARD.md exists, template pattern established) |

### Sampling Rate
- **Per task commit:** Confirm the relevant JSON receipt was written and is well-formed (`python3 -m json.tool output/bench17/*.json`).
- **Per wave merge:** Re-diff MODEL_CARD.md's new Benchmarks section against the receipt JSONs to catch transcription errors.
- **Phase gate:** Both receipts present, pre-registration doc committed BEFORE the SWE-bench eval report exists (git history / commit timestamps prove ordering), MODEL_CARD.md updated.

### Wave 0 Gaps
- [ ] `docker run --rm --platform linux/arm64/v8 hello-world` and the amd64 equivalent — confirms native vs emulated arch behavior on this host, needed before any SWE-bench Docker work
- [ ] SWE-bench throughput probe script (5-10 real prompts at full context length, both candidate variants if time permits) — feeds the pre-registration doc's wall-clock projection
- [ ] `arch="arm64"` wrapper around `make_test_spec`/`run_instances` — the one piece of genuinely new code this phase needs for BENCH-02
- [ ] Dataset instance-count re-verification (`len(load_swebench_dataset(...))` for the chosen variant(s)) before finalizing the pre-registration doc (A4 above)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-------------------|
| V2 Authentication | no | No auth surface introduced — local-only serving, no new user-facing endpoints |
| V3 Session Management | no | N/A |
| V4 Access Control | no | N/A |
| V5 Input Validation | yes (narrow) | SWE-bench predictions (`model_patch`) are unified diffs applied inside per-instance Docker containers, not on the host — this containment is the existing swebench harness design and must not be bypassed (e.g., never `git apply` a model-generated patch directly against the host filesystem) |
| V6 Cryptography | no | N/A |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|----------------------|
| Model-generated patch contains destructive/malicious shell content picked up by a naive apply script | Tampering | Always route patch application through the swebench harness's containerized apply step (`docker_build.py` / `test_spec.py` generated scripts), never a bespoke host-side `git apply` |
| sb-cli fallback path exfiltrates code/instance data to a third-party service | Information Disclosure | Treat sb-cli as an explicit, user-confirmed external dependency (per Open Question / A3 above) — do not invoke silently as an automatic fallback without surfacing the network call to the user first |
| Docker container escape from a per-instance SWE-bench eval image (arbitrary repo code + test suite execution) | Elevation of Privilege | This is the swebench harness's own existing containment model — no new mitigation needed beyond using the harness as designed (not running instance test suites directly on host) |

## Sources

### Primary (HIGH confidence)
- `wp-bench/README.md`, `wp-bench/AGENTS.md`, `config/wp-bench.yaml` (this repo) — wp-bench entrypoint, config, test-suite structure
- `output/packaging/gate1_bf16_baseline.json`, `output/packaging/MODEL_CARD.md`, `output/packaging/pkg03_ens8192_results.json`, `.planning/phases/15-packaging/15-02-SUMMARY.md` (this repo) — confirms gen model was never quantized, shipping stack = vLLM bf16
- `.claude/skills/wp-finetune:run-evaluation/SKILL.md` (this repo) — empirical wp-bench wall-clock (~1h regardless of `--limit`), vLLM serving pattern, health-timeout guidance
- `/home/robert_li/miniconda3/lib/python3.13/site-packages/swebench/` (installed package v4.1.0, read directly this session) — `harness/test_spec/test_spec.py` (arch/platform logic), `harness/run_evaluation.py` (CLI arg surface, confirms no `--arch` flag), `harness/constants/php.py` (PHP-Multilingual repo list), `harness/constants/__init__.py` (KEY_INSTANCE_ID/KEY_MODEL/KEY_PREDICTION field names)
- Direct environment probes this session: `uname -m`, `docker --version` / `docker info`, `df -h`, `free -h`, `pip index versions swebench`/`sb-cli`

### Secondary (MEDIUM confidence)
- WebFetch of `github.com/swe-bench/sb-cli` README this session — auth flow, dataset support, submission command
- WebSearch: "SWE-bench evaluation harness arm64 aarch64 docker image support" — GitHub issues #375/#520, Epoch AI prebuilt registry details, greynewell.com ARM64 benchmark blog
- WebSearch: "SWE-bench multilingual PHP dataset repos" — confirms 9-language, 300-instance, 42-repo structure and per-language counts (PHP: 43)
- WebSearch: SWE-bench oracle/BM25 retrieval dataset field structure (`princeton-nlp/SWE-bench_*_oracle`, `text`/`patch`/`problem_statement` fields)

### Tertiary (LOW confidence)
- Standard SWE-bench dataset instance counts (full 2294 / Verified 500 / Lite 300) — from training knowledge, not re-counted via a live `datasets` load this session; flagged in Assumptions Log (A4) for Wave-0 verification
- Throughput/wall-clock projections for SWE-bench-scale prompts on this specific host — no measurement exists anywhere in this repo; entirely a placeholder pending the mandatory Wave-0 probe

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — wp-bench stack confirmed from repo files; swebench package confirmed installed and its source read directly
- Architecture: HIGH for BENCH-01 (proven pattern, reused verbatim from Phases 3-15); MEDIUM for BENCH-02 (pattern is analogous but the arm64 wrapper is new, untested code)
- Pitfalls: HIGH — all four pitfalls are sourced from either this repo's own prior incidents (Phase 15 lessons) or directly-read package source, not speculation

**Research date:** 2026-07-11
**Valid until:** 2026-08-10 (30 days — stable domain; re-check `swebench`/`sb-cli` versions and the arm64 CLI gap if the phase is delayed past this window, since the harness is under active development per its GitHub issue tracker)
