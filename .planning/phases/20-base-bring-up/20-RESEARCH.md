# Phase 20: Base Bring-Up - Research

**Researched:** 2026-07-13
**Domain:** Model download/load/serve smoke-testing for a new MoE-VL base (Qwen3.6-35B-A3B) on an existing GB10 fine-tuning pipeline
**Confidence:** HIGH

## Summary

Phase 20 has no new external-research surface — `.planning/research/{STACK,ARCHITECTURE,PITFALLS}.md` (dated
2026-07-12, re-verified 2026-07-12) already answer every technical question this phase raises, against
primary sources (live HF `config.json`/`model.safetensors.index.json`, vLLM recipe docs, GitHub issues). This
document's job is different: it maps those findings onto the **actual repo**, so the planner can write tasks
against real file paths and reuse real harness code instead of inventing new patterns.

The repo already has every building block Phase 20 needs, proven working on the old base: `scripts/download_model.py`
(HF `snapshot_download` with resume, config-driven `local_dir`), `scripts/_p0_vllm_smoke_serve.py`
(`boot_vllm`/`wait_healthy`/`generate`/`stop_vllm` with the "real generation, not `/v1/models`" gate — carry-forward
lesson 2), `scripts/merge_adapter.py` (LoRA merge + verification roundtrip + fallback command), and a working vLLM
launch precedent for **this exact model family** already committed to the repo:
`recipes/qwen3.6-35b-a3b-fp8-vllm.yaml` (used for Phase 0 LLM-assisted rubric checks, `container:
ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest`, `gpu_memory_utilization: 0.55`, `trust_remote_code: true`).
Phase 20's job is to adapt these four proven pieces to the new base's specifics (VL wrapper key prefix, bf16 not
FP8, DeltaNet CUDA-graph capture, eos/pad token mismatch) — not to build new infrastructure.

**Primary recommendation:** Extend `download_model.py`/`merge_adapter.py`/`_p0_vllm_smoke_serve.py` with
base-specific parameters (new model name/local_dir, `--language-model-only`, `trust_remote_code=True`,
`use_kernels` toggle) rather than writing new scripts from scratch; write one new script for the Stage 1.5
eos/pad alignment gate (no prior equivalent exists); write JSON gate receipts to `output/base20/` following the
existing `output/tinker/PROMOTED_*.json` / `output/merge_v4_winner/merge_report.json` receipt convention (flat
JSON, `status` field, assert-checked fields, human-readable).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Weight download | Host filesystem (`models/`) | HuggingFace Hub (remote) | `snapshot_download` writes directly to `models/<name>/`, no intermediate service |
| eos/pad token alignment | Model config + tokenizer (in-process Python) | — | Pure config/tokenizer object mutation, no serving involved; must run before any Tinker/HF trainer touches the config |
| DeltaNet kernel smoke | Inference runtime (transformers `from_pretrained` in-process, or vLLM container) | GB10 CUDA driver/SM121 | Kernel dispatch decision (`use_kernels=True`) is a `transformers`-level flag; vLLM's own kernel stack is a separate, second surface that must be smoke-tested independently |
| VL merge (adapter -> merged weights) | Host Python process (CPU, `device_map="cpu"`) | PEFT/transformers library | Same as old base — `merge_adapter.py` already does this on CPU/unified RAM, architecture-agnostic beyond key-prefix resolution |
| Serving smoke (merge -> serve round-trip) | vLLM container (Docker, GB10 GPU) | `dgx_toolbox.py` orchestration layer | vLLM owns CUDA-graph capture, `--language-model-only` remap, `--enforce-eager` fallback — none of this is host-Python-visible |
| Gate receipts | Host filesystem (`output/base20/*.json`) | — | Matches existing `output/tinker/`, `output/merge_v4_winner/` convention; consumed by later phases and by human review, not a live service |

## User Constraints

<user_constraints>
### Locked Decisions (from V4-RERUN-ROADMAP.md, treated as CONTEXT.md-equivalent per task framing — no separate CONTEXT.md exists for this phase)

- Base is locked: `Qwen/Qwen3.6-35B-A3B` (35B total / 3B active, 256 experts top-8 + 1 shared, hybrid
  Gated-DeltaNet/Gated-Attention, Apache-2.0, VL checkpoint). No other candidate is in scope.
- Phase 20 scope is exactly: download/load smoke test, eos/pad token-ID alignment (work item 2, Stage 1.5
  prerequisite gate — **must block Stage 2/3 SFT on failure**), DeltaNet-on-aarch64 op smoke check, VL
  merge-path check (Tinker adapter export + `merge_adapter.py` handle `model.language_model.*` prefix; serve
  text-only via `--language-model-only`).
- Phase 20 has NO dependency within v4.0 (first phase; depends only on Phase 19 sign-off, which is already
  granted — this document exists because sign-off happened).
- Cost/scheduling: if Stage 2/3 Tinker spend lands after 2026-07-17, budget the price rise (train ~+10%,
  dominant driver) — informational for Phase 20 itself (Phase 20 has no Tinker spend; it uses locally-loaded
  weights for the download/load/merge/serve smoke, and an existing/throwaway adapter, not a fresh Tinker run
  — see Open Questions for the "how to get a LoRA to test merge with, cheaply" resolution below).
- Carry-forward lesson 2 (real-generation warm-up gating) and lesson 4 (CI-aware / bootstrap-lower-bound gates
  where applicable) apply to every smoke gate in this phase.
- Pre-registered success criteria for the WHOLE milestone (judge rho >0.85/>0.87, wp-bench >=0.4286) are
  Phase 23's concern, not Phase 20's — Phase 20 has no quality bar, only pass/fail infra gates.

### Claude's Discretion

- Exact script/file naming for the new Stage 1.5 eos/pad alignment check (no prior equivalent exists in repo).
- Whether to build a throwaway/zero-init adapter locally vs. spend a minimal real Tinker run to exercise the
  VL merge path (BASE-04) — see Open Questions; research recommends the zero-cost local option.
- Exact gate-receipt JSON schema for `output/base20/*.json` (follow existing `output/tinker/PROMOTED_*.json`
  / `output/merge_v4_winner/merge_report.json` conventions, not a new schema).
- `use_kernels=True` vs `False` decision for DeltaNet inference — research (STACK.md) recommends trying
  `True` and recording whether `trust_remote_code=True` for the `Atlas-Inference/gdn` community kernel is
  acceptable under this project's threat model, falling back to `False` (slower prefill, flat decode) if not.

### Deferred Ideas (OUT OF SCOPE for Phase 20)

- Any SFT training, generation-model or judge-model (Phase 21).
- Sieve/protected-mask tooling adaptation (Phase 25 — explicitly independent, can start early but is a
  separate phase/plan, not Phase 20's job).
- RL re-test (Phase 24), pruning re-test (Phase 26), packaging/quantization (Phase 27).
- Re-opening the relabel campaign or the base-model lock — both are settled decisions from Phase 19.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BASE-01 | Qwen3.6-35B-A3B downloads and loads on GB10 (`trust_remote_code` on model+tokenizer, transformers 5.x import check, `Qwen3_5MoeForConditionalGeneration` class resolution) | STACK.md §5 (class names verified against live HF doc); `scripts/download_model.py` is the reusable download harness; local env already has `transformers==5.3.0` installed (verified this session via `python3 -c "import transformers; print(transformers.__version__)"`) |
| BASE-02 | eos/pad token-ID alignment gate passes — assert-match tokenizer special tokens + stop-token smoke generate; blocks Stage 2/3 on failure | PITFALLS.md Pitfall 1 (primary source: QwenLM/Qwen3.6 discussion #96, maintainer-confirmed WAI); exact IDs given (`tokenizer.eos_token_id=248046` vs `model.config.eos_token_id=248044`, `model.config.pad_token_id=None`) |
| BASE-03 | DeltaNet aarch64 serving smoke passes WITH CUDA-graph capture enabled (vLLM >=0.19.0); `--enforce-eager` fallback documented; `use_kernels` decision recorded | STACK.md §2 (measured `use_kernels` prefill numbers), ARCHITECTURE.md §4 (vLLM issue #35945 CUDA-graph/causal_conv1d crash), PITFALLS.md Pitfall 2 (same, with 0.80 `--gpu-memory-utilization` recommendation); `scripts/_p0_vllm_smoke_serve.py` + `recipes/qwen3.6-35b-a3b-fp8-vllm.yaml` are the reusable serving harness/precedent |
| BASE-04 | VL merge-path validated end-to-end: Tinker LoRA export -> merge onto `model.language_model.*` keys -> vLLM serve (`--language-model-only`) -> real generation | ARCHITECTURE.md §2-3 (key-prefix facts verified against live `model.safetensors.index.json`), PITFALLS.md Pitfall 3 (dual key-prefix convention, merge-time vs serve-time); `scripts/merge_adapter.py` is the reusable merge harness (needs prefix-awareness change, see Findings) |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `transformers` | 5.x, latest (installed locally: **5.3.0**, verified this session) [VERIFIED: local env] | Model/tokenizer loading, `Qwen3_5MoeForConditionalGeneration` class | Already the project's ML framework; STACK.md confirms class exists at `/v5.13.1/` doc path but exact min version is unpinned — gate via import smoke test, not a version string [CITED: huggingface.co/docs/transformers/model_doc/qwen3_5_moe] |
| `vllm` | `>=0.19.0` (vendor-recommended minimum on this model's own HF README; `>=0.17.0` hard floor) [CITED: recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B] | Serving, VL merge-path round-trip, DeltaNet smoke | Already the project's serving stack (`scripts/dgx_toolbox.py`, `serve_*_vllm.sh`); runs inside `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest` container, not host pip |
| `huggingface_hub` | already installed (used by `scripts/download_model.py`) [VERIFIED: repo] | `snapshot_download` with resume | Existing project dependency, no change needed |
| `peft` | already installed (used by `scripts/merge_adapter.py`) [VERIFIED: repo] | LoRA adapter load + `merge_and_unload()` | Existing project dependency; PEFT resolves target modules by traversing the live model object, not raw safetensors key strings — architecture-robust to the `model.language_model.*` prefix change per ARCHITECTURE.md §6 |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `kernels` | latest (`0.16.0` confirmed live on PyPI this session via `pip index versions kernels`) [CITED: huggingface.co/docs/transformers/model_doc/qwen3_5_moe] | Enables `use_kernels=True` for the `Atlas-Inference/gdn` Hub kernel (1.38x prefill speedup on DeltaNet layers) | Only if the BASE-03 discretion decision picks `use_kernels=True`; requires `pip install -U kernels` and `trust_remote_code=True` for the community kernel repo (not yet on the trusted-kernels allowlist) — record the trust decision explicitly per Claude's Discretion above |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `use_kernels=True` (Hub kernel) | `use_kernels=False` (PyTorch fallback, default) | 1.38x slower prefill only (0.73s vs 0.53s at 1024-token prompt), decode throughput flat (~16 tok/s either way — memory-bandwidth bound); no `trust_remote_code` exposure to a community (non-allowlisted) kernel repo. Given eval-harness wall-clock sensitivity (~19 min/wp-bench arm budget) and this project's existing conservative `trust_remote_code` posture, **recommend defaulting to `False` and only flipping to `True` if a later phase's wall-clock budget is tight** — record the choice, don't silently pick one. |
| Real Tinker LoRA run to exercise BASE-04 | Locally-built zero-init/tiny adapter targeting the actual `model.language_model.*` module paths | See Open Questions — locally-built adapter is $0 and faster; a real Tinker run is the only way to also validate Tinker's own export-side key-prefix behavior, which is the actual unresolved risk (ARCHITECTURE.md §6: "Verify... this is the real risk, not the merge script itself"). Recommend a **minimal real Tinker run** (see Open Questions) specifically because the zero-cost local adapter cannot test the one unverified link in the chain. |

**Installation:**
```bash
pip install -U kernels   # only if use_kernels=True is chosen for BASE-03
```

**Version verification:** `transformers==5.3.0` confirmed installed and importable in the local host env this
session. `vllm` is not in the host pip environment (by design — it runs inside the
`ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest` Docker container via `dgx_toolbox.py`); confirm the
container's vLLM version as part of the BASE-03 smoke test itself (`vllm.__version__` inside the container, or
`docker exec ... pip show vllm`), do not assume it matches the host pin.

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| `kernels` | PyPI | Multi-year (HuggingFace-maintained, versions back to 0.1.7) | Not independently queried this session, but this is HuggingFace's own official package, referenced directly in the transformers docs for this exact model architecture | `github.com/huggingface/kernels` (HF org) | OK | Approved — official HF package, cited by official transformers docs, not a WebSearch-discovered name |
| `Atlas-Inference/gdn` (Hub kernel, not a PyPI package) | HuggingFace Hub kernel repo (not npm/PyPI/crates — a Hub-hosted kernel artifact loaded via `kernels`) | Not independently verified this session | N/A | Community repo, explicitly "not yet on the trusted-kernels allowlist" per official HF docs | SUS | Flagged — this is a `trust_remote_code=True` exposure to a **community, non-allowlisted** kernel; planner must add a `checkpoint:human-verify` task before enabling `use_kernels=True` in any smoke test, per the Alternatives Considered recommendation above to default `False` |

**Packages removed due to SLOP verdict:** none.
**Packages flagged as suspicious (SUS):** `Atlas-Inference/gdn` Hub kernel — not a hallucinated package (it is
real and documented by HF), but flagged because it requires `trust_remote_code=True` against non-allowlisted
community code. This is a security posture decision, not a legitimacy-of-existence concern; still gate it
behind human sign-off before use.

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────┐
                    │  HuggingFace Hub         │
                    │  Qwen/Qwen3.6-35B-A3B    │
                    └───────────┬─────────────┘
                                │ snapshot_download (resume=True)
                                ▼
                    ┌─────────────────────────┐
                    │ models/Qwen3.6-35B-A3B/  │  <- BASE-01: download + load smoke
                    │ (67 GiB bf16, VL ckpt)   │     (transformers 5.x, trust_remote_code,
                    └───────────┬─────────────┘      Qwen3_5MoeForConditionalGeneration)
                                │
                    ┌───────────▼─────────────┐
                    │ In-process Python:       │  <- BASE-02: eos/pad alignment gate
                    │ tokenizer + model.config │     assert model.config.{eos,pad}_token_id
                    │ alignment check          │     == tokenizer's; stop-token smoke generate
                    └───────────┬─────────────┘      writes output/base20/token_alignment.json
                                │ (gate must PASS before Stage 2/3 can start)
                    ┌───────────▼─────────────┐
                    │ transformers from_pretrained│  <- BASE-03: DeltaNet smoke (in-process,
                    │ use_kernels=True/False   │     no vLLM) — prefill/decode timing check
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ vLLM container (Docker)  │  <- BASE-03 (cont'd): serving smoke WITH
                    │ GB10 GPU, --gpu-memory-  │     CUDA-graph capture enabled; fallback
                    │ utilization 0.80         │     --enforce-eager if #35945 reproduces
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┴──────────────────┐
              │ BASE-04: VL merge-path round-trip   │
              │                                     │
              │  Tinker LoRA export (minimal/tiny)  │
              │           │ (target_modules resolved│
              │           │  against language_model.*)
              │           ▼                         │
              │  scripts/merge_adapter.py           │
              │  (load base cpu bf16 -> load LoRA   │
              │   -> merge_and_unload -> save)       │
              │           │                          │
              │           ▼                          │
              │  vLLM serve --language-model-only    │
              │  (merged model)                      │
              │           │                          │
              │           ▼                          │
              │  Real generation request             │
              │  (coherent, adapter-influenced output)│
              └─────────────────┬────────────────────┘
                                 ▼
                    output/base20/*.json gate receipts
                    (consumed by human review + Phase 21 kickoff gate)
```

### Recommended Project Structure

No new top-level directories needed. Follow existing conventions:

```
scripts/
├── download_model.py          # MODIFY: parameterize model name/local_dir (currently hardcoded to Qwen3-30B-A3B via config/train_config.yaml)
├── merge_adapter.py           # MODIFY: verify/adapt for model.language_model.* prefix (likely no-op per ARCHITECTURE.md §6, but must be verified, not assumed)
├── _p0_vllm_smoke_serve.py    # REUSE as-is: boot_vllm/wait_healthy/generate/stop_vllm
├── check_token_alignment.py   # NEW: Stage 1.5 eos/pad alignment gate (BASE-02) — no prior equivalent
└── smoke_base_bringup.py      # NEW (optional, or split per BASE-0x): orchestrates BASE-01/03/04 smoke sequence, writes gate receipts

config/
└── train_config_v4.yaml       # NEW (or a v4-scoped override): model.name=Qwen/Qwen3.6-35B-A3B, model.local_dir=./models/Qwen3.6-35B-A3B — do NOT overwrite config/train_config.yaml (still serves the v3.x pipeline/eval scripts)

recipes/
└── qwen3.6-35b-a3b-vllm.yaml  # NEW bf16 recipe (existing qwen3.6-35b-a3b-fp8-vllm.yaml is FP8-quantized, wrong precision for BASE-03/04 bf16 smoke — reuse its structure, not its content)

output/
└── base20/                    # NEW: gate receipts (token_alignment.json, deltanet_smoke.json, vl_merge_roundtrip.json), following output/tinker/PROMOTED_*.json / output/merge_v4_winner/merge_report.json conventions
```

### Pattern 1: Real-generation warm-up gate (carry-forward lesson 2, already proven in this repo)

**What:** Never trust `/v1/models` health-check alone as "server is ready" — always follow with one real
generation request and assert non-empty output before proceeding to the actual smoke/eval work.
**When to use:** Every vLLM boot in this phase (BASE-03, BASE-04).
**Example (already committed, reuse directly):**
```python
# Source: scripts/bench_wpbench_base_anchor.py (this repo, committed)
from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, generate, stop_vllm

boot_vllm(MODEL_DIR, CONTAINER_NAME, PORT, GPU_MEM_UTIL)
served = wait_healthy(PORT, CONTAINER_NAME)

# Phase 15 LOCKED lesson: gate on a REAL generation, not /health.
warm = generate(PORT, served,
                [{"instruction": "Reply with exactly one word: OK",
                  "source_val_idx": "warmup"}],
                max_tokens=16)
if not warm or not warm[0].strip():
    raise RuntimeError(f"Real-generation warm-up returned empty output: {warm!r}")
```

### Pattern 2: Idempotent gate receipt with fallback command on failure

**What:** Gate scripts write a small JSON receipt with a `status`/pass-fail field and, on failure, print a
concrete fallback command rather than a bare traceback.
**When to use:** BASE-04 merge verification (already implemented, see below); apply the same shape to the new
BASE-02 token-alignment script.
**Example (already committed, reuse directly):**
```python
# Source: scripts/merge_adapter.py:129-167 (this repo, committed)
assert len(wp_gen_ids) == 1, f"<wp_gen> must be single token, got {wp_gen_ids}"
print("MERGE VERIFICATION PASSED")
# ...on AssertionError:
print("Fallback: serve adapter directly with vLLM (no merge needed):")
print(f"  vllm serve {base_model} --lora-modules qwen3-wp={adapter_dir}")
sys.exit(1)
```

### Anti-Patterns to Avoid

- **Trusting merge exit-code 0 as success:** PITFALLS.md Pitfall 3 — a VL-checkpoint merge can silently
  partial-load (shape-coincidental match on some layers) and exit cleanly while producing garbage/untrained-
  looking generations. The merge script's own special-token check (`_verify_merged_model`) is necessary but
  NOT sufficient for BASE-04 — it verifies tokenizer round-trip, not that the LoRA delta actually landed on
  `model.language_model.*` weights. BASE-04 must add a served-generation comparison (adapter-influenced output
  differs observably from base-model-only output), not just re-use the existing token check.
- **Smoke-testing DeltaNet in eager mode only:** PITFALLS.md Pitfall 2 / "Looks Done But Isn't" checklist —
  vLLM issue #35945 only fires during CUDA-graph **capture**, so an eager-only smoke test gives a false pass.
  BASE-03's acceptance criterion must explicitly exercise graph capture (i.e., do NOT pass `--enforce-eager`
  on the first attempt; only fall back to it if capture fails, and record which path was taken).
- **Assuming `model.config.eos_token_id` is authoritative:** PITFALLS.md Pitfall 1 — this is the literal bug.
  Any code (in this new script or reused from elsewhere) that reads `model.config.eos_token_id` instead of
  `tokenizer.eos_token_id` / `model.generation_config.eos_token_id` will reproduce the mismatch.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Resumable HF weight download | Custom retry/chunking logic | `huggingface_hub.snapshot_download(..., resume_download=True)` — already used by `scripts/download_model.py` | Battle-tested in this exact repo already (Qwen3-30B-A3B 57 GiB download precedent); only the model name/local_dir need to change |
| vLLM boot/health-poll/generate/stop | New boot script | `scripts/_p0_vllm_smoke_serve.py` (`boot_vllm`, `wait_healthy`, `generate`, `stop_vllm`) | Already encodes the 900s boot-timeout lesson (Pitfall 3 from the OLD pitfalls doc — 30B bf16 weight load can exceed 600s), the boot-failure log capture, and the real-generation gate. The new base is ~1.2x larger (67 GiB vs 57 GiB) — reuse the same timeout constant or raise it, don't rebuild the polling loop. |
| LoRA merge + verification | New merge script | `scripts/merge_adapter.py` | PEFT's `merge_and_unload()` already handles arbitrary module-path depth (traverses the live model object) — per ARCHITECTURE.md §6 this likely needs zero code changes, only a verification pass |
| vLLM launch flags for this exact model family | Guessing flags | `recipes/qwen3.6-35b-a3b-fp8-vllm.yaml` (already committed, working, used in production for Phase 0 rubric checks) | This is a real precedent for the SAME model on the SAME hardware/container, not a hypothetical — copy its `gpu_memory_utilization`, `trust_remote_code`, `max_model_len` structure and swap `--quantization fp8` for bf16 defaults + add `--language-model-only` |

**Key insight:** Every piece of infrastructure BASE-01 through BASE-04 needs already exists in this repo in a
working, base-agnostic-enough form. The actual engineering work in Phase 20 is narrow: (1) parameterize two
scripts that currently hardcode the old base's name/paths, (2) write one genuinely new script (token
alignment — no prior equivalent), and (3) extend acceptance criteria (CUDA-graph-capture-on, served-generation
diff-check) that the old base's smoke tests didn't need because it doesn't have DeltaNet layers or a VL
wrapper.

## Common Pitfalls

(Full detail in `.planning/research/PITFALLS.md` Pitfalls 1-4, 9, 10 — all directly scoped to Phase 20 per that
document's own Pitfall-to-Phase Mapping table. Condensed here with repo-specific file pointers.)

### Pitfall 1: eos/pad token mismatch is maintainer-classified "working as intended," not a bug

**What goes wrong:** `tokenizer.eos_token_id` (248046) != `model.config.eos_token_id` (248044);
`model.config.pad_token_id` is `None`. Qwen intentionally ships `model.generation_config.eos_token_id =
[248046, 248044]` for thinking/non-thinking flexibility.
**Why it happens:** Deliberate multi-stop-token design (QwenLM/Qwen3.6 discussion #96, maintainer response
2026-03-24) — will not be "fixed" upstream.
**How to avoid:** New script sets `model.config.eos_token_id = tokenizer.eos_token_id` and
`model.config.pad_token_id = tokenizer.pad_token_id` explicitly, then asserts the match, then runs a
stop-token smoke generation (confirm generation actually stops at a natural boundary, not just that the ID
fields match — a mismatched ID could still coincidentally "work" if never hit in a short smoke prompt).
**Warning signs:** Generation runs to `max_tokens` instead of stopping naturally.

### Pitfall 2: vLLM CUDA-graph capture crashes on DeltaNet layers — distinct from the platform-level GB10/sm_121 support gap

**What goes wrong:** vLLM issue #35945 — `AssertionError: assert num_cache_lines >= batch` in
`causal_conv1d_update` during CUDA-graph capture on Gated-DeltaNet layers. Separately, the stock NGC vLLM
container ships vLLM 0.13.0 (predates Qwen3.5/3.6 support) — this repo already avoids that trap by building/
running vLLM in a project-controlled container (`ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest`), so this
is a carried-forward non-issue, not a new risk to solve.
**How to avoid:** Run the BASE-03 smoke WITH CUDA-graph capture enabled (default vLLM behavior — don't pass
`--enforce-eager` preemptively); if it crashes, document the fallback (`--enforce-eager`) and the throughput
cost, don't block the phase on an upstream fix. Set `--gpu-memory-utilization 0.80` (community-verified stable
value for GB10 long-running sessions, per `adadrag/qwen3.5-dgx-spark`), not the vLLM default of 0.90.
**Warning signs:** Server crash/hang specifically during graph-capture phase of startup, not at first request.

### Pitfall 3: VL checkpoint has two different key-prefix conventions (merge-time vs. serve-time)

**What goes wrong:** `merge_adapter.py` must target `model.language_model.*` keys (VL wrapper prefix,
confirmed live against `model.safetensors.index.json`). Separately, at SERVE time, vLLM's
`--language-model-only` flag does its OWN remap (`model.layers.*` -> `language_model.model.layers.*`). If
merge and serve assume the same convention, weights can silently partial-load.
**How to avoid:** Test the FULL merge -> serve round-trip with a real generation comparison (adapter-merged
output vs base-only output should differ observably), not just "merge exits 0" or "serve boots healthy"
independently.
**Warning signs:** Merge succeeds, serve boots, but generations look base-model-like (adapter effect invisible)
— the classic silent-partial-load symptom.

### Pitfall 4: `trust_remote_code=True` required on BOTH model and tokenizer/processor; intermittent multi-worker init race

**What goes wrong:** A plain `AutoTokenizer.from_pretrained()` without `trust_remote_code=True` fails or
mis-loads custom tokenizer logic. Separately, vLLM issue #40249 reports intermittent `KeyError:
'qwen3_5_moe'` from `transformers.CONFIG_MAPPING` during multi-worker init.
**How to avoid:** Set `trust_remote_code=True` everywhere in the BASE-01 bring-up scripts (model AND
tokenizer/processor). This project's GB10 deployment is single-GPU/single-worker (per `dgx_toolbox.yaml`
`tensor_parallel: 1`), which reduces but does not eliminate the race surface — no TP>1 retry logic needed for
Phase 20 given the current single-worker deployment, but note the race exists if TP is ever raised later.

## Runtime State Inventory

Phase 20 is a greenfield bring-up (new model download, new smoke scripts), not a rename/refactor/migration —
this section is omitted per the standard trigger condition (no string-rename or data-migration surface in this
phase's scope). The old base's files (`models/Qwen3-30B-A3B/`, `config/train_config.yaml`) are left untouched;
the new base gets parallel, non-colliding paths (`models/Qwen3.6-35B-A3B/`, a new `config/train_config_v4.yaml`
or equivalent override — do not edit `config/train_config.yaml` in place, since v3.x scripts/tests still read
it).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Disk space (`/`, `models/` mount) | BASE-01 download (67 GiB) | Yes [VERIFIED: `df -h` this session] | 2.6 TB free of 3.7 TB | — (ample headroom; 67 GiB is <3% of free space) |
| `transformers` (host) | BASE-01 import smoke | Yes [VERIFIED: `python3 -c "import transformers"` this session] | 5.3.0 | — |
| `vllm` (host pip) | — | No (by design) | — | Runs inside `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest` Docker container via `dgx_toolbox.py`, not host pip — confirm container's version as part of the smoke test itself |
| `peft` (host) | BASE-04 merge | Assumed present (imported by `scripts/merge_adapter.py`, which is a working, previously-exercised script) | not independently re-verified this session | — |
| `huggingface_hub` (host) | BASE-01 download | Assumed present (imported by `scripts/download_model.py`) | not independently re-verified this session | — |
| Docker + GPU passthrough | BASE-03/04 vLLM smoke | Assumed present (existing `serve_*_vllm.sh` scripts already depend on this; not re-probed this session — out of scope for a research pass, verify at Phase 20 execution start) | — | — |
| `kernels` package (PyPI) | BASE-03 `use_kernels=True` path (discretionary) | Not yet installed; confirmed installable [VERIFIED: `pip index versions kernels` this session, 0.16.0 live] | 0.16.0 available | If not installed, defaults to `use_kernels=False` (PyTorch fallback path) — this is itself the recommended default per Alternatives Considered |

**Missing dependencies with no fallback:** none identified.

**Missing dependencies with fallback:** `kernels` package — fallback is simply not installing it and running
the (recommended-default) PyTorch fallback path.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (project-standard; `tests/` dir with 20+ existing test files, `tests/conftest.py` shared fixtures) [VERIFIED: repo] |
| Config file | none found at repo root (no `pytest.ini`/`pyproject.toml` `[tool.pytest]` section) — tests currently run via bare `pytest tests/` from repo root; `.pytest_cache/` present confirming prior runs |
| Quick run command | `pytest tests/test_<new_file>.py -x` |
| Full suite command | `pytest tests/` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BASE-01 | Model downloads (resume-safe) and loads with correct class | unit (config/path logic) + manual-only (actual 67 GiB download + GPU load is not CI-automatable) | `pytest tests/test_download_model_v4.py -x` (path/config logic only); full download+load is a smoke script, not a pytest test | ❌ Wave 0 — new test file needed for the config/path-resolution logic mirroring `tests/test_prepare_tokenizer.py`'s pattern |
| BASE-02 | eos/pad alignment assert + stop-token smoke | unit (assert logic, mockable) + manual smoke (real stop-token generation needs GPU) | `pytest tests/test_check_token_alignment.py -x` | ❌ Wave 0 — new test file, new module under test |
| BASE-03 | DeltaNet smoke with CUDA-graph capture | manual-only (requires actual GB10 GPU + vLLM container; not unit-testable) | N/A — smoke script run manually/via skill, receipt written to `output/base20/deltanet_smoke.json` | N/A (justified manual-only: GPU-dependent, matches existing project pattern — `scripts/_p0_vllm_smoke_serve.py` has no accompanying pytest suite either, it's exercised via smoke-run scripts) |
| BASE-04 | VL merge -> serve round-trip, real generation | manual-only (same GPU/Docker dependency) + unit (merge script's existing `_verify_merged_model` logic, if extracted to be testable) | Existing `scripts/merge_adapter.py` has no dedicated test file either (verified: no `tests/test_merge_adapter.py` found) — consistent with project convention of GPU-touching scripts being smoke-tested, not unit-tested | N/A (consistent with existing `merge_adapter.py` precedent — no gap to fill beyond what BASE-03 already needs) |

### Sampling Rate

- **Per task commit:** `pytest tests/test_<new_file>.py -x` for the two testable pieces (BASE-01 config
  logic, BASE-02 alignment logic).
- **Per wave merge:** `pytest tests/` (full suite, confirms no regression to the 20+ existing test files from
  the v3.x pipeline, which Phase 20 must not touch).
- **Phase gate:** All four gate receipts (`output/base20/*.json`) show pass status, reviewed by a human before
  Phase 21 (SFT) is allowed to start — matches BASE-02's explicit "blocks Stage 2/3 on failure" requirement.

### Wave 0 Gaps

- [ ] `tests/test_download_model_v4.py` — covers BASE-01 config/path-resolution logic (mirror
      `tests/test_prepare_tokenizer.py`'s structure: config loading, path resolution, idempotency-check logic
      — NOT the actual network download, which is unit-untestable)
- [ ] `tests/test_check_token_alignment.py` — covers BASE-02's new alignment-check module (mockable: feed a
      fake tokenizer/config pair with mismatched IDs, assert the check catches it; feed a matched pair, assert
      it passes)
- [ ] No pytest gaps for BASE-03/BASE-04 — GPU-dependent smoke scripts are consistently NOT unit-tested
      elsewhere in this repo (`_p0_vllm_smoke_serve.py`, `merge_adapter.py` both lack dedicated test files);
      Phase 20 should follow the same convention rather than introduce a new testing pattern for these two
      requirements specifically.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No auth surface in this phase — local model download/smoke, no user-facing auth |
| V3 Session Management | No | N/A |
| V4 Access Control | No | N/A — single-operator local infra |
| V5 Input Validation | Marginal | eos/pad token-ID values and tokenizer special-token IDs are validated (asserted) before use — already the core of BASE-02 |
| V6 Cryptography | No | N/A — no crypto surface introduced |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| `trust_remote_code=True` executing arbitrary code from a HuggingFace repo at load time | Tampering / Elevation of Privilege | Standard mitigation is trusting only vetted repos — this project already accepts this risk for the base model itself (Qwen's own repo, official/allowlisted); the NEW exposure this phase introduces is the `Atlas-Inference/gdn` community kernel (not yet on HF's trusted-kernels allowlist) if `use_kernels=True` is chosen. Mitigation: default `use_kernels=False` (see Alternatives Considered); if `True` is chosen, require explicit human sign-off (`checkpoint:human-verify`) before enabling, and document the acceptance in the gate receipt. |
| Downloading 67 GiB of untrusted binary weight data over the network | Tampering | `snapshot_download` already uses HF Hub's standard integrity mechanism (file hashes in the repo's own metadata); no additional action needed beyond what `scripts/download_model.py` already does |

## Code Examples

### Reused: real-generation warm-up gate

```python
# Source: scripts/_p0_vllm_smoke_serve.py + scripts/bench_wpbench_base_anchor.py (this repo)
from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, generate, stop_vllm, VllmBootTimeout

boot_vllm(model_dir="models/Qwen3.6-35B-A3B", name="base20-deltanet-smoke", port=8020, gpu_mem_util=0.80)
served = wait_healthy(8020, "base20-deltanet-smoke", timeout=1200)  # raise timeout: 67 GiB > old base's 57 GiB
warm = generate(8020, served, [{"instruction": "Reply with exactly one word: OK", "source_val_idx": "warmup"}], max_tokens=16)
assert warm and warm[0].strip(), f"empty warm-up generation: {warm!r}"
```

### Reused: LoRA merge with fallback

```python
# Source: scripts/merge_adapter.py (this repo)
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig
model = AutoModelForCausalLM.from_pretrained(local_dir, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
peft_config = PeftConfig.from_pretrained(adapter_dir)
model = PeftModel.from_pretrained(model, adapter_dir, config=peft_config)
merged_model = model.merge_and_unload()
```
Note the `trust_remote_code=True` addition needed here — the existing call (line 87-91) does not currently
pass it, because the old base doesn't need it. This is a required, not optional, change for BASE-04.

### New pattern needed: eos/pad alignment check (no prior repo example — write per PITFALLS.md Pitfall 1)

```python
# Pattern to implement (not yet in repo) — based on PITFALLS.md Pitfall 1 guidance
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)

# The bug: model.config.eos_token_id/pad_token_id may not match tokenizer's.
model.config.eos_token_id = tok.eos_token_id
model.config.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

assert model.config.eos_token_id == tok.eos_token_id
assert model.config.pad_token_id is not None

# Then: real stop-token smoke generation, assert it stops before max_tokens on an obvious prompt.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Unfused per-expert weight tensors (128 experts x 3 tensors/layer) | Fused stacked tensor per layer (256 experts, 2 tensors/layer: `experts.gate_up_proj`, `experts.down_proj`) | Qwen3.6-35B-A3B release (this base) vs Qwen3-30B-A3B (old base) | Does not affect Phase 20 directly (Sieve profiler is Phase 25's concern) — noted here only because BASE-01's load smoke should not assume the old base's tensor layout if any code path inspects raw state_dict keys |
| Flat `model.layers.*` key space | `model.language_model.*` wrapper prefix (VL checkpoint) | Same | Directly affects BASE-04 — this is the whole point of the merge-path check |
| `--enforce-eager` as a default safety choice | CUDA-graph capture attempted first, `--enforce-eager` as documented fallback only | This phase's own acceptance-criteria design (per Pitfall 2's "looks done but isn't" guidance) | A smoke test that defaults to eager mode gives a false pass on issue #35945 |

**Deprecated/outdated:** N/A — no deprecated approach within this phase's scope; all findings are additive
smoke-test requirements on top of an otherwise-reused harness.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `docker` + GPU passthrough is currently working on this host (not re-probed this session; inferred from existing working `serve_*_vllm.sh` scripts and recent commit history showing `Judge vLLM container restarted successfully`) | Environment Availability | If Docker/GPU access has regressed since the last vLLM run, BASE-03/04 smoke will fail for an environment reason unrelated to the new base — planner should have execution start with a trivial `docker ps`/`nvidia-smi` check before the real smoke tests, not assume it silently |
| A2 | `peft` and `huggingface_hub` host-pip versions are adequate for the new base (not independently version-checked this session, only inferred from `merge_adapter.py`/`download_model.py` being working, previously-exercised scripts) | Standard Stack | If PEFT's target-module resolution has an undocumented version floor for VL-wrapped models, BASE-04 could fail for a dependency-version reason; the phase's own BASE-01 import-smoke step should be extended to also print `peft.__version__`/`huggingface_hub.__version__` for the record |
| A3 | The container `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest` currently resolves to a vLLM build that meets the `>=0.19.0` recommended floor (STACK.md) — not independently checked this session; the tag is `:latest` (nightly), so it plausibly already exceeds the floor, but this must be confirmed as a BASE-03 smoke step, not assumed from the tag name | Standard Stack | If the nightly image has NOT kept pace, BASE-03 could hit a "not recognized architecture" failure unrelated to the DeltaNet-specific risks this research focuses on — first BASE-03 smoke-test action should log `vllm.__version__` from inside the container |

**If this table is empty:** N/A — see rows above; all three are execution-environment assumptions carried
forward from "this infra worked for the old base," not claims about the new base's architecture (those are all
`[VERIFIED]`/`[CITED]` per the upstream research docs).

## Open Questions (RESOLVED)

1. **How to cheaply exercise BASE-04 (VL merge path) before any real Stage 2/3 SFT run exists?**
   RESOLVED: 20-04-PLAN.md Task 1 implements the recommended minimal real Tinker run (primary) with local zero-init adapter fallback (reduced-confidence flag).
   - What we know: The roadmap task framing suggests either "a tiny throwaway Tinker run" or "a zero-init
     adapter constructed locally." ARCHITECTURE.md §6 identifies the actual unresolved risk as: "if Tinker's
     LoRA was trained with target modules resolved against `model.layers.N...` [old convention], merge will
     silently fail to find matching modules" — this is specifically about **Tinker's own export-side
     behavior**, which only a real (even if tiny/cheap) Tinker run can validate. A locally-built zero-init
     adapter (constructed directly against known `model.language_model.*` module names) tests
     `merge_adapter.py`'s own correctness but does NOT test whether Tinker's export naturally targets the
     right prefix.
   - What's unclear: Tinker's minimum billable unit for a LoRA training run against this specific model/size
     class (STACK.md's $0.36/$0.54/$1.07 prefill/sample/train pricing is per-token/per-request, not a fixed
     minimum-run cost — the cheapest possible real run's total dollar cost wasn't independently derived this
     session).
   - Recommendation: **Do a minimal real Tinker run** — smallest possible LoRA config (r=1 or a tiny r, 1
     training step or a handful of steps, a trivial 1-2 example dataset) purely to exercise the export path
     and log the actual target-module names Tinker attaches LoRA to (per PITFALLS.md Pitfall 10's own
     recommendation: "print/log the actual list of modules LoRA attached to"). This resolves BOTH BASE-04's
     merge-path risk AND Pitfall 10's LoRA-target-naming risk in one cheap run, budget order-of-magnitude
     cents to low-single-digit dollars given the per-unit pricing and a trivial dataset/step-count — well
     under the $2/run full-SFT anchor. If v4.0 sign-off happens after 2026-07-17, this cost has the ~10%
     train-price/~50% prefill-price rise applied, still trivial at this scale. Fall back to a locally-built
     zero-init adapter only if this minimal Tinker run turns out to be blocked for an unrelated reason
     (account/API access issue) — that fallback validates `merge_adapter.py` alone, not the full chain, and
     should be flagged as a reduced-confidence pass if used.

2. **Does the Phase 0 FP8 recipe's container image (`ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest`) need
   an explicit vLLM version bump, or does `:latest`/nightly already clear the `>=0.19.0` floor?**
   RESOLVED: 20-03-PLAN.md Task 2 logs the resolved vLLM version as the FIRST smoke action and asserts >=0.19.0 in the gate receipt.
   - What we know: The recipe is real, committed, and was used successfully in production (Phase 0 rubric
     checks) for the SAME model family on the SAME container tag.
   - What's unclear: Exact vLLM version currently resolved by that `:latest` tag as of Phase 20's execution
     date (tags drift; "nightly" as of Phase 0's commit date may differ from "nightly" as of Phase 20
     execution).
   - Recommendation: First BASE-03 action is to log the resolved vLLM version from inside the container before
     running any DeltaNet-specific test — cheap, and turns an implicit assumption into a recorded fact in the
     gate receipt.

3. **Does `config/train_config.yaml` need a v4-specific sibling file now, or can Phase 20 defer that to Phase
   21?**
   RESOLVED: 20-01-PLAN.md Task 1 creates `config/train_config_v4.yaml` + `--config-path` flag now; v3.x config untouched (git diff --exit-code asserted).
   - What we know: `download_model.py` and `merge_adapter.py` both read `model.local_dir`/`model.name` from
     `config/train_config.yaml` via `load_config()`. Editing that file in place would break v3.x
     scripts/tests that still reference the old base's paths (the merge report/tinker-export receipts under
     `models/tinker_export/`, `output/tinker/`, etc. all assume old-base structure and must remain valid/
     inspectable).
   - What's unclear: Whether Phase 20's scripts should accept a `--config` override flag (both scripts already
     support a `config_path` parameter/`--adapter-dir`/`--output-dir` CLI args, so this is likely a small,
     low-risk change) vs. a wholly separate `train_config_v4.yaml`.
   - Recommendation: Add a `config/train_config_v4.yaml` (copy + edit `model.name`/`model.local_dir`/
     `lora.target_modules` for the new prefix) and pass `--config-path` (a new, small CLI flag) to
     `download_model.py`/`merge_adapter.py` rather than mutating the shared file — lowest-risk option, keeps
     v3.x fully reproducible.

## Sources

### Primary (HIGH confidence)

- `.planning/research/STACK.md`, `.planning/research/ARCHITECTURE.md`, `.planning/research/PITFALLS.md`
  (all dated 2026-07-12, this repo) — synthesize live-fetched HF `config.json`/`model.safetensors.index.json`,
  vLLM recipe docs, and primary-source GitHub issues; not re-fetched this session, reused per task instruction
  ("do NOT re-do web research already in .planning/research/")
- `scripts/download_model.py`, `scripts/merge_adapter.py`, `scripts/_p0_vllm_smoke_serve.py`,
  `scripts/bench_wpbench_base_anchor.py`, `recipes/qwen3.6-35b-a3b-fp8-vllm.yaml`,
  `scripts/serve_v4_judge_vllm.sh`, `config/dgx_toolbox.yaml`, `config/train_config.yaml` — read directly
  this session [VERIFIED: repo]
- `output/tinker/PROMOTED_v1.3.json`, `output/merge_v4_winner/merge_report.json` — read directly this session
  for gate-receipt schema precedent [VERIFIED: repo]
- Local shell verification this session: `transformers==5.3.0` importable; `pip index versions kernels` ->
  `0.16.0` live on PyPI; `df -h /` -> 2.6 TB free of 3.7 TB

### Secondary (MEDIUM confidence)

- `.planning/V4-RERUN-ROADMAP.md`, `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md` (Phase 20 section) —
  locked planning artifacts defining scope/requirements, read directly this session

### Tertiary (LOW confidence)

- None — this phase's research surface was fully covered by the existing HIGH/MEDIUM-confidence research docs
  and direct repo inspection; no new WebSearch/WebFetch was needed or performed this session.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all library/version claims trace to `.planning/research/STACK.md` (live-fetched
  primary sources) or this session's own local verification (`transformers` import, `pip index versions`)
- Architecture: HIGH — repo file paths, existing script behavior, and gate-receipt conventions confirmed by
  direct `Read`/`Bash` inspection this session; architecture facts (key prefixes, layer counts) trace to
  `.planning/research/ARCHITECTURE.md`'s live HF-source verification
- Pitfalls: HIGH — all pitfalls trace to `.planning/research/PITFALLS.md`'s primary-source GitHub issue/
  discussion citations, mapped onto specific repo files this session

**Research date:** 2026-07-13
**Valid until:** 30 days (stable domain — repo conventions and the upstream research docs are not expected to
shift quickly; re-verify the vLLM container's resolved version and the `kernels`/PyPI package state at actual
execution time regardless, since those are fast-moving/nightly-tagged surfaces)
