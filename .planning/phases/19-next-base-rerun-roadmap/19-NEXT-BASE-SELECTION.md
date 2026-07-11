# Next-Base Selection — v4.0 Rerun Candidate (NEXT-01)

**Decision date:** 2026-07-11
**Status:** LOCKED
**Locked base:** `Qwen/Qwen3.6-35B-A3B`

This doc satisfies NEXT-01: verify and lock the base for the v4.0 PIPELINE.md rerun. All claims below
were re-verified live on 2026-07-11 (not only inherited from `19-RESEARCH-BASESCAN.md`) via direct
`curl` fetches of HF model card raw READMEs, the HF models-search API, and Tinker's live models table.
Source URLs and raw evidence are cited per claim. No weights were downloaded.

---

## Verification method

- `curl https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/README.md` — architecture, license, benchmark table.
- `curl https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/README.md` — fallback cross-check (same family).
- `curl https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/README.md` — current-base cross-check (params, license).
- `curl https://huggingface.co/api/models?search=Qwen3.6-35B-A3B` — ecosystem scan (GGUF/AWQ/NVFP4/MLX builds, Unsloth presence).
- `curl https://tinker-docs.thinkingmachines.ai/tinker/models/` — live Tinker-supported-model table (tinker-id, context cap, LoRA pricing).

## Axis 1 — Architecture match to the pipeline

**Verified (source: `Qwen/Qwen3.6-35B-A3B/README.md`, fetched 2026-07-11):**

- 35B total / 3B activated params, 40 layers.
- Hidden layout: `10 x (3 x (Gated DeltaNet -> MoE) -> 1 x (Gated Attention -> MoE))` — hybrid linear-attention
  + full-attention stack, confirming the RESEARCH-BASESCAN architecture-delta claim verbatim.
- MoE: 256 experts, **8 routed + 1 shared** activated per token, expert intermediate dim 512.
- Token embedding 248,320 (padded) — task-token extension (`<wp_gen>`, `<wp_judge>`) follows the same
  vocab-append pattern used on Qwen3-30B-A3B; transferable.
- Context: 262,144 native, extensible to 1,010,000 via YaRN.

**PIPELINE.md fit:** task-token MoE routing is preserved (Prerequisites: "A task-token MoE base"). The
architecture is NOT a drop-in match to the current 128-expert/no-shared/uniform-attention base — see the
two LOCKED architecture-delta work items carried into `V4-RERUN-ROADMAP.md` (Sieve/protected-mask tooling
must handle mixed DeltaNet/Attention layers + an always-on shared expert; eos/pad token-ID alignment).

## Axis 2 — GB10 121 GB memory budget

**Verified arithmetic (bf16 = 2 bytes/param, matches the exact Qwen3-30B-A3B precedent: 30.5B x 2 /
1024^3 = 56.8 GiB, which is the measured figure in `output/packaging/MODEL_CARD.md`):**

| Checkpoint | Params | bf16 size | Two-checkpoint pair (gen+judge, bf16) |
|---|---|---|---|
| Qwen3-30B-A3B (current) | 30.5B | 56.8 GiB | 113.6 GiB (fits 121 GB with ~7 GiB headroom) |
| Qwen3.6-35B-A3B (candidate) | 35B | **65.2 GiB** | **130.4 GiB — exceeds the 121 GB host** |

**Finding (new, not in RESEARCH-BASESCAN):** unlike the current base, the Qwen3.6 pair does NOT fit
concurrently at bf16 on GB10. This makes Stage 5 quantization a **hard prerequisite for concurrent
pair-serving**, not an optional size optimization. Scaling the v3.0 Q8 GGUF ratio (30.2 GiB / 56.8 GiB =
53.2% of bf16) to the candidate: ~34.7 GiB/checkpoint, ~69.4 GiB for the pair at Q8 — comfortably inside
121 GB. This is captured as a memory-driven (not quality-driven) packaging requirement in
`V4-RERUN-ROADMAP.md` Stage 5. Sequential (load-one-at-a-time) serving is the bf16 fallback if a
pre-quantized dev loop is needed before Stage 5 runs.

## Axis 3 — Tinker / Unsloth / vLLM / llama.cpp support

**Tinker (verified live, `tinker-docs.thinkingmachines.ai/tinker/models/`, fetched 2026-07-11):**
`Qwen/Qwen3.6-35B-A3B` is a live row — tinker-id `Qwen/Qwen3.6-35B-A3B`, type "Hybrid + Vision", arch MoE,
size Medium, **context cap 64K** (below the model's native 262K — a Tinker-side ceiling on SFT context,
not a model limit; WordPress function-level SFT examples are well inside 64K so this is not blocking),
LoRA pricing train/sample/eval $0.36 / $0.89 / $1.07 (same per-unit price tier as the current base's
30B-A3B-Base row, i.e. no cost-class jump). Fallback `Qwen/Qwen3.5-35B-A3B-Base` is also live in the same
price tier, listed as "Base" type only (no dedicated instruct-tuned row) — a minor tooling asymmetry noted
for the fallback trigger below, not blocking since PIPELINE.md SFT starts from base weights either way.

**Unsloth (verified via HF ecosystem scan):** `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`,
`unsloth/Qwen3.6-35B-A3B-NVFP4`, `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`, `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit`
are all live org repos — Unsloth has first-party day-of-release conversions for this exact model, matching
the RESEARCH-BASESCAN "Unsloth fine-tune guide, router training disabled by default" claim.

**vLLM (verified, `Qwen/Qwen3.6-35B-A3B/README.md`):** `vllm>=0.19.0` is the vendor-recommended minimum
(`uv pip install vllm --torch-backend=auto`) — resolves the RESEARCH-BASESCAN "vLLM v0.19.0+" claim from
inferred to confirmed. **In-repo precedent:** `CHANGELOG.md` D-03 entry already runs
`Qwen/Qwen3.6-35B-A3B` as the local vLLM CoT-generation endpoint on this exact GB10 host — the base
already loads and serves here today, independent of this verification pass.

**llama.cpp / GGUF (verified via HF ecosystem scan):** `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF` (bartowski is
the standard llama.cpp community quantizer used for the v3.0 Q8 ship tier) and the Unsloth GGUF repos
above are both live, plus NVIDIA's official NVFP4 build (`nvidia/Qwen3.6-35B-A3B-NVFP4`, a Blackwell/GB10
validation signal) and AWQ (`QuantTrio/Qwen3.6-35B-A3B-AWQ`). The full Stage-5 quantization ladder
(Q8/Q6/Q5/Q4 GGUF, AWQ) has ecosystem support at launch.

**Resolves RESEARCH-BASESCAN "Unverified" list:** Tinker support — VERIFIED (live model-table row).
Unsloth fine-tune guide — VERIFIED (org repos exist; guide text itself not re-fetched, low-risk since
first-party GGUF conversions already exist). DeltaNet ops on GB10 aarch64 — still inferred-OK (no direct
op-level benchmark available pre-download; carried as a Stage-1/2 smoke-test item in the roadmap, not a
lock blocker since D-03 already runs successful vLLM inference with this architecture on this host).
Terminal-Bench "matches Opus 4.5" single-source framing — dropped from the rationale; the verified figure
used below is Terminal-Bench 2.0 51.5 on its own terms, no comparison claim carried forward.

## Axis 4 — License

**Verified (source: `Qwen/Qwen3.6-35B-A3B/README.md` frontmatter, fetched 2026-07-11):**
`license: apache-2.0`, `license_link: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE`.
Same permissive license class as the current base (Qwen3-30B-A3B, also verified `apache-2.0` this session)
— no license-driven change to downstream fine-tune/redistribution rights (the project already publishes
its derived pair under Apache-2.0 on HuggingFace, per Phase 18).

## Axis 5 — Coding-benchmark deltas vs Qwen3-30B-A3B

**Verified (source: `Qwen/Qwen3.6-35B-A3B/README.md` benchmark tables, fetched 2026-07-11):**

| Benchmark | Qwen3.6-35B-A3B | Qwen3-30B-A3B (current base) |
|---|---|---|
| SWE-bench Verified | **73.4** | not published on the base's own HF card (RESEARCH-BASESCAN flagged the 50.3% figure as possibly the Coder variant; re-checked `Qwen3-30B-A3B/README.md` this session — confirmed NO SWE-bench/LiveCodeBench rows exist on that card at all, so no apples-to-apples base-card number exists to diff against) |
| LiveCodeBench v6 | **80.4** | not published on the base's own HF card |
| Terminal-Bench 2.0 | **51.5** | not published on the base's own HF card |

**Disposition:** the "Unverified" SWE-bench-50.3%-may-be-Coder-variant item is now resolved as
**confirmed-absent** — Qwen3-30B-A3B's own model card carries no coding-benchmark table to diff against,
so no fabricated delta is used here. The qualitative signal instead: Qwen3.6-35B-A3B publishes strong
absolute coding-agent numbers (SWE-bench Verified 73.4, comparable to Claude-Sonnet-4.5-class scores shown
in the same table) at 3B active params, and the project's own judge-rho ceiling problem
(`output/relabel/gap_closure_summary.json`) is a reasoning/calibration-quality wall, not a raw coding-skill
wall — coding-agent strength is a supporting signal, not the primary lock criterion. The primary criterion
is architecture concentration (256 experts vs 128 — more redundancy headroom for the Sieve/prune gates)
and reasoning capability class, both qualitatively favor 3.6 over 3.0 per the vendor's own comparison table
(3.6 beats the 3.5 predecessor on every coding-agent row shown).

---

## Lock decision

**LOCKED: `Qwen/Qwen3.6-35B-A3B`.** All five rationale axes are covered with live-verified, source-cited
evidence. No axis failed verification (existence confirmed 200 OK + valid README, license Apache-2.0,
Tinker path live, vLLM/llama.cpp/Unsloth ecosystem live). The one real complication found during
verification — bf16 pair-serving no longer fits GB10 concurrently — is not a lock blocker; it converts
Stage 5 quantization from a size optimization into a memory-driven hard prerequisite, which
`V4-RERUN-ROADMAP.md` documents explicitly.

### Pre-authorized fallback

**`Qwen/Qwen3.5-35B-A3B`** — same 35B/3B, 256-expert, hybrid-attention architecture family (verified
live: identical param counts, identical MoE config, identical Apache-2.0 license, same 262,144/1,010,000
context). Older/safer twin with more third-party REAP-pruned variants in the wild (per
RESEARCH-BASESCAN). **Fallback trigger:** switch to this base only if Qwen3.6-35B-A3B fails at
download/load time on GB10 (a live-verification blind spot — HF card claims cannot confirm aarch64/Blackwell
runtime behavior without downloading weights, which this phase explicitly does not do) or if Tinker
deprecates the 3.6 row before v4.0 execution starts. Tooling note: Tinker currently lists the fallback
only as `Qwen/Qwen3.5-35B-A3B-Base` (no dedicated instruct row) — acceptable since PIPELINE.md SFT starts
from base weights regardless of which twin is used.

### Documented alternative (not selected)

**`Qwen/Qwen3.6-27B` (dense)** — capability wildcard per RESEARCH-BASESCAN (SWE-bench Pro 53.5 beats the
397B MoE tier), simplest fine-tune story, ~50 GB bf16. **Not equated with the MoE candidates**: dense
breaks the RL/Sieve/prune conditional-gate assumptions as designed (Sieve and expert-drop have no meaning
without experts; prune becomes pure weight-norm surgery on a dense stack) and turns task-token routing
into two separate LoRA adapters instead of one router-conditioned base. This is a methodology change, not
a base swap, and is out of scope for a same-methodology PIPELINE.md rerun. Recorded here as a live option
if a future milestone explicitly wants to test the dense-methodology hypothesis, not as part of this lock.

---

## Sources

- https://huggingface.co/Qwen/Qwen3.6-35B-A3B (README.md, fetched 2026-07-11)
- https://huggingface.co/Qwen/Qwen3.5-35B-A3B (README.md, fetched 2026-07-11)
- https://huggingface.co/Qwen/Qwen3-30B-A3B (README.md, fetched 2026-07-11)
- https://huggingface.co/api/models?search=Qwen3.6-35B-A3B (ecosystem scan, fetched 2026-07-11)
- https://tinker-docs.thinkingmachines.ai/tinker/models/ (live pricing/support table, fetched 2026-07-11)
- `.planning/phases/19-next-base-rerun-roadmap/19-RESEARCH-BASESCAN.md` (prior-session shortlist, cross-checked)
- `output/packaging/MODEL_CARD.md` (current-base bf16 size precedent: 56.8 GiB)
- `CHANGELOG.md` D-03 entry (in-repo precedent: Qwen3.6-35B-A3B already serves via vLLM on this GB10 host)
