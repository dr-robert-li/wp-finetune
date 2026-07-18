# Phase 27: Packaging & Publication Refresh - Research

**Researched:** 2026-07-17
**Domain:** GGUF quantization of a physically-pruned MoE checkpoint (llama.cpp), cascading compression gates, HuggingFace publication with operator-only model card
**Confidence:** HIGH (toolchain/conversion-path claims verified by reading installed llama.cpp source; project history/prior-art claims verified against on-disk artifacts; a few pipeline-level projections are explicitly flagged LOW/ASSUMED)

## Summary

Phase 27 packages one model, not two. The v4.0 milestone retired the generation role as a
deliverable in 2026-07-15 (every fine-tuned gen candidate regressed below the raw Qwen3.6-35B-A3B
base; `PROJECT.md:14` records "The generation half was **retired as a deliverable**"). The project
already renamed itself "Qwen 3 WP Judge" and the current `README.md` explicitly tells operators to
use a current base model for generation. The ROADMAP's Phase 27 success-criteria wording ("Q8 GGUF
**pair** conversion", "134 GiB bf16 **pair**") is stale, inherited from the v3.0 packaging template
before the judge-only pivot. This is the single most important scoping correction for planning:
**Phase 27 converts and ships one GGUF artifact** — the pruned v4 judge at
`models/Qwen3.6-35B-A3B-judge-v4-pruned-k224` (60 GB bf16 on disk, verified by `du -sh`) — not a
gen+judge pair. Flag this explicitly in the plan so tasks don't get sized for two models.

The 33.6 GiB Q8 figure in `selection_v4.json` is an explicit **linear-scaling projection**
("Projection is linear-scaling; real Q8 size measured at Phase 27"), not a measurement. Phase 27's
first job is to run the actual conversion and record the real number.

The installed llama.cpp toolchain (`~/llama.cpp`, commit `8f114a9`, build tag range `b9999`-`b10004`,
built 2026-07-10) is verified **ahead of** the `>=b9180` floor named in PKG4-01/ROADMAP — b9180 is
specifically where MTP (multi-token-prediction) conversion support for Qwen3.5/3.6 landed, which
matters here because the pruned checkpoint retains its untouched `mtp.*` tensor (1 MTP layer,
`mtp_num_hidden_layers: 1` in config). Reading the actual conversion code
(`~/llama.cpp/conversion/qwen.py`, `conversion/base.py`) confirms the 224-expert non-standard MoE
topology converts correctly with **no code changes needed**: expert count is read generically from
`config.json`'s `text_config.num_experts` (flattened into top-level hparams before GGUF metadata is
written) and the stacked `gate_up_proj`/`down_proj` tensor-splitting path operates on the tensor's
actual on-disk shape, not a hardcoded 256. The C++ loader's only expert-count assertions
(`n_expert <= LLAMA_MAX_EXPERTS(512)`, `n_expert_used <= n_expert`) are satisfied trivially by
224/8, and the Qwen3.6 architecture carries no `n_group` expert-grouping key that could produce a
divisibility landmine at 224. This is a genuinely non-trivial thing to get wrong on a pruned MoE and
the code reading resolves it: **no runtime breakage from the reduced expert count is expected**, but
the existing block-count sanity check (`scripts/eval4_ext_gguf_convert.sh`) should be extended with
an **expert-count sanity check** (GGUF `qwen35moe.expert_count` metadata field vs
`config.json text_config.num_experts`) as cheap, high-value insurance — this pattern doesn't exist
yet and Phase 26's physical surgery is exactly the kind of change a silent off-by-N would slip
through undetected.

Prior art fully covers the mechanics needed here: `scripts/eval4_ext_gguf_convert.sh` (HF→GGUF
conversion + block-count sanity gate, already exercises the merged v4 judge on this exact llama.cpp
build), `scripts/_pkg_gguf_eval_run.sh` (serve GGUF via `llama-server --parallel N`, capture judge
responses, score rho — this already IS the "concurrent-sequence CUDA-backend smoke" PKG4-01 asks
for), `scripts/relabel/eval_relabel.py` (rho scoring), `scripts/_pub03_upload.sh` (sequential
per-file HF upload with stall-watchdog, because `hf upload-large-folder` deadlocked twice on this
host), and the PUB-03 round-trip receipt shape (`output/packaging/pub03_validation_receipt.json`:
download via API listing, load the GGUF, run one judge smoke + one gen smoke). All of these need
light adaptation (single judge-only target instead of a pair, k=224 in the sanity check, updated
repo names) rather than rewriting. `scripts/run_packaging_recipe.md` documents the quantize-ladder
recipe (`llama-quantize MODEL.f16.gguf MODEL.Q8_0.gguf Q8_0`, etc.) and the pre-registered ±2pp stop
rule from Gate 1.

The existing `output/packaging/hf_cards/judge_gguf_README.md` (v3 card, currently live on HF) is a
useful **negative example**: it is exactly the "recount the pipeline" style CONTEXT.md's LOCKED
DECISION 2 forbids for v4 (a full 7-stage compression-lineage narrative with two "no-winner" gate
explanations inline in the card body). The v4 card must NOT follow this template; it must follow the
tone of the current `README.md` (~150 lines, quickstart-led) but tighter, and push all lineage detail
out to a GitHub link.

**Primary recommendation:** Convert the single pruned v4 judge checkpoint to GGUF with the installed
llama.cpp (no upgrade needed), add an expert-count sanity check alongside the existing block-count
check, run the Q8→Q6→Q5 ladder reusing `_pkg_gguf_eval_run.sh` against the pre-registered ±2pp bands
(need fresh v4 bands — the existing `pkg03_quantization_ladder.json` bands are v3/v1.3 numbers and do
not apply), rewrite the model card from scratch in the operator-only format (do not adapt
`judge_gguf_README.md`), and reuse `_pub03_upload.sh` + the PUB-03 round-trip pattern for the
publish step.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| HF→GGUF format conversion | Local toolchain (llama.cpp CLI on GB10) | — | `convert_hf_to_gguf.py` is a local Python script reading the safetensors checkpoint directly; no serving layer involved |
| Quantization (Q8/Q6/Q5) | Local toolchain (llama-quantize) | — | Same local CLI binary; pure tensor-precision transform, no network |
| Quality gating per tier | Serving layer (llama-server, local GB10) | Eval harness (scripts/relabel/eval_relabel.py) | GGUF must be served to be scored; server is local-only (`127.0.0.1`), harness is a thin HTTP client |
| Concurrent-sequence smoke | Serving layer (llama-server `--parallel N`) | — | Tests the CUDA backend's batched-decode path under the pruned 224-expert topology, not a conversion-time concern |
| HF model card content | Publication / distribution tier (HF Hub repo README) | GitHub repo (PIPELINE.md/JOURNAL.md for methodology) | LOCKED DECISION 2: operator surface lives on HF, methodology narrative lives in git — a hard tier split, not just a style choice |
| Post-upload round-trip validation | Publication tier (download from HF) + local serving | — | Must prove the *uploaded* bytes work, not just the local pre-upload artifact — download is not optional |
| Upload mechanics (stall handling, retries) | Local orchestration script (`_pub03_upload.sh`) | HF Hub API | `hf upload-large-folder` measured to deadlock on this host; sequential per-file with a stall watchdog is the proven path |

## Package Legitimacy Audit

No new external packages are needed for this phase. The full toolchain is already installed and was
exercised by the prior phase (23-02/26):

| Package | Registry | Evidence | Verdict | Disposition |
|---------|----------|----------|---------|-------------|
| `gguf` (Python) | PyPI | Installed, v0.19.0 (`pip show gguf`); used by the existing block-count sanity script | OK | Reuse, no reinstall |
| `huggingface_hub` / `hf` CLI | PyPI | Installed, v1.23.0 (`hf version`); used by `_pub03_upload.sh` | OK | Reuse, no reinstall |
| `llama.cpp` (built from source) | GitHub `ggml-org/llama.cpp` | `~/llama.cpp` build `8f114a9`, tag range b9999-b10004, `[VERIFIED: local build inspection]` — built 2026-07-10, past the b9180 MTP-support floor | OK | Reuse; no rebuild required unless a fresher tag is explicitly desired |

**Packages removed due to `[SLOP]` verdict:** none.
**Packages flagged as suspicious `[SUS]`:** none.

If the plan chooses to `git pull` llama.cpp to a newer tag before conversion (optional, not
required — the installed build already exceeds the floor), re-run this audit step against the new
commit only if a *different* fork/mirror URL is proposed; the current `ggml-org/llama.cpp` origin is
the canonical upstream and does not need re-verification.

## Architecture Patterns

### System Architecture Diagram

```
                     models/Qwen3.6-35B-A3B-judge-v4-pruned-k224/
                     (60 GB bf16, 224 experts/layer, mtp.* + shared_expert.* intact)
                                        │
                                        ▼
                     [1] convert_hf_to_gguf.py --outtype q8_0
                         (Qwen3_5MoeTextModel path; reads text_config.num_experts=224
                          via hparams flatten; stacked gate_up_proj/down_proj split
                          by actual tensor shape, no hardcoded expert count)
                                        │
                                        ▼
                     wp-judge-v4-pruned-k224.Q8_0.gguf  (~33.6 GiB projected)
                                        │
                          ┌─────────────┼──────────────────┐
                          ▼             ▼                  ▼
                 [2a] block-count   [2b] expert-count   [2c] llama-server
                     sanity check     sanity check         --parallel N smoke
                 (existing script,   (NEW — extend the    (concurrent-sequence
                  vs safetensors     existing script)      CUDA-backend gate,
                  index)                                   PKG4-01 requirement)
                          │             │                  │
                          └─────────────┴──────────────────┘
                                        ▼
                     [3] Cascading compression gate (PKG4-02)
                         Gate 1 bf16 baseline (already the s1 rho 0.8134 from Phase 26,
                             re-anchor on the SAME Q8/llama.cpp stack, not bf16/vLLM)
                         Gate 2 pre-determined WARRANTED (60 GB checkpoint alone fits
                             121 GB host easily — re-derive the actual warrant rationale;
                             the ROADMAP's 134 GiB pair-based rationale no longer applies
                             to a judge-only ship)
                         Ladder: Q8 -> Q6_K -> Q5_K_M, each served + scored via
                             scripts/_pkg_gguf_eval_run.sh, stop at first tier
                             >2pp below the Gate-1 rho floor
                                        │
                                        ▼
                     [4] Ship-tier GGUF selected (lowest tier within ±2pp)
                                        │
                          ┌─────────────┴──────────────┐
                          ▼                             ▼
                 [5a] Operator-only HF model card   [5b] scripts/_pub03_upload.sh
                     (NEW — do not adapt              (sequential per-file upload,
                      judge_gguf_README.md;            stall watchdog, reuse as-is)
                      LOCKED DECISION 2 shape)
                          │                             │
                          └─────────────┬───────────────┘
                                        ▼
                     [6] Post-upload round-trip validation (PUB4-01)
                         download from HF -> load GGUF via llama-server ->
                         1 judge smoke prompt -> parse <judge_output> ->
                         confirm coherent + matches pre-upload local smoke
                         (pattern: output/packaging/pub03_validation_receipt.json)
```

### Recommended Project Structure

No new top-level directories needed — this phase follows the existing `output/` and `scripts/`
conventions from Phases 15/18/23/26:

```
output/pkg-v4/                      # NEW — mirrors output/packaging/ (v3) and output/prune-v4/ (Ph26)
├── gate1_bf16_baseline_v4.json     # re-anchor rho on the shipped Q8/llama.cpp stack
├── gate2_quantization_decision_v4.md
├── pkg4_quantization_ladder.json   # Q8/Q6/Q5 results, fresh v4 bands (not v3's)
├── expert_count_sanity.json        # NEW check output (paired with block-count)
├── hf_cards/
│   └── judge_v4_README.md          # operator-only card, NOT adapted from v3
└── pub4_upload_manifest.json / pub4_validation_receipt.json

scripts/
├── eval4_ext_gguf_convert.sh       # REUSE + extend: add expert-count check
├── _pkg_gguf_eval_run.sh           # REUSE as-is (already does concurrent-sequence serve+score)
├── _pub03_upload.sh                # REUSE, generalize repo/manifest names (or copy to _pub4_upload.sh)
└── relabel/eval_relabel.py         # REUSE as-is (rho scorer)
```

### Pattern 1: HF→GGUF conversion for a physically-pruned MoE

**What:** Convert a checkpoint whose expert count was surgically reduced (256→224) and whose
`config.json` was rewritten accordingly, without assuming any code path hardcodes the architecture's
canonical expert count.
**When to use:** Any Phase 27 conversion step.
**Why it's safe here (verified by reading the installed converter):**
```python
# Source: ~/llama.cpp/conversion/base.py:1112-1113 (verified local install, commit 8f114a9)
# text_config is flattened into top-level hparams BEFORE any expert-count read:
self.hparams = {**self.hparams, **self.hparams["text_config"]}

# Source: ~/llama.cpp/conversion/base.py:1276-1277
# generic expert-count writer reads the flattened hparam, no hardcoded 256:
if (n_experts := self.find_hparam(["num_local_experts", "num_experts", "n_routed_experts"], optional=True)) is not None:
    self.gguf_writer.add_expert_count(n_experts)   # writes 224 for the pruned checkpoint

# Source: ~/llama.cpp/conversion/qwen.py:92-111 (Qwen2MoeModel.modify_tensors,
# inherited by Qwen3_5MoeTextModel via Qwen3NextModel -> Qwen2MoeModel)
# stacked gate_up_proj/down_proj tensors are split by ACTUAL torch tensor shape,
# not by a hardcoded expert count -- the axis-0-sliced 224-wide tensor "just works":
if name.endswith("mlp.experts.gate_up_proj") or name.endswith("mlp.experts.gate_up_proj.weight"):
    n_ff = data_torch.shape[-2] // 2
    gate = data_torch[..., :n_ff, :].contiguous()
    up = data_torch[..., n_ff:, :].contiguous()
```
The C++ loader's only hard checks are `n_expert <= LLAMA_MAX_EXPERTS` (512, `src/llama-hparams.h:10`)
and `n_expert_used <= n_expert` (`src/llama-model.cpp:1094-1095`) — both trivially satisfied by
224/8. The `n_expert_groups` divisibility assertion (`src/llama-model.cpp:1098-1100`) only fires when
`n_expert_groups > 1`, and Qwen3.6 carries no `n_group` config key (confirmed absent from
`config.json`), so it never triggers. `[VERIFIED: local llama.cpp source read, commit 8f114a9]`

### Pattern 2: Concurrent-sequence smoke via `llama-server --parallel`

**What:** PKG4-01 requires "concurrent-sequence CUDA-backend smoke passing." This is not a new
mechanism to build — `scripts/_pkg_gguf_eval_run.sh` already serves with `--parallel 4` and a context
size scaled to the parallel slot count (`CTX = PAR * (MAXTOK + 3072)`), which is exactly a
multi-sequence CUDA-graph/batch-decode exercise. Extending its existing readiness-probe (a real
generation, not `/health`) covers the smoke requirement without new code.
**When to use:** Directly reuse for the Q8/Q6/Q5 ladder eval runs; the "smoke" and "ladder eval" are
the same serve session, not two separate steps.

```bash
# Source: scripts/_pkg_gguf_eval_run.sh (existing, verified in repo)
PAR=4
CTX=$(( PAR * (MAXTOK + 3072) ))
"$LS" -m "$GGUF" --host 127.0.0.1 --port "$PORT" -ngl 999 -c "$CTX" --jinja -a "$ALIAS" \
  --parallel "$PAR" > "$OUT/serve.log" 2>&1 &
```

### Pattern 3: Operator-only HF model card (LOCKED DECISION 2)

**What:** A card that is a *product surface*, not a lab notebook. Five sections only: what it is,
acquisition, use, evals, links out.
**When to use:** The Phase 27 card task. Do NOT start from `judge_gguf_README.md` (v3's card) —
that file's whole "Compression lineage" section (7-stage narrative, two negative-gate paragraphs) is
exactly what's now out of scope for the card body.
**Correct starting point:** `README.md`'s tone (the project's own operator-first rewrite,
explicitly named as the style reference in CONTEXT.md) — trim further, since the card needs to be
*tighter* than even the README.

Minimal required YAML frontmatter for a GGUF repo (standard HF convention, cross-checked against the
existing published `judge_gguf_README.md` frontmatter which is already correct in shape):
```yaml
---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
pipeline_tag: text-generation
library_name: gguf
language:
  - en
tags:
  - wordpress
  - php
  - code-review
  - moe
  - qwen3
  - gguf
---
```
`[CITED: existing repo's own live frontmatter — same fields HF's model-card spec expects for a
GGUF-tagged repo: license, base_model (for lineage linking), pipeline_tag, library_name: gguf so the
Hub renders the right widget/quant picker]`

### Anti-Patterns to Avoid

- **Copying the v3 card's lineage table into the v4 card:** LOCKED DECISION 2 explicitly forbids
  this. The "no-winner" narrative for RL/Sieve/prune belongs in `PIPELINE.md`/`JOURNAL.md`, referenced
  by one link, not reproduced.
- **Sizing tasks for a gen+judge pair conversion:** the gen role is retired; converting/quantizing/
  publishing a v4 gen model is out of scope for this phase unless a future decision reverses the
  2026-07-15 retirement.
- **Reusing v3's `pkg03_quantization_ladder.json` bands as the v4 gate:** those bands (`wp_bench_floor:
  0.4284`, `judge_ensemble_rho_floor: 0.7554`) are v1.3/v3 numbers on a different base model and a
  different (unpruned) checkpoint. Phase 27 needs a **fresh Gate 1 bf16-equivalent baseline on the
  shipped Q8/llama.cpp stack** before it can gate Q6/Q5 — the closest existing v4 number
  (`selection_v4.json`'s s1 rho 0.8134) is bf16-vLLM, not Q8-llama.cpp, and the stack caveat in that
  file says explicitly these are "NOT comparable until the Phase-27 Q8 conversion."
- **Trusting `--outtype q8_0` direct conversion as automatically final:** it works (confirmed used in
  `eval4_ext_gguf_convert.sh` for the merged v4), but if a Q6/Q5 *ladder* is run, the standard two-step
  path (convert to f16 first, then `llama-quantize` per tier from the f16 master) avoids re-running
  the slow HF-tensor-read conversion three times — `scripts/run_packaging_recipe.md` already documents
  this as the intended flow for multi-tier ladders.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|--------------|-----|
| HF→GGUF conversion | Custom safetensors→GGUF writer | `convert_hf_to_gguf.py` (installed, verified Qwen3.6-MoE-aware) | Already handles stacked-expert-tensor layout, MTP tensors, MRoPE sections correctly for this exact architecture |
| Block/expert-count sanity | Ad-hoc manual inspection | Extend `eval4_ext_gguf_convert.sh`'s existing `GGUFReader`-based check | The block-count check already exists and works; adding an expert-count check is a ~10-line addition to a proven pattern, not a new tool |
| Quantization | Custom INT8/K-quant kernel | `llama-quantize` (built binary at `~/llama.cpp/build/bin/`) | Q8_0/Q6_K/Q5_K_M are standard GGUF quant types with mature, tested kernels; the project's own Gate 2 decision doc explicitly warns against hand-rolled uniform quantization (nf4 collapse precedent) |
| Concurrent-sequence serving smoke | Custom multi-request harness | `llama-server --parallel N` + existing readiness-probe pattern in `_pkg_gguf_eval_run.sh` | Already does real concurrent decode under the CUDA backend; a hand-rolled harness would just reinvent this |
| HF upload | `hf upload-large-folder` (the "obvious" tool) | `_pub03_upload.sh`'s sequential per-file pattern | **Documented failure**, not a style preference: upload-large-folder measured to deadlock twice on this exact host even with Xet disabled; the sequential+stall-watchdog script is the proven fallback |
| Judge-response scoring | New Spearman-rho scorer | `scripts/relabel/eval_relabel.py` | Already the canonical rho scorer used by every prior packaging/eval phase on this project |

**Key insight:** every mechanical piece of this phase (conversion, sanity checks, quantization,
serving, scoring, upload) already exists in this repo with a proven track record on this exact host
and toolchain. The actual new work is (1) one incremental sanity check (expert-count), (2) re-running
the existing pipeline against the pruned v4 single-model target instead of the v3 pair, and (3)
writing a genuinely new model card in a genuinely new (tighter) format. Nothing here calls for new
infrastructure.

## Common Pitfalls

### Pitfall 1: Treating the ROADMAP's "pair" framing as current scope
**What goes wrong:** A plan that budgets tasks for converting/quantizing/publishing two models (gen +
judge) when the gen role was retired 2026-07-15.
**Why it happens:** ROADMAP.md's Phase 27 section and the PKG4-01/PKG4-02 requirement text were
written before the judge-only pivot and still say "pair" / "134 GiB bf16 pair."
**How to avoid:** Plan against `selection_v4.json`'s `routes_phase_27` field and `PROJECT.md:14`
("generation half was retired as a deliverable"), not the ROADMAP prose. Confirm scope explicitly in
the plan document.
**Warning signs:** Any task mentioning a v4 gen model checkpoint path that doesn't exist on disk.

### Pitfall 2: Silently inheriting v3/v1.3 quantization gate bands for v4
**What goes wrong:** Gating the Q6/Q5 ladder against `pkg03_quantization_ladder.json`'s
`wp_bench_floor: 0.4284` / `judge_ensemble_rho_floor: 0.7554` — numbers from a different base model,
different (unpruned) checkpoint, and a judge-only vs pair context.
**Why it happens:** The file exists, is in the right shape, and is tempting to copy.
**How to avoid:** Compute a fresh Gate 1 baseline on the shipped stack (Q8-llama.cpp, this pruned
checkpoint) before descending the ladder. The nearest available v4 number
(0.8134, bf16-vLLM, single-seed s1) is explicitly stack-flagged as not-yet-comparable in
`selection_v4.json`.
**Warning signs:** A ±2pp gate computed against a floor that references a different model name.

### Pitfall 3: Assuming `--outtype q8_0` and never checking the expert axis
**What goes wrong:** The conversion silently "succeeds" (no crash) at a wrong expert count if the
safetensors file and `config.json` ever get out of sync (e.g., a future re-run of the surgery script
against a differently-shaped mask, or accidentally converting the pre-surgery `-s1-merged` checkpoint
instead of the pruned one).
**Why it happens:** `convert_hf_to_gguf.py` trusts the config; there is currently no check that the
GGUF's written `expert_count` metadata matches the actual number of expert slices found in the
tensors it processed.
**How to avoid:** Add the expert-count sanity check (Pattern 1 above) as a mandatory post-conversion
gate, symmetric with the existing block-count check. Cheap (`GGUFReader` field read + int compare),
high-value (catches exactly the class of error this phase is most exposed to — a surgically modified
non-standard topology).
**Warning signs:** GGUF loads and serves fine but produces subtly wrong routing (would likely surface
as a rho regression during the Gate 1 baseline re-measurement, not a hard crash — making the explicit
check strictly better than "it loaded, so it's fine").

### Pitfall 4: `hf upload-large-folder` deadlock
**What goes wrong:** The "obvious" HF upload command hangs indefinitely with zero I/O progress.
**Why it happens:** Measured twice on this exact host (`_pub03_upload.sh`'s header comment: "PUB-03
upload runner v3: SEQUENTIAL per-file `hf upload`. upload-large-folder deadlocked twice on this host
(10 workers wedged at pre-upload, zero io) even with Xet disabled + hub 1.23"). This is an
environment-specific, previously-diagnosed failure, not a hypothetical.
**How to avoid:** Reuse `_pub03_upload.sh`'s sequential-per-file + stall-watchdog pattern directly.
**Warning signs:** An upload command that shows no `wchar` growth in `/proc/<pid>/io` for several
minutes.

### Pitfall 5: Publishing a card that fails the operator-only bar
**What goes wrong:** Reflexively adapting `judge_gguf_README.md` (find/replace v3→v4 numbers) instead
of writing fresh, and shipping a card that still narrates RL-rejected/Sieve-no-winner/prune-history —
violating LOCKED DECISION 2 outright.
**Why it happens:** Adapting an existing file is the path of least resistance and the file is
otherwise well-formatted.
**How to avoid:** Treat the card as a new document scoped by CONTEXT.md's five-section list (what
it is/for, acquisition, use, evals, links out), styled after `README.md`, not
`judge_gguf_README.md`. Do a final read-through checking specifically for any sentence that describes
*how* the model was built (training runs, LoRA ranks, gate names, "no-winner" results) — any such
sentence should become a link, not stay as prose.
**Warning signs:** Card mentions "AIMER," "REAP," "MoE-Sieve," "GSPO," "Tinker," or any phase number —
none of these belong in the card body per LOCKED DECISION 2's explicit exclusion list.

## Code Examples

### Extending the block-count sanity check with an expert-count check

```python
# Pattern to add alongside the existing block-count check in eval4_ext_gguf_convert.sh
# (adapt the existing python3 -c block; same GGUFReader instance, one more field read)
import json
from gguf import GGUFReader

cfg = json.load(open(f"{merged}/config.json"))
tc = cfg.get("text_config", cfg)
expected_experts = tc["num_experts"]  # 224 for the pruned v4 checkpoint

r = GGUFReader(out)
ec = None
for f in r.fields:
    if f.endswith(".expert_count"):
        fld = r.fields[f]
        ec = int(fld.parts[fld.data[0]][0])
        break
assert ec == expected_experts, f"EXPERT COUNT MISMATCH: gguf={ec} vs config={expected_experts}"
print(f"[convert] expert-count sanity: PASS ({ec})")
```

### Serving + scoring one GGUF quant tier (reuse, not new)

```bash
# Source: scripts/_pkg_gguf_eval_run.sh (existing, verified in repo — reuse verbatim)
scripts/_pkg_gguf_eval_run.sh \
  output/pkg-v4/wp-judge-v4-pruned-k224.Q8_0.gguf \
  wp_judge_v4_q8 \
  output/pkg-v4/q8_eval \
  8091
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Per-expert individual tensors (`mlp.experts.{idx}.{proj}.weight`) | Stacked 3D expert tensors (`mlp.experts.gate_up_proj`/`down_proj`) | Already the format both HF's Qwen3.6 release and this project's Phase 26 surgery use | Conversion path in `conversion/qwen.py` handles the stacked case directly by tensor shape (lines 92-111), which is exactly why the 224-expert slice needs no converter changes |
| Fused/legacy Qwen3-MoE-only conversion class | `Qwen3_5MoeTextModel` (dedicated Qwen3.5/3.6 class with MTP + MRoPE mixins) | Landed with MTP support around b9180 (2026-05-16 per public release notes) `[CITED: WebSearch — bartowski/Qwen_Qwen3.6-35B-A3B-GGUF and llama.cpp release history]` | This is the class actually used for our conversion; confirms the `>=b9180` floor in PKG4-01 is the correct one and the installed b10000+ build is safely past it |
| `upload_folder`/`upload-large-folder` as the default HF upload path | Sequential per-file `hf upload` with stall watchdog | Discovered during Phase 18 PUB-03 (documented in `_pub03_upload.sh` header) | Directly reusable; don't re-litigate this choice |

**Deprecated/outdated:**
- v3's `judge_gguf_README.md` narrative-card format: superseded by LOCKED DECISION 2's operator-only
  format for v4. Keep the v3 file as historical reference only; do not template from it.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Gate 2 "quantization warranted" rationale needs re-derivation for a single 60 GB checkpoint (vs the ROADMAP's stale 134 GiB pair-based rationale) — the pruned model alone already fits the 121 GB host at bf16 with headroom, so the warrant may need to rest on a different constraint (e.g., leaving room for concurrent serving/KV cache/other project models) rather than "doesn't fit" | Architecture Patterns, Common Pitfalls #1 | If the plan copies the old Gate 2 justification verbatim it will state a false constraint (a 60 GB model plainly fits 121 GB); low risk to ship correctness, but it is a documentation-honesty issue this project has been careful about elsewhere (see Gate 2's own "honest execution status" precedent) |
| A2 | The exact ±2pp gate bands (wp-bench floor does not apply — no gen model — only a judge-rho floor) for the v4 Q6/Q5 ladder are not yet computed; this research recommends computing them fresh from a Q8-llama.cpp Gate-1 baseline rather than reusing any existing number | Common Pitfalls #2 | If the planner or executor reuses a stale floor, a genuinely-regressed tier could ship, or a genuinely-fine tier could be wrongly rejected |
| A3 | HF model card YAML frontmatter fields (license/base_model/pipeline_tag/library_name/tags) are based on this project's own already-published, working card and general HF Hub convention, not independently re-verified against HF's current schema docs this session | Architecture Patterns, Pattern 3 | Low risk — the existing repo's frontmatter is live and rendering correctly on HF today; if HF's schema changed since, a field might need updating, but this is a cosmetic/rendering risk, not a functional one |
| A4 | `LLAMA_MAX_EXPERTS` limit (512) and the absence of an `n_group` key in this Qwen3.6 config are read from the *currently installed* llama.cpp source tree; if the plan chooses to pull a newer llama.cpp tag before conversion, these specific line numbers/values should be re-confirmed against the new commit (though the underlying architectural facts — Qwen3.6 has no expert grouping, 224 < 512 — are very unlikely to change) | Architecture Patterns, Pattern 1 | Very low — even if line numbers drift, the qualitative conclusion (no hardcoded-256 landmine) is derived from reading multiple independent code paths, not one fragile line |

## Open Questions

1. **Does the v4 card need a separate "gen" pointer repo, or does it just recommend "a current Qwen
   base model" generically (as `README.md` already does)?**
   - What we know: `README.md` already handles this exact framing today for the live v3 judge repo
     ("point at a current base model for that" — no specific repo named).
   - What's unclear: Whether CONTEXT.md's acquisition-section requirement wants a concrete
     recommended base (e.g., `Qwen/Qwen3.6-35B-A3B` itself, since that's literally what the judge was
     trained from) named explicitly in the v4 card, vs. the vaguer existing phrasing.
   - Recommendation: Default to naming `Qwen/Qwen3.6-35B-A3B` explicitly (it's the judge's own base
     and a natural, low-risk recommendation) unless CONTEXT.md's discuss-phase surfaces a different
     preference.

2. **Exact repo name / rename for the HF Hub target.**
   - What we know: v3's repo is `iamchum/wp-qwen3-30b-a3b-wp-judge-v1.3-gguf`; the project itself
     renamed to "Qwen 3 WP Judge" locally (README title) as of the 2026-07-15 pivot.
   - What's unclear: Whether Phase 27 publishes to a *new* HF repo (e.g.,
     `iamchum/qwen3-wp-judge-v4-gguf`) or overwrites/versions within the existing v3 repo. This has
     real consequences for the upload manifest and for whether the v3 repo stays live as a fallback.
   - Recommendation: New repo (matches the project's own naming rename and keeps v3 available as a
     documented fallback per the pipeline's "no-winner is a valid, recorded outcome" ethos) — but this
     is a publication decision, not a research one; confirm at plan/discuss time if not already locked.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| llama.cpp (built binaries: `llama-cli`, `llama-server`, `llama-quantize`) | PKG4-01/PKG4-02 conversion + serving + quantization | Yes | commit `8f114a9`, tag range b9999-b10004 (2026-07-10 build) | Rebuild from a fresher tag if a specific bugfix is needed; not required for the >=b9180 floor |
| `gguf` Python package | GGUF metadata sanity checks | Yes | 0.19.0 | — |
| `huggingface_hub` / `hf` CLI | Upload + post-upload API listing | Yes | 1.23.0 | — |
| Disk space (scratch for f16 master + 3 quant tiers) | Ladder conversion | Yes | 1.7 TB free on `/` | — |
| GB10 unified memory (121 GB) | Serving each GGUF tier for eval | Yes | 121 GiB total, 114 GiB available at rest | 60 GB pruned checkpoint fits comfortably even at bf16-equivalent; no memory-wall risk for a judge-only ship (unlike v3's pair-serving constraint) |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none — full toolchain already provisioned and proven on this
exact host by Phases 15/18/23/26.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (repo-wide convention; `tests/` directory, e.g. `tests/test_sieve_arch.py`, `tests/test_prune_selection.py`) |
| Config file | none dedicated — no `pytest.ini`/`conftest.py` found at repo root; tests run via `pytest tests/` directly |
| Quick run command | `pytest tests/test_prune_selection.py -x` (or a new `tests/test_pkg_v4_gguf_sanity.py` for this phase) |
| Full suite command | `pytest tests/` |

This project's convention for packaging/conversion scripts specifically is a `--self-check` flag
baked into the script itself (see `scripts/prune_apply_physical_v4.py --self-check`,
`scripts/eval4_ext_unmerged_lora_convert_llamacpp.py`), not a separate pytest file, for scripts whose
correctness is a shape/assertion check rather than a unit-testable pure function. This phase should
follow the same convention for the expert-count sanity extension (self-check embedded in the
conversion driver) rather than introducing a new pytest file for a single assertion.

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PKG4-01 | GGUF conversion completes, block-count + expert-count sanity pass | script self-check | `scripts/eval4_ext_gguf_convert.sh <merged_dir> <out.gguf>` (extended) | ✅ block-count exists; ❌ expert-count check is Wave-0 work |
| PKG4-01 | Concurrent-sequence CUDA-backend smoke | integration (served) | `scripts/_pkg_gguf_eval_run.sh <gguf> <alias> <out_dir> <port>` | ✅ exists, reuse as-is |
| PKG4-01 | Shared-expert quant-type metadata independently verified | GGUF metadata inspection | new small script/self-check reading `gguf.GGUFReader` tensor `type` field for `shared_expert.*`/`shared_expert_gate.weight` tensors post-quantize | ❌ Wave 0 — no existing script inspects per-tensor quant type; `llama-quantize`'s generic `tensor_allows_quantization()` (verified read, `src/llama-quant.cpp:288-355`) applies no shared-expert-specific override, so this check should assert the *expected* uniform behavior (shared-expert tensors quantized at the same tier as routed experts, since no special-case exists in the code) rather than assume a different precision |
| PKG4-02 | Cascading gates re-run (Gate1/Gate2/ladder) | integration (measured, not unit-testable) | `scripts/_pkg_gguf_eval_run.sh` per tier + `scripts/relabel/eval_relabel.py` | ✅ harness exists; ❌ fresh v4 bands file (`pkg4_quantization_ladder.json`) is Wave 0 |
| PUB4-01 | Post-upload round-trip (download, GGUF load, judge smoke) | integration (live HF) | pattern from `output/packaging/pub03_validation_receipt.json` generation step (script not found standalone — was likely run inline/interactively during Phase 18; **treat as Wave 0: write a small driver script** `scripts/pub4_validate_upload.py` following the receipt's exact shape) | ❌ Wave 0 — the *receipt* exists from Phase 18 but no reusable standalone script producing it was found in `scripts/`; this phase should extract that logic into a real script rather than re-improvise interactively |

### Sampling Rate
- **Per task commit:** run the relevant self-check flag / script directly (conversion sanity, gate
  eval for the tier just produced).
- **Per wave merge:** re-run the full ladder comparison (`pkg4_quantization_ladder.json` regenerated)
  and `pytest tests/` for any touched shared code (e.g., if `prune_apply_physical_v4.py` or shared
  eval harness code is modified).
- **Phase gate:** full round-trip (upload → download → GGUF load → judge smoke) must pass green before
  `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] Extend `scripts/eval4_ext_gguf_convert.sh` (or a v4-specific copy) with the expert-count sanity
      check (Pattern 1 / Code Example above).
- [ ] New small script/self-check for shared-expert quant-type independent verification (per-tensor
      GGUF type inspection post-quantize).
- [ ] `output/pkg-v4/gate1_bf16_baseline_v4.json` equivalent for the shipped Q8/llama.cpp stack (there
      is no bf16 GGUF Gate-1 baseline yet for the pruned v4 checkpoint on this stack — the closest
      number, s1 rho 0.8134, is bf16-**vLLM**, explicitly flagged non-comparable in `selection_v4.json`).
- [ ] `scripts/pub4_validate_upload.py` — extract the Phase-18 PUB-03 round-trip logic
      (API listing + GGUF load + judge/gen smoke) into a standalone, re-runnable script; currently
      only the output receipt survives, not a reusable driver.

## Security Domain

`security_enforcement` is not explicitly disabled in `.planning/config.json` (absent = enabled), so
this section is included, scoped honestly to what actually applies to a local-conversion +
HF-publication phase — most standard ASVS web/API categories are not applicable to this domain.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | Partial | HF Hub authentication for upload uses the `hf` CLI's stored token (`huggingface_hub` credential store), not a hand-rolled auth path — standard tool-managed auth, no new code |
| V3 Session Management | No | No user sessions in this phase; local script + one-shot HF API calls only |
| V4 Access Control | No | No access-control logic implemented in this phase; repo visibility (public/private) is an HF Hub setting decided at publish time, not code |
| V5 Input Validation | Partial | Conversion/quantization scripts validate shapes via assertions (block-count, and the new expert-count check) rather than trusting inputs blindly — this is the correct control for this domain (data/tensor integrity, not user input) |
| V6 Cryptography | No | No cryptographic operations performed by this phase's code; HF Hub transport security (TLS) is handled by `huggingface_hub`, not hand-rolled |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Publishing a checkpoint with an undetected structural defect (e.g., silently wrong expert count from a mismatched config/tensor pair) | Tampering (data integrity, not adversarial) | Block-count + expert-count sanity gates before any upload step — exactly the pattern this research recommends extending |
| Uploading to the wrong / a stale HF repo, or exposing a private research artifact publicly before sign-off | Information Disclosure | This project's established pattern: HF upload is a distinct, explicitly-gated final step (see PKG-04's note "Upload push is human-authorized final step" for v3) — Phase 27 should carry the same discipline: conversion/quantization/local-validation complete and reviewed BEFORE the publish step runs |
| Leaking an HF write token in a committed script or log | Information Disclosure | `_pub03_upload.sh` uses the `hf` CLI's own credential store (no token in the script); follow the same pattern, never inline a token in a new script |

Nothing in this phase touches user-facing authentication, session handling, or application-level
input validation in the traditional web-app sense — this is a local ML-artifact pipeline plus a
one-way publish step, and the security-relevant controls are data-integrity gates (already this
project's established idiom) and credential hygiene (already followed by the reused upload script).

## Sources

### Primary (HIGH confidence)
- `~/llama.cpp/conversion/base.py` (local install, commit `8f114a9`) — hparams flattening
  (`text_config` merge), generic expert-count GGUF writer (`add_expert_count`)
- `~/llama.cpp/conversion/qwen.py` (local install) — `Qwen3_5MoeTextModel` registration, stacked
  expert-tensor splitting logic, MTP mixin (`_Qwen35MtpMixin`)
- `~/llama.cpp/src/llama-model.cpp`, `src/llama-hparams.h`, `src/llama-quant.cpp` (local install) —
  expert-count runtime assertions, `LLAMA_MAX_EXPERTS`, tensor quantization eligibility logic
- `output/prune-v4/selection_v4.json`, `.planning/phases/26-conditional-gate-c-merge-prune-re-test/26-02-SUMMARY.md`
  — ship disposition, size numbers (60 GB pruned bf16, 33.6 GiB Q8 *projection*), stack caveats
- `PROJECT.md:14`, `README.md`, `JOURNAL.md` — confirmation that the gen role is retired as a
  deliverable (judge-only ship)
- `scripts/eval4_ext_gguf_convert.sh`, `scripts/_pkg_gguf_eval_run.sh`, `scripts/_pub03_upload.sh`,
  `scripts/run_packaging_recipe.md`, `output/packaging/gate2_quantization_decision.md`,
  `output/packaging/pkg03_quantization_ladder.json`, `output/packaging/pub03_validation_receipt.json`
  — all read directly from the repo, all prior-art scripts/artifacts this phase should reuse/adapt
- `models/Qwen3.6-35B-A3B-judge-v4-pruned-k224/config.json` (on-disk artifact) — verified
  `num_experts: 224`, `num_experts_per_tok: 8`, `mtp_num_hidden_layers: 1`, no `n_group` key
- `.planning/phases/27-packaging-publication-refresh/CONTEXT.md` — the two LOCKED user decisions

### Secondary (MEDIUM confidence)
- WebSearch: "llama.cpp b9180 changelog Qwen3.6 MoE convert_hf_to_gguf" — confirms b9180 as the MTP
  support landing point for Qwen3.5/3.6 conversion, cross-checked against the local code's MTP mixin
  and the ROADMAP's own `>=b9180` floor

### Tertiary (LOW confidence)
- None used as authoritative — all package/toolchain claims were either read directly from installed
  source or cross-checked against on-disk project artifacts.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages, full toolchain already installed and proven on this host
- Architecture (conversion correctness for 224-expert topology): HIGH — verified by reading the
  actual installed converter and loader source, not inferred from documentation
- Scope correction (judge-only, not pair): HIGH — verified against three independent on-disk sources
  (`PROJECT.md`, `README.md`, `selection_v4.json`)
- Pitfalls: HIGH — sourced from this project's own documented prior failures (nf4 collapse,
  upload-large-folder deadlock) plus one newly-identified gap (no expert-count sanity check exists
  yet)
- Gate bands for the v4 ladder: LOW/open — genuinely unknown until Gate 1 is re-measured on the
  correct stack; flagged as Wave 0 work, not assumed

**Research date:** 2026-07-17
**Valid until:** 2026-07-31 (fast-moving toolchain domain — llama.cpp ships near-daily; re-verify the
installed build's tag if planning is delayed more than ~2 weeks)
