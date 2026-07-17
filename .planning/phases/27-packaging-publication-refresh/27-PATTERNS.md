# Phase 27: Packaging & Publication Refresh - Pattern Map

**Mapped:** 2026-07-17
**Files analyzed:** 8 (4 create, 4 modify/reuse) + 1 fresh-write (card)
**Analogs found:** 9 / 9

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `scripts/pub4_validate_upload.py` | script/driver | request-response (HF API + local GGUF serve) | `output/packaging/pub03_validation_receipt.json` (receipt shape) + `scripts/prune_gate_v4.py` (`--self-check` driver shape) | role-match (receipt exact-shape, driver structural) |
| shared-expert quant-type self-check (new, likely appended to `scripts/eval4_ext_gguf_convert.sh` or a sibling script) | utility/config-validator | file-I/O (GGUF metadata read) | `scripts/eval4_ext_gguf_convert.sh` block-count check | exact (same `GGUFReader` idiom) |
| `output/pkg-v4/gate1_baseline_v4.json` | config/data artifact | batch (one-shot measurement record) | `output/packaging/gate1_bf16_baseline.json` schema (referenced by `pkg03_quantization_ladder.json`'s `gate1_baseline_ref`) | role-match (schema, not values) |
| `output/pkg-v4/pkg4_quantization_ladder.json` | config/data artifact | batch | `output/packaging/pkg03_quantization_ladder.json` | exact (shape only; values non-transferable) |
| `scripts/eval4_ext_gguf_convert.sh` | script/utility | file-I/O + transform | itself (extend in place) | exact — extend, don't replace |
| `scripts/_pkg_gguf_eval_run.sh` | script/integration | request-response (serve+score) | itself (reuse as-is) | exact |
| `scripts/_pub03_upload.sh` (or a `_pub4_upload.sh` copy) | script/orchestration | batch (sequential file upload) | itself | exact |
| `scripts/relabel/eval_relabel.py` | utility/scorer | transform (batch score) | itself (reuse as-is) | exact |
| v4 HF model card (`output/pkg-v4/hf_cards/judge_v4_README.md`) | config/docs | transform (content authoring) | frontmatter: `output/packaging/hf_cards/judge_gguf_README.md` (positive for YAML only); body tone: root `README.md` | split — YAML exact-match, body is a NEGATIVE example, use README.md instead |

## Pattern Assignments

### `scripts/pub4_validate_upload.py` (new script, request-response + file-I/O)

**Analogs:** `output/packaging/pub03_validation_receipt.json` (target output shape — this is what Phase 18 produced *without* a standalone script) + `scripts/prune_gate_v4.py` (how this repo structures a `--self-check`, argparse-driven, JSON-emitting driver) + `scripts/_pkg_gguf_eval_run.sh` (the serve+probe pattern to call into, don't reimplement).

**Why:** No standalone Phase-18 script exists (`grep` found none) — only the receipt JSON survives. The receipt IS the target schema; `prune_gate_v4.py` is the closest analog in the repo for "a Python driver script with `--self-check`, argparse, and a JSON artifact write at the end."

**Target receipt shape to reproduce** (`output/packaging/pub03_validation_receipt.json`, full file, 82 lines):
```json
{
  "requirement": "PUB-03",
  "title": "Post-upload validation — round-trip from DOWNLOADED HF artifacts",
  "generated_utc": "2026-07-12",
  "downloaded_from_hf": true,
  "scratch_paths": {"judge_gguf": "models/_hf_dl_scratch/judge/...", "note": "scratch cleaned after validation; re-download to reproduce"},
  "api_listing": {"ok": true, "repos": {"<repo_id>": {"public": true, "files": {"<name>": <size_bytes>}, "matches_manifest": true}}},
  "gguf_load": {
    "ok": true,
    "engine": "llama.cpp llama-server (~/llama.cpp/build/bin), -ngl 999 -c 12288 --jinja",
    "header": {"magic": "GGUF", "version": 3, "n_tensors": 579, "arch": "qwen3moe", "expert_count": 128, "block_count": 48, "file_type": "7 (Q8_0)", "size_bytes": 32483931840},
    "judge_smoke_parsed": true,
    "judge_smoke": {"prompt_source": "data/phase4_4/smoke_prompts.json idx 0", "parse_format": "json", "prose_rubric_dims": 9, "overall_score": 74.0, "response_file": "..."}
  }
}
```
Note: Phase 27 is judge-only (LOCKED DECISION per CONTEXT.md scope correction) — drop the `gen_smoke` block entirely; do not carry it forward into `pub4_validation_receipt.json`.

**Self-check driver shape** (`scripts/prune_gate_v4.py:27,475-480`):
```python
# .venv-tinker/bin/python -m scripts.prune_gate_v4 --self-check   # CPU, no GPU
import argparse
...
print("self-check OK")
...
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--self-check", action="store_true")
```
Mirror this for `pub4_validate_upload.py`: argparse with a `--self-check` mode that validates the script's own JSON-shape logic without hitting the network (e.g. fabricate a small in-memory listing dict and confirm the schema/asserts hold), plus the real mode that does: (1) `huggingface_hub` API listing of the new repo, compare against an upload manifest, (2) download the GGUF to scratch, (3) load via `~/llama.cpp/build/bin/llama-server` (reuse the exact serve-then-probe idiom in `scripts/_pkg_gguf_eval_run.sh:16-32` — real generation as readiness probe, not `/health`), (4) run one judge smoke prompt, parse via `eval.output_parsers.parse_judge_scores` (same parser `scripts/relabel/eval_relabel.py:39` uses), (5) write `output/pkg-v4/pub4_validation_receipt.json` in the shape above.

**Credential pattern:** do NOT inline an HF token — follow `_pub03_upload.sh`'s convention of relying on the `hf`/`huggingface_hub` CLI's stored credential store.

---

### Shared-expert quant-type self-check (new, extends `scripts/eval4_ext_gguf_convert.sh`)

**Analog:** the existing block-count check in the same file — copy the `GGUFReader` idiom verbatim and add a second assertion.

**Existing block-count check to extend** (`scripts/eval4_ext_gguf_convert.sh:28-47`, read in full):
```bash
echo "[convert] block-count sanity check vs safetensors index"
python3 -c "
import json, sys
from gguf import GGUFReader
merged, out = '$MERGED', '$OUT'
cfg = json.load(open(f'{merged}/config.json'))
tc = cfg.get('text_config', cfg)
expected = tc['num_hidden_layers'] + tc.get('mtp_num_hidden_layers', 0)
r = GGUFReader(out)
bc = None
for f in r.fields:
    if f.endswith('.block_count'):
        fld = r.fields[f]
        bc = int(fld.parts[fld.data[0]][0])
        break
print(f'expected block_count (num_hidden_layers + mtp)={expected} gguf_block_count={bc}')
assert bc == expected, f'BLOCK COUNT MISMATCH: gguf={bc} vs safetensors-index/config={expected}'
print('[convert] block-count sanity: PASS')
"
```

**Parallel expert-count check** — RESEARCH.md already wrote this exact addition (27-RESEARCH.md "Code Examples" section, verbatim, no invention needed):
```python
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

**Shared-expert per-tensor quant-type check (net-new, no code-level analog — same idiom, new field):** iterate `r.tensors` (not `r.fields`) for names matching `shared_expert.*`/`shared_expert_gate.weight`, read `.tensor_type`, and assert it equals the quant tier applied everywhere else (e.g. `GGML_TYPE_Q8_0`) since `llama-quantize`'s `tensor_allows_quantization()` (`~/llama.cpp/src/llama-quant.cpp:288-355`, cited in RESEARCH.md) applies no shared-expert special case — the check should assert uniform behavior, not a different expected precision. Use the same `GGUFReader` instance/idiom as the two checks above; extend the same python3 -c block or a sibling `--self-check` on the driver.

---

### `output/pkg-v4/gate1_baseline_v4.json` + `pkg4_quantization_ladder.json` (data artifacts, batch)

**Analog:** `output/packaging/pkg03_quantization_ladder.json` (read in full, 30 lines) — schema is directly reusable, values are not (v3/v1.3 numbers, different base model, different checkpoint — RESEARCH.md Pitfall 2 explicitly forbids reuse of `wp_bench_floor: 0.4284` / `judge_ensemble_rho_floor: 0.7554`).

**Exact schema to replicate** (field names, nesting, stop-rule prose):
```json
{
  "requirement": "PKG4-02",
  "title": "Incremental quantization ladder Q8->Q6->Q5 with +/-2pp stop rule (v4, judge-only)",
  "gate1_baseline_ref": "output/pkg-v4/gate1_baseline_v4.json",
  "bands": {
    "judge_ensemble_rho_floor": "<fresh v4 Q8-llama.cpp measurement — DO NOT copy 0.7554>",
    "epsilon_pp": 2
  },
  "stop_rule": "Descend Q8->Q6->Q5. Ship the lowest tier whose judge rho is within 2pp of the Gate 1 baseline. Halt descent at the first tier that drops >2pp.",
  "ladder": [
    {"tier": "Q8", "method": "GGUF Q8_0 (llama.cpp)", "status": "MEASURED|pending-toolchain", "measured": { "...": "..." }}
  ]
}
```
Note v3's ladder entries also carry a `wp_bench_floor` — drop it for v4: judge-only ship, no gen model, no wp-bench axis (RESEARCH.md Architecture Patterns, Pattern 3 area / Common Pitfalls #1).

`gate1_baseline_v4.json` has no directly-readable analog on disk under that exact name (v3's `output/packaging/gate1_bf16_baseline.json` is referenced by path in `pkg03_quantization_ladder.json:5` but was not independently opened here — the *reference pattern* is what matters: the ladder file always points at a separate, standalone gate1-baseline JSON file rather than inlining Gate 1 numbers). Structure `gate1_baseline_v4.json` as a standalone sibling file with at minimum: `{"stack": "Q8-llama.cpp (NOT bf16-vLLM)", "judge_rho": <measured>, "n": <val-set-size>, "measured_utc": "...", "note": "re-anchor per RESEARCH.md — s1 rho 0.8134 from Phase 26 is bf16-vLLM and explicitly non-comparable (selection_v4.json)"}`.

---

### `scripts/eval4_ext_gguf_convert.sh` (extend in place)

**Analog:** itself — read in full above (49 lines). Extension point: append the expert-count + shared-expert-type checks directly after the existing block-count `python3 -c` block (same file, same `$MERGED`/`$OUT` shell vars already in scope), or fold both new assertions into the same `python3 -c "..."` heredoc to avoid a second Python interpreter spin-up. Preserve the file's existing conventions: `set -euo pipefail`, `ROOT`/`LLAMACPP` absolute-path vars at top, `echo "[convert] ..."` progress lines, final `PASS` print per check.

---

### `scripts/_pkg_gguf_eval_run.sh` (reuse as-is)

**Analog:** itself. CLI shape confirmed (43 lines, read in full): `_pkg_gguf_eval_run.sh <gguf_path> <alias> <out_dir> <port> [maxtok]`. A v4 invocation (per RESEARCH.md Code Examples, confirmed matches script's actual arg order):
```bash
scripts/_pkg_gguf_eval_run.sh \
  output/pkg-v4/wp-judge-v4-pruned-k224.Q8_0.gguf \
  wp_judge_v4_q8 \
  output/pkg-v4/q8_eval \
  8091
```
No script changes needed — internals already do concurrent-sequence smoke (`--parallel 4`, `PAR*(MAXTOK+3072)` context sizing, lines 14-16) + real-generation readiness probe (lines 24-32) + judge capture (`scripts.sieve_capture_judge_http`, lines 35-39) + rho scoring via `scripts/relabel/eval_relabel.py` (line 42) in one session.

---

### `scripts/_pub03_upload.sh` (adapt: repo names + manifest path only)

**Analog:** itself, read in full (67 lines). Reusable verbatim except:
- Manifest source: `output/packaging/pub03_upload_manifest.json` → new `output/pkg-v4/pub4_upload_manifest.json`.
- **Manifest format** (extracted from the inline python3 iterator, lines 58-64): top-level JSON key `"repos"`, each entry has `repo_id` + `files: [{path, repo_path}]`. The shell loop consumes it as tab-separated `repo_id<TAB>local<TAB>remote` per line via:
```python
import json
man = json.load(open('output/packaging/pub03_upload_manifest.json'))
for r in man['repos']:
    for f in r['files']:
        print(f"{r['repo_id']}\t{f['path']}\t{f['repo_path']}")
```
- **Stall-detection pattern** (lines 15-22, 30-42): sums `wchar` from `/proc/<pid>/io` across the `hf upload` process and its forked children (`pgrep -P`) every 30s; if cumulative `wchar` doesn't grow for 10 consecutive checks (5 min), sends TERM then KILL to the process group and retries (3 attempts total per file, 30s backoff between attempts).
- Env: `unset HF_XET_HIGH_PERFORMANCE; export HF_HUB_DISABLE_XET=1` at top — carry forward unchanged (documented root-cause workaround for the `upload-large-folder` deadlock, RESEARCH.md Pitfall 4).
- Target repo per CONTEXT.md LOCKED DECISION 3: `iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf` (new repo, do not touch the v3 repo).

---

### `scripts/relabel/eval_relabel.py` (reuse as-is)

**Analog:** itself, read in full (75 lines). CLI/entry shape:
```
.venv-tinker/bin/python scripts/relabel/eval_relabel.py <path/to/judge_responses.jsonl>
```
Positional arg defaults to `output/relabel/eval_relabel_v1/judge_responses.jsonl` if omitted (line 24) — always pass explicitly for v4 runs. Writes `eval_summary.json` next to the input capture (lines 68-75), not to a fixed path — this is exactly why `_pkg_gguf_eval_run.sh` can call it once per tier without clobbering. No changes needed; it already reads `data/reasoning_dataset/openai_val.jsonl` and `output/relabel/val_labels_v1.json` generically via `parse_judge_scores`/`load_dim_map` from `eval/output_parsers.py` and `eval/eval_judge.py` — reusable regardless of which model produced the capture.

---

### v4 HF model card (fresh write, NOT an adaptation)

**Positive analog (tone/structure):** root `README.md` (read lines 1-40) — operator-first: title, badges, one-paragraph "what it is," a `## The model` property table, `## Quickstart`. This is what LOCKED DECISION 2 and RESEARCH.md Pattern 3 both name as the correct starting point.

**Negative analog (explicitly do NOT copy the body):** `output/packaging/hf_cards/judge_gguf_README.md` — its `## The v4.0 finding` / methodology-narrative sections (referenced from README.md's own `> Why a judge and not a generator?` callout, which itself links to `#the-v40-finding-qwen36` inside the v3 card) are exactly the "recount the pipeline" style forbidden for the v4 card body. Do not adapt this file's structure past the frontmatter.

**Reusable YAML frontmatter fields (structure only, not values)** (`output/packaging/hf_cards/judge_gguf_README.md:1-15`, read directly):
```yaml
---
license: apache-2.0
base_model: Qwen/Qwen3-30B-A3B
pipeline_tag: text-generation
library_name: gguf
language:
  - en
tags:
  - wordpress
  - ...
---
```
For v4: same field set, values updated — `base_model: Qwen/Qwen3.6-35B-A3B` (LOCKED DECISION 4 names this explicitly), same `pipeline_tag`/`library_name`/`tags` shape.

**Required v4 card sections** (from CONTEXT.md LOCKED DECISION 2, five sections only — no analog needed, this is the spec): what it is/for (judges, doesn't generate — point at `Qwen/Qwen3.6-35B-A3B` by name for generation per LOCKED DECISION 4), acquisition (repo `iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf`, which GGUF/quant), use (llama.cpp quickstart, request/response shape — mirror README.md's `## Quickstart` code-block style), performance/evals (judge rho headline + benchmark table, no v3/v4 study narrative), links out (GitHub repo for methodology, license, one-line provenance).

## Shared Patterns

### GGUF metadata sanity via `GGUFReader`
**Source:** `scripts/eval4_ext_gguf_convert.sh:28-47`
**Apply to:** the expert-count check, the shared-expert quant-type check, and any GGUF header inspection inside `pub4_validate_upload.py`'s `gguf_load` step.
```python
from gguf import GGUFReader
r = GGUFReader(out)
for f in r.fields:
    if f.endswith(".<metadata_key>"):
        fld = r.fields[f]
        val = int(fld.parts[fld.data[0]][0])
        break
```

### `--self-check` embedded driver convention
**Source:** `scripts/prune_gate_v4.py:27,475-480`
**Apply to:** `pub4_validate_upload.py` and the shared-expert quant-type check. Per RESEARCH.md's Validation Architecture section, this repo's convention for shape/assertion-style correctness is an in-script `--self-check` flag, not a separate pytest file.

### Serve-then-real-generation readiness probe (never trust `/health` alone)
**Source:** `scripts/_pkg_gguf_eval_run.sh:24-32`
**Apply to:** any new script that starts `llama-server` and needs to know it's actually ready — `/health` returns ok while still loading a 30B+ model; poll with a real 1-token chat completion instead.
```bash
out=$(curl -sf -m 30 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" 2>/dev/null)
echo "$out" | grep -q '"content"' && READY=1
```

### Sequential-upload + stall-watchdog (never `hf upload-large-folder`)
**Source:** `scripts/_pub03_upload.sh` (whole file)
**Apply to:** `_pub4_upload.sh`/adapted `_pub03_upload.sh` for the v4 repo push. Documented deadlock, not a preference (RESEARCH.md Pitfall 4).

### No hardcoded HF token
**Source:** `scripts/_pub03_upload.sh` header comment + credential-store usage throughout
**Apply to:** `pub4_validate_upload.py` (download step) and the upload script — always rely on the `hf` CLI's stored credential store.

## No Analog Found

None — RESEARCH.md's "Don't Hand-Roll" and "Wave 0 Gaps" sections already established that every mechanical piece of this phase has a direct prior-art match in this repo; the only genuinely new logic is (1) the expert-count/shared-expert-type sanity extensions, (2) the standalone `pub4_validate_upload.py` driver (receipt shape exists, driver script does not), and (3) the v4 card body (spec-driven, no code analog by design — LOCKED DECISION 2 forbids templating from the one file that would otherwise be the obvious analog).

## Metadata

**Analog search scope:** `scripts/`, `output/packaging/`, `output/prune-v4/`, repo root (`README.md`), `.planning/phases/27-packaging-publication-refresh/`
**Files scanned:** `scripts/eval4_ext_gguf_convert.sh`, `scripts/_pkg_gguf_eval_run.sh`, `scripts/_pub03_upload.sh`, `scripts/relabel/eval_relabel.py`, `scripts/prune_gate_v4.py`, `output/packaging/pub03_validation_receipt.json`, `output/packaging/pkg03_quantization_ladder.json`, `output/packaging/hf_cards/judge_gguf_README.md`, `README.md`
**Pattern extraction date:** 2026-07-17
</content>
