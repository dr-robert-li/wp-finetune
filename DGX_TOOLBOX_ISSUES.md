# DGX-Toolbox Generic Issues

Tracked by: wp-finetune project (Robert Li)
Started: 2026-05-11
Last updated: 2026-05-12

This file collects systemic issues affecting any dgx-toolbox user, NOT
wp-finetune-specific problems. Each entry has: severity, reproduction,
generic impact, and suggested fix. Issues were discovered while integrating
the toolbox into a HuggingFace + PEFT + Unsloth fine-tuning workflow on
DGX Spark hosts, but the underlying bugs are reproducible by any user who
follows the toolbox's documented quickstart paths.

Code references point at the in-tree copy under `deps/dgx-toolbox/` but
apply equally to the standalone `~/dgx-toolbox` checkout (same source).

## Severity levels

- **P0** — blocks normal workflow with no workaround
- **P1** — blocks normal workflow with a documented workaround
- **P2** — friction; correct behaviour reachable with effort
- **P3** — cosmetic / docs gap

---

## #1 (P1) — ngc-pytorch.sh / ngc-jupyter.sh silent exit on missing host bind-mount files

**Files**: `containers/ngc-pytorch.sh:5-10`, `containers/ngc-jupyter.sh:9-14`, `lib.sh:85-89`

**Reproduction**:
1. Fresh user, freshly cloned dgx-toolbox; do NOT pre-create
   `$HOME/ngc-quickstart.sh` or `$HOME/requirements-gpu.txt`.
2. Run `containers/ngc-pytorch.sh` (or `ngc-jupyter.sh`).
3. Docker silently creates empty *directories* at the host paths, then
   bind-mounts them. Inside the container `/usr/local/bin/quickstart` is
   a directory, not a script.
4. The container's inner cmd is
   `pip install --no-deps -r /tmp/requirements-gpu.txt && quickstart && exec bash`
   (ngc-pytorch.sh:10). `pip install` on an empty requirements file
   succeeds, `quickstart` fails with "is a directory", the `&&` chain
   short-circuits, `exec bash` never runs, and the container exits.
5. The user is dropped back to the host shell with no clear error.

**Generic impact**: Any first-time toolbox user who has not run the
documented setup step hits this. The failure mode looks identical to a
clean container exit, so it is easy to misdiagnose as a Docker/NGC issue.

**Suggested fix**: Add a `require_files` helper to `lib.sh` (mirroring
the existing `ensure_dirs` at line 87) and call it at the top of both
launchers. Example:

```bash
require_files() {
  for f in "$@"; do
    if [ ! -f "$f" ]; then
      echo "ERROR: required file '$f' not found on host." >&2
      echo "Run 'setup/install.sh' (or copy the template into \$HOME) first." >&2
      exit 1
    fi
  done
}
```

Then prepend `require_files "$HOME/requirements-gpu.txt" "$HOME/ngc-quickstart.sh"`
to ngc-pytorch.sh and ngc-jupyter.sh.

---

## #2 (P0) — ngc-pytorch container HF stack incompatible with released PEFT

**Files**: `containers/ngc-pytorch.sh:9` (base image `nvcr.io/nvidia/pytorch:26.02-py3`)

**Reproduction**:
1. Launch `ngc-pytorch.sh`.
2. `python -c "import transformers, huggingface_hub, tokenizers; print(transformers.__version__, huggingface_hub.__version__, tokenizers.__version__)"`
   reports transformers 5.8.0, huggingface-hub 1.14.0, tokenizers 0.20.3.
3. `pip install peft==0.19.1` (current stable). Loading any LoRA adapter
   raises `WeightConverter.__init__() got an unexpected keyword argument
   'distributed_operation'` because PEFT 0.19 was built against the
   transformers 4.x WeightConverter signature.
4. Attempting to downgrade transformers to <5 cascades: tokenizers must
   move to 0.22–0.23, and huggingface-hub must be <1.0. The 26.02 image's
   torch / nvidia-* metapackages pin against the 5.x line, so resolving
   the downgrade requires `--force-reinstall --no-deps` across four
   packages and breaks other toolbox-installed extras.

**Generic impact**: Anyone trying to use a HF LoRA adapter, Sentence
Transformers, TRL <0.30, or most current open-source fine-tuning recipes
inside the 26.02 image fails out of the box. The `quickstart` guide
(containers/ngc-quickstart.sh:34-48) explicitly advertises PEFT and TRL
as "pre-installed", so this contradicts documentation.

**Suggested fix**: Pin a known-good HF stack in
`setup/requirements-gpu.txt` (or in a Dockerfile layer that
`pip install --force-reinstall`s on container build), e.g.
`transformers==4.46.*`, `tokenizers==0.20.*`, `huggingface-hub==0.26.*`,
`peft==0.13.*`. Document the supported stack range in README.md under a
"Container compatibility matrix" section.

---

## #3 (P0) — unsloth-studio.sh / unsloth-headless.sh auto-pull `latest` unsloth, cascading to transformers 5.x

**Files**: `containers/unsloth-studio.sh:43-71`, `containers/unsloth-headless.sh:49-75`

**Reproduction**:
1. Run `containers/unsloth-studio.sh` (or `unsloth-headless.sh`).
2. The inner cmd executes `pip install --no-deps unsloth unsloth_zoo`
   (unsloth-studio.sh:44, unsloth-headless.sh:50) with no version pin.
3. The metadata-driven dep resolver block (lines 45-66 / 51-72) then
   installs every transitive requirement from the freshly pulled
   `unsloth` package.
4. Latest unsloth (2026.5.x at time of writing) requires
   `transformers>=5.0`, which itself requires `tokenizers>=0.22`,
   `huggingface-hub>=1.0`. There is no opt-out — the install runs every
   container start.

**Generic impact**: Any user with a LoRA checkpoint trained against
transformers 4.x or peft <0.19 (i.e. most existing public checkpoints)
will get adapter-loading errors after this auto-resolve. The bug is also
silent: the container starts successfully, the breakage appears only when
the user's first `model.load_adapter(...)` call runs.

**Suggested fix**: Read an `UNSLOTH_VERSION` env var in both launchers,
defaulting to `latest` to preserve current behaviour:

```bash
UNSLOTH_VERSION="${UNSLOTH_VERSION:-latest}"
# inner cmd:
PIN=""
[ "$UNSLOTH_VERSION" != "latest" ] && PIN="==$UNSLOTH_VERSION"
pip install --no-deps "unsloth${PIN}" "unsloth_zoo${PIN}"
```

Document in README that for older-checkpoint workflows users should
export e.g. `UNSLOTH_VERSION=2026.3.5`. Also publish a "known-good"
unsloth + transformers compatibility table.

---

## #4 (P1, RESOLVED upstream @ e014af7) — unsloth-studio.sh container exits when `unsloth studio setup` fails (studio venv missing)

**File**: `containers/unsloth-studio.sh:43-71`

**Reproduction**:
1. Pull a fresh `nvcr.io/nvidia/pytorch:25.11-py3` image, no prior
   unsloth-studio state on the host volume.
2. Run `containers/unsloth-studio.sh`.
3. The inner cmd (line 43-71) ends with
   `... unsloth studio setup && ... unsloth studio -H 0.0.0.0 -p ${PORT}`.
   If `unsloth studio setup` fails (e.g. studio venv at
   `/root/.unsloth/studio/unsloth_studio` cannot be provisioned because
   of network, disk, or a half-installed previous run), the `&&` chain
   aborts.
4. The container exits. The poll-for-readiness loop on line 75 detects
   "Container exited unexpectedly", removes the container, and the user
   is dropped back to host shell — without a clear, surfaced error
   message above the docker logs scrollback.

**Generic impact**: Any user pulling the unsloth-studio container fresh
or after a failed install hits this. UX is confusing: the launcher
prints the URL banner (lines 21-31), claims studio is starting, and
then disappears.

**Suggested fix**: Decouple the studio UI bringup from the container
shell lifecycle. Either:

1. Replace `unsloth studio setup && ... unsloth studio ...` with
   `unsloth studio setup || { echo 'Studio setup failed — keeping
   container alive for debugging'; sleep infinity; }`, OR
2. Split studio bringup into a separate launcher
   (`unsloth-studio-ui.sh`) that runs against an already-up headless
   container, mirroring the headless/studio split that already exists
   for the docker-run layer.

Either approach avoids the silent host-shell drop.

**Resolution (2026-05-13, upstream commit e014af7 on dr-robert-li/dgx-toolbox#main):**
applied a hybrid of the original suggestions. The launcher now auto-bootstraps
the studio venv via the upstream-documented install.sh before
`unsloth studio setup` runs, when the venv directory is missing:

```bash
if [ ! -x /root/.unsloth/studio/unsloth_studio/bin/python ]; then
    echo "[unsloth-studio] venv missing ...; bootstrapping via install.sh"
    curl -fsSL https://unsloth.ai/install.sh | sh
fi && unsloth studio setup && ...
```

install.sh is idempotent; subsequent container starts skip the curl when
the venv exists. Removes the silent host-shell drop without sacrificing
the `&&` chain semantics that catch real setup failures.

---

## #5 (P0) — `bitsandbytes` wheel missing CUDA 13.x binary

**Files**: `containers/unsloth-studio.sh:44-67`, `containers/unsloth-headless.sh:50-72`
(installed transitively via the unsloth dep resolver block)

**Reproduction**:
1. Launch any unsloth container against the 26.02-py3 base (or the
   25.11-py3 base with CUDA 13.1 forward-compat, which is the current
   default for unsloth-studio.sh:42 / unsloth-headless.sh:48 once a
   user upgrades).
2. The unsloth dep resolver pulls `bitsandbytes==0.49.2`. Its wheel
   ships `libbitsandbytes_cuda121.so` and lower but no
   `libbitsandbytes_cuda131.so`.
3. On first import: `Configured CUDA binary not found at
   /usr/local/lib/python3.12/dist-packages/bitsandbytes/libbitsandbytes_cuda131.so`.
   Quantized model loads fall back to a CPU path or error out.

**Generic impact**: 4-bit / 8-bit quantization (advertised in
ngc-quickstart.sh:50-56 as a supported pre-installed feature) is broken
inside the toolbox's modern containers until upstream bitsandbytes ships
a CUDA 13 wheel.

**Suggested fix**: Pin `bitsandbytes==0.48.0` (last version that
gracefully falls back on CUDA 12.x and is known to work in mixed
CUDA-version environments) in the toolbox-managed pip layer until
upstream releases a CUDA 13 wheel. Document the pin and the rationale
in README.md, and add a tracking link to the upstream bitsandbytes
release notes so the pin can be lifted in a future toolbox release.

---

## #6 (P2) — Inconsistent project-mount UX between `ngc-pytorch.sh` and `unsloth-headless.sh` / `unsloth-studio.sh`

**Severity:** P2 (friction; workaround documented).

**Reproduction:**
- `ngc-pytorch.sh:8` auto-mounts the host's `$PWD` to `/workspace` (`-v "${PWD}:/workspace" -w /workspace`).
- `unsloth-headless.sh:43-44` mounts only `$HOME/.cache/huggingface` and `$HOME/unsloth-data` (the latter to `/workspace/work`). The host `$PWD` is not mounted unless the user sets `EXTRA_MOUNTS=...`.
- `unsloth-studio.sh:36-37` has the same mount profile as `unsloth-headless.sh`.

Users switching between containers run identical commands (e.g. `pip install -r config/requirements.txt`) and get `FileNotFoundError` in two of the three because there is no consistent project path inside.

**Generic impact:** Anyone migrating a workflow from `ngc-pytorch.sh` (which works) to `unsloth-headless.sh` or `unsloth-studio.sh` will hit this. Documentation (`README.md`, `example.bash_aliases`) mentions `EXTRA_MOUNTS` but does not flag the cross-container inconsistency.

**Suggested fix:** Either (a) standardise all three launchers to auto-mount `$PWD` to `/workspace/project` by default and use `$EXTRA_MOUNTS` only for additional mounts, or (b) document the asymmetry prominently in README.md and add a one-line note inside each container's startup banner ("Tip: pass `EXTRA_MOUNTS=\"$(pwd):/workspace/project\"` to mount your repo"). Option (a) is the lower-friction choice; option (b) is the lower-risk choice.

---

## #7 (P1) — Unsloth-trained MoE LoRA adapter `target_parameters` does not match PEFT 0.18.1 expectation

**Severity:** P1 (silent failure with workaround; affects any consumer of unsloth-trained MoE LoRA).

**Reproduction:**
Train a LoRA adapter against a MoE base (e.g. Qwen3-30B-A3B) inside `unsloth-headless.sh` with `target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]` (Unsloth's recommended pattern for MoE expert LoRA). Adapter saves correctly. Load with `PeftModel.from_pretrained` inside the same container env with peft==0.18.1, transformers==4.56.2. PEFT prints:

```
RuntimeWarning: target_parameters=['mlp.experts.gate_up_proj', 'mlp.experts.down_proj'] were set but no parameter was matched.
```

`modules_to_save` (e.g. `embed_tokens`, `lm_head`) loads correctly. Standard `target_modules` (q_proj, k_proj, v_proj, o_proj, etc.) also loads. Only the expert-MLP `target_parameters` silently fails to bind. Generated output then comes from BASE expert weights, not trained expert weights — appears working, scores at base-model quality.

**Generic impact:** Any user loading an Unsloth-trained MoE adapter outside of an Unsloth-`FastLanguageModel.from_pretrained()` call (i.e. via raw PEFT) gets a silent quality regression. There is no error, just a warning that's easy to miss.

**Suggested fix:** Either (a) document in the dgx-toolbox README that MoE LoRA adapters trained with `target_parameters` MUST be loaded via Unsloth's `FastLanguageModel.from_pretrained()` not raw `PeftModel.from_pretrained()`, OR (b) upstream a PEFT change so `target_parameters` raises an error (not a warning) when zero params match, OR (c) add a `verify_adapter_load.py` helper to dgx-toolbox that diffs expected-vs-loaded LoRA module names and exits non-zero if any are missing.

---

## #8 (P1) — `${VAR:-default}` template in recipe `model:` field not expanded by sparkrun

**Severity:** P1 (silent breakage of documented pattern with workaround).

**Reproduction:**

`deps/dgx-toolbox/recipes/eval-checkpoint.yaml:6` uses
```yaml
model: ${MODEL:-/models/checkpoint}
```
and `deps/dgx-toolbox/scripts/eval-checkpoint.sh:193` invokes
```bash
MODEL="$CHECKPOINT_DIR" sparkrun run "$EVAL_RECIPE_REF" --port "$EVAL_VLLM_PORT" --solo
```

sparkrun (vendor binary at `~/.local/bin/sparkrun`) does NOT expand the shell-style `${MODEL:-...}` template before resolving the model. The literal string `${MODEL:-/models/checkpoint}` is treated as an HF repo id and errors:

```
Repo id must use alphanumeric chars, '-', '_' or '.'. ... '${MODEL'.
Error: Failed to download model: ${MODEL:-/...
```

**Generic impact:** anyone copying the eval-checkpoint.yaml pattern to a new recipe and invoking `MODEL=<path> sparkrun run ...` gets a confusing repo-id-format error. The eval-checkpoint.sh workflow itself appears broken unless sparkrun has special-cased that recipe name (no evidence it has).

**Suggested fix:** either (a) have sparkrun expand `${VAR}` / `${VAR:-default}` template strings in recipe fields against the calling shell's environment (then document this clearly), or (b) replace eval-checkpoint.yaml's template with a plain default (`model: /models/checkpoint`) and document that callers MUST pass `-o model=<path>` to override (not via env var), or (c) keep current behaviour but explicitly document the env-var-doesn't-expand gotcha at the top of eval-checkpoint.yaml and in README.md.

The wp-finetune workaround for now: hardcode the absolute host path in the recipe and override with `-o model=...` per `sparkrun run --help`.

---

## #9 (P0) — sparkrun cannot serve local model directories at all

**Severity:** P0 (blocks the entire "load your fine-tuned checkpoint" workflow with NO recipe-level workaround).

**Reproduction:**

Any recipe with a `model:` field that points at a local directory (instead of an HF repo id) fails at the [3/6] "Distributing resources" step:

```
Failed to download model /absolute/host/path/to/checkpoint: Repo id must be
in the form 'repo_name' or 'namespace/repo_name': '/absolute/host/path/...'.
```

This holds whether the path is:
- absolute host path
- in-container path
- a `${VAR:-default}` env-substitution template (also blocked, see #8)

**Root cause** in source:
- `deps/dgx-toolbox/vendor/sparkrun/src/sparkrun/orchestration/distribution.py:541` → `distribute_model_from_local()` → `download_model()`.
- `deps/dgx-toolbox/vendor/sparkrun/src/sparkrun/models/download.py:350-400` (`download_model()`) unconditionally calls `huggingface_hub.snapshot_download(repo_id=...)`. There is no `os.path.exists()` branch, no `local://` prefix support, no `--model-source-type=local` CLI flag.

**Generic impact:** Any dgx-toolbox user who wants to serve a locally-trained checkpoint (the entire `eval-checkpoint.sh` workflow, every fine-tune-and-evaluate pipeline) hits this wall. There is currently no documented sparkrun-level workaround. Toolbox consumers must either:
- Symlink the checkpoint into the HF cache directory tree (fragile — needs fabricated commit SHA + `refs/main` file; can be invalidated by `huggingface_hub` cache validation).
- Bypass sparkrun entirely and `docker run` the underlying vLLM image directly (what wp-finetune ended up doing via `scripts/serve_30_70_vllm.sh`).
- Wait for sparkrun upstream fix.

**Suggested fix:** add a path-existence check at the top of `models/download.py` `download_model()`:

```python
# Pseudo-code addition near download_model() entry
from pathlib import Path
candidate = Path(model_ref)
if candidate.is_absolute() and candidate.is_dir() and (candidate / "config.json").is_file():
    return str(candidate)  # treat as already-distributed local checkpoint
```

Combined with the recipe `${VAR}` expansion fix from #8, this enables the documented `MODEL=<path> sparkrun run eval-checkpoint` flow.

---


### Watch Log — 2026-05-12T07:48:04Z

- **DRIFT**: submodule HEAD `95d7b30e109a` differs from `https://github.com/dr-robert-li/dgx-toolbox.git` `main` (`00a457a0daa9`). Inspect: `git -C deps/dgx-toolbox log --oneline 95d7b30e..00a457a0`.
- **ANTI-PATTERN**: `containers/unsloth-studio.sh` still chains `unsloth studio setup && ... unsloth studio ...`. Root cause of issue #4 (P1: container silent exit when studio venv at `/root/.unsloth/studio/unsloth_studio` is missing). #4 suggested fix not upstreamed.
- **EXIT**: container `unsloth-headless` (ba45884f5c0f) exited with code **137** (Exited (137) 8 hours ago). Inspect: `docker logs --tail 200 ba45884f5c0f`. Code 137 = SIGKILL (OOM or manual kill); check host memory pressure.


### Watch Log — 2026-05-12T20:36:38Z

- **DRIFT**: submodule HEAD `21bb3e533f0c` differs from `https://github.com/dr-robert-li/dgx-toolbox.git` `main` (`00a457a0daa9`). Inspect: `git -C deps/dgx-toolbox log --oneline 21bb3e53..00a457a0`.

## How to add issues

Append new entries ABOVE this footer, numbered sequentially, and bump
the `Last updated` date at the top. Keep each entry under ~250 words,
include a file path + line number citation for the offending code, and
state both the reproduction steps and the suggested fix. Anything
project-specific (e.g. a single consumer's training hyperparameters or
dataset paths) belongs in that project's own tracker, not here — the
purpose of this file is to surface issues that affect *any* dgx-toolbox
user running HF / PyTorch / Unsloth workflows.
