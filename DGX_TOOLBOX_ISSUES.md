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

## #10 (P0, RESOLVED upstream @ 75681b8) — `unsloth install.sh` pins `torchcodec==0.10.0` which has no aarch64 wheel

**Files**: upstream `unsloth install.sh` (fetched via `curl -fsSL https://unsloth.ai/install.sh | sh`), invoked by `containers/unsloth-studio.sh:51` after the #4 auto-bootstrap fix lands.

**Reproduction**:
1. Run `containers/unsloth-studio.sh` on an aarch64 host (DGX Spark / Grace).
2. On fresh container, the #4 bootstrap fix detects missing venv and calls `curl -fsSL https://unsloth.ai/install.sh | sh`.
3. install.sh creates `/root/.unsloth/studio/unsloth_studio` (Python 3.13.9 venv) and begins installing `extras-no-deps.txt`.
4. uv resolver aborts:
    ```
    × No solution found when resolving dependencies:
    ╰─▶ Because torchcodec==0.10.0 has no wheels with a matching platform tag
        (e.g., `manylinux_2_39_aarch64`) and you require torchcodec==0.10.0, we
        can conclude that your requirements are unsatisfiable.
        hint: Wheels are available for `torchcodec` (v0.10.0) on the following
        platforms: `manylinux_2_28_x86_64`, `macosx_12_0_arm64`, `win_amd64`
    ```
5. pip fallback also fails: `Could not find a version that satisfies the requirement torchcodec==0.10.0`. install.sh exits 1.
6. `unsloth studio setup` exits 1 immediately afterward. Studio UI never starts.

**Generic impact**: Any aarch64 dgx-toolbox user (DGX Spark, Grace, ARM workstations) running `unsloth-studio.sh` after the #4 bootstrap fix lands. Surfaced 2026-05-13 on `nvcr.io/nvidia/pytorch:25.11-py3` (manylinux_2_39_aarch64). `torchcodec==0.11.x` does have aarch64 wheels.

**Suggested fix** (in dgx-toolbox, since upstream `install.sh` is outside this repo): patch `containers/unsloth-studio.sh` post-bootstrap to relax the torchcodec pin before `unsloth studio setup` runs. The launcher already does `pip uninstall -y torchcodec 2>/dev/null` at lines 53 + 57 — make it idempotent + cover the venv too:

```bash
# After install.sh bootstrap, before `unsloth studio setup`:
VENV_PY=/root/.unsloth/studio/unsloth_studio/bin/python
[ -x "$VENV_PY" ] && "$VENV_PY" -m pip uninstall -y torchcodec 2>/dev/null
[ -x "$VENV_PY" ] && "$VENV_PY" -m pip install --no-deps "torchcodec>=0.11,<0.12" 2>/dev/null
```

Long-term: file upstream issue against `unsloth-studio` extras-no-deps.txt to bump torchcodec pin to a multi-arch version.

---

## #11 (P0, RESOLVED upstream @ 75681b8) — `unsloth studio` resolves to system Python, not the bootstrapped venv

**File**: `containers/unsloth-studio.sh:55` (`unsloth studio -H 0.0.0.0 -p ${PORT}`)

**Reproduction**:
1. Run `containers/unsloth-studio.sh` after the #4 bootstrap fix lands.
2. install.sh provisions `/root/.unsloth/studio/unsloth_studio` (Python 3.13.9 venv).
3. Launcher invokes `unsloth studio -H 0.0.0.0 -p 8000`. The `unsloth` binary on `$PATH` resolves to the system Python 3.12 install (the launcher's earlier `python /tmp/install-deps.py unsloth unsloth_zoo` step at line 47 installs unsloth into the system site-packages).
4. The system `unsloth studio` entry-point imports from `/usr/local/lib/python3.12/dist-packages/studio/backend/run.py`:
    ```
    File "/usr/local/lib/python3.12/dist-packages/studio/backend/loggers/handlers.py", line 22
    import structlog
    ModuleNotFoundError: No module named 'structlog'
    ```
5. Studio crashes immediately. The venv at `/root/.unsloth/studio/unsloth_studio` is never used at runtime.

**Generic impact**: Two parallel unsloth installs in the container — system Python 3.12 (line 47 pip) and venv Python 3.13 (install.sh). The launcher's last line uses the wrong one. install.sh provisions a clean studio environment; the launcher then bypasses it. Affects every user who hits the #4 bootstrap fix path.

**Suggested fix**: dispatch `unsloth studio` through the venv after `setup` succeeds. Two options:

```bash
# Option A: source the venv before invoking studio
source /root/.unsloth/studio/unsloth_studio/bin/activate
unsloth studio setup && unsloth studio -H 0.0.0.0 -p ${PORT}

# Option B: call the venv binary directly
VENV=/root/.unsloth/studio/unsloth_studio
"$VENV/bin/unsloth" studio setup && "$VENV/bin/unsloth" studio -H 0.0.0.0 -p ${PORT}
```

Option B is more robust (no `$PATH` leakage across bash subshells inside the `bash -c '...'` block). Combined with the #10 torchcodec patch, this completes the bootstrap-to-running-studio chain on aarch64.

---


### Watch Log — 2026-05-12T07:48:04Z

- **DRIFT**: submodule HEAD `95d7b30e109a` differs from `https://github.com/dr-robert-li/dgx-toolbox.git` `main` (`00a457a0daa9`). Inspect: `git -C deps/dgx-toolbox log --oneline 95d7b30e..00a457a0`.
- **ANTI-PATTERN**: `containers/unsloth-studio.sh` still chains `unsloth studio setup && ... unsloth studio ...`. Root cause of issue #4 (P1: container silent exit when studio venv at `/root/.unsloth/studio/unsloth_studio` is missing). #4 suggested fix not upstreamed.
- **EXIT**: container `unsloth-headless` (ba45884f5c0f) exited with code **137** (Exited (137) 8 hours ago). Inspect: `docker logs --tail 200 ba45884f5c0f`. Code 137 = SIGKILL (OOM or manual kill); check host memory pressure.


### Watch Log — 2026-05-12T20:36:38Z

- **DRIFT**: submodule HEAD `21bb3e533f0c` differs from `https://github.com/dr-robert-li/dgx-toolbox.git` `main` (`00a457a0daa9`). Inspect: `git -C deps/dgx-toolbox log --oneline 21bb3e53..00a457a0`.


### Watch Log — 2026-05-12T20:40:23Z

- **EXIT**: container `unsloth-headless` (ba45884f5c0f) exited with code **137** (Exited (137) 21 hours ago). Inspect: `docker logs --tail 200 ba45884f5c0f`. Code 137 = SIGKILL (OOM or manual kill); check host memory pressure.


### Watch Log — 2026-05-12T20:48:55Z

- **DRIFT**: submodule HEAD `e014af7ff8e9` differs from `https://github.com/dr-robert-li/dgx-toolbox.git` `main` (`e42656493645`). Inspect: `git -C deps/dgx-toolbox log --oneline e014af7f..e4265649`.

## Cascading Fix Plan

Drafted 2026-05-13. The 11 numbered issues above plus three Watch Log
entries (DRIFT × 2, EXIT 137 × 1, ANTI-PATTERN × 1) form **eight causal
chains**. Several "fixed" issues are only partially mitigated
(template-only, opt-in, or worked around downstream rather than fixed at
root); several open issues are children of a parent fix that has already
landed. This section consolidates them into one dependency-ordered view.

Status tags: **OPEN** = no mitigation. **MITIGATED** = workaround
shipped, root cause still open. **RESOLVED** = root cause fixed.

**Live reproduction (2026-05-13 ~07:00 GMT+10)** confirming Chain B
fires on the user's aarch64 host the moment `unsloth-studio.sh` is run:

```
ERROR: Could not find a version that satisfies the requirement torchcodec==0.10.0
       (from versions: 0.0.0.dev0, 0.11.0, 0.11.1)
ERROR: No matching distribution found for torchcodec==0.10.0
  error          studio setup failed (exit code 1)
...
File "/usr/local/lib/python3.12/dist-packages/studio/backend/loggers/handlers.py", line 22
    import structlog
ModuleNotFoundError: No module named 'structlog'
```

Lines 1-3 = #10 (aarch64 torchcodec). Lines 5-9 = #11 (system Python 3.12
shadow). Both in one launch — must be patched together.

### Chain A — HF stack drift (#2, #3, partially #5) — OPEN

```
NGC base image transformers 5.x line
        │
        ├── #2 PEFT 0.19 WeightConverter signature mismatch  (P0)
        │
        ├── unsloth latest auto-pull (#3, P0)
        │     └── transformers >=5.0 + tokenizers >=0.22 + hf-hub >=1.0
        │           → adapter load failure on every public 4.x checkpoint
        │
        └── bitsandbytes 0.49.2 transitively pulled (#5, P0)
              └── no libbitsandbytes_cuda131.so → 4-/8-bit quant broken
```

**Partial mitigations already shipped**:

- `setup/requirements-gpu.example.txt` (commit 7b70d00) pins
  transformers 4.46.*, tokenizers 0.20.*, hub 0.26.*, peft 0.13.*,
  accelerate, trl, datasets, bitsandbytes==0.48.0. **Opt-in only** —
  user must copy template → `$HOME/requirements-gpu.txt`. NGC
  launchers consume it; unsloth launchers do NOT.
- `UNSLOTH_VERSION` env var (commits 6a673db / 73ca2da) lets user pin
  unsloth. Does NOT pin transitives — latest unsloth's *runtime*
  imports transformers 5.x APIs even if `--no-deps` blocks the
  install-time upgrade.
- `install-deps.py` (commit 00a457a) supports `# no-walk` marker and
  `-r <file>` invocation.
- README "Known Issues and Compatibility" (commit 4f5a34d) documents
  the trade-offs.

**Remaining work** (apply in order):

1. `containers/unsloth-headless.sh` and `containers/unsloth-studio.sh`:
   detect `$HOME/requirements-gpu.txt`. If present, prepend
   `-r "$HOME/requirements-gpu.txt"` to the `install-deps.py`
   invocation so the pinned HF stack flows into unsloth containers,
   not just NGC ones.
2. README: add explicit "for reproducible HF pins across **all three**
   container families, copy `setup/requirements-gpu.example.txt` →
   `$HOME/requirements-gpu.txt`" line.
3. Long-term: bake the pin file into the `setup/install.sh` step so
   it's automatic on fresh checkout.

### Chain B — Studio bootstrap regression (#4 fix → #10 + #11) — OPEN (firing live as of 2026-05-13)

```
#4 silent host-shell drop on missing studio venv  (P1)
        │
        └── Fix: curl install.sh | sh   (commits e014af7, e426564)
                  │
                  ├── #10 install.sh pins torchcodec==0.10.0; no aarch64 wheel  (P0)
                  │       → uv resolver fails → pip fallback fails → exit 1
                  │       → studio setup exit 1
                  │
                  └── #11 unsloth on $PATH = system Python 3.12 (install-deps.py output)
                          NOT venv Python 3.13 (install.sh venv)             (P0)
                          → /usr/local/lib/python3.12/dist-packages/studio/...
                            ModuleNotFoundError: structlog
                          → studio crashes immediately
```

**Why #10 + #11 must ship in one patch**: even if #10 is fixed in
isolation (venv exists), #11 still routes execution to the wrong
Python. Even if #11 is fixed in isolation (`$VENV/bin/unsloth`
invoked), #10 prevents the venv from existing on aarch64 in the first
place. Half-fixes do nothing.

**Unified fix** to `containers/unsloth-studio.sh` (current lines 57-66):

```bash
bash -c '\
  python /tmp/install-deps.py '"${UNSLOTH_SPEC}"' && \
  pip uninstall -y torchcodec 2>/dev/null; \
  VENV=/root/.unsloth/studio/unsloth_studio; \
  if [ ! -x "$VENV/bin/python" ]; then \
    echo "[unsloth-studio] venv missing; bootstrapping via install.sh"; \
    curl -fsSL https://unsloth.ai/install.sh | sh || true; \
  fi; \
  if [ "$(uname -m)" = "aarch64" ] && [ -x "$VENV/bin/python" ]; then \
    echo "[unsloth-studio] aarch64 host — overriding torchcodec pin in venv"; \
    "$VENV/bin/python" -m pip uninstall -y torchcodec 2>/dev/null; \
    "$VENV/bin/python" -m pip install --no-deps "torchcodec>=0.11,<0.12"; \
  fi && \
  "$VENV/bin/unsloth" studio setup && \
  "$VENV/bin/unsloth" studio -H 0.0.0.0 -p '"${PORT}"''
```

Three deltas vs. current code:

1. Hoist `VENV=...` once.
2. After bootstrap, if `uname -m == aarch64`, force-replace torchcodec
   **inside the venv** (fixes #10 at the affected interpreter, not the
   system interpreter).
3. Invoke `"$VENV/bin/unsloth"` (absolute path) for both `setup` and
   the studio server (fixes #11 — bypasses `$PATH` shadow from the
   system-Python install-deps.py install).

**Cross-chain interaction with Chain A's #5**: install.sh provisions
Python 3.13. Bitsandbytes 0.48.0 (the #5 pin) must have a Python 3.13
wheel; if not, the venv loses quantisation capability even though the
system Python 3.12 still has it. Verify with
`"$VENV/bin/python" -c "import bitsandbytes"` after the unified fix
lands; if missing, pin bitsandbytes in the venv post-bootstrap or fall
back to Python 3.12 for studio.

### Chain C — sparkrun local-checkpoint (#8, #9) — MITIGATED, root still open

```
#8 ${VAR:-default} template not expanded by sparkrun  (P1)
        │
        └── #9 sparkrun.download_model() unconditionally calls         (P0)
            huggingface_hub.snapshot_download() — no path-exists branch
                  │
                  └── Workaround: scripts/eval-checkpoint.sh dual-path
                      routing (_launch_vllm_direct, commit d155307)
                      — bypasses sparkrun entirely for local checkpoints
```

Long-term root fix to
`vendor/sparkrun/src/sparkrun/models/download.py:350-400` (the
`os.path.exists()` branch from #9's suggested fix) still required. No
new dgx-toolbox-side action; tracked under #9.

### Chain D — Project mount asymmetry (#6) — RESOLVED

Commit 73ca2da auto-mounts `${PWD}:/workspace/project` in both unsloth
launchers (current `unsloth-studio.sh:52`, `unsloth-headless.sh:58`).
Parity with `ngc-pytorch.sh` achieved. README cross-references the
behaviour.

### Chain E — MoE LoRA silent fail (#7) — MITIGATED, upstream still open

`scripts/verify_adapter_load.py` (commit bfc60d4) diffs expected vs.
loaded LoRA module names and exits non-zero on mismatch. README
documents the recommended post-training invocation. Upstream PEFT
change (warning → error on zero-match `target_parameters`) remains the
right long-term fix and is not blocked by dgx-toolbox.

### Chain F — Container pre-flight (#1) — RESOLVED

`require_files` helper in `lib.sh` plus call sites in
`ngc-pytorch.sh` / `ngc-jupyter.sh` / `unsloth-*.sh` (commits 67ca82a,
3f27d17). Silent host-shell drop on missing host files is gone. No
remaining action.

### Chain G — Operational signals (Watch Log) — OPEN, observation-driven

```
Watch Log 2026-05-12T07:48:04Z / 20:40:23Z
  ├── DRIFT: submodule HEAD differs from upstream main         (informational)
  │   → wp-finetune's vendored dgx-toolbox is stale; cascades any
  │     upstream fix to be re-pulled before consumption.
  │
  ├── ANTI-PATTERN: unsloth studio setup && unsloth studio chain
  │   → root of #4. Resolved upstream by commits e014af7 / e426564.
  │
  └── EXIT 137 on unsloth-headless container                    (OPEN)
      → SIGKILL; OOM or manual kill. Repeated 21 hours apart.
      → No host-memory pre-flight; no swap monitor; no graceful
        OOM message. Likely cascade with Chain A — transformers 5.x
        memory footprint regression on small-VRAM hosts.
```

**Remaining work**:

1. Launcher-side pre-flight memory check: warn if `MemAvailable` (or
   `nvidia-smi --query-gpu=memory.free`) below a threshold matching the
   requested unsloth model class.
2. On container exit, surface a recognizable banner — `docker inspect`
   → `OOMKilled: true` → print "OOM — try smaller batch size or
   model".
3. Capture exit-code interpretation in README "Known Issues and
   Compatibility".

### Chain H — Submodule freshness (Watch Log DRIFT × 2) — OPEN, process-level

Two DRIFT entries flag wp-finetune's `deps/dgx-toolbox/` lagging
upstream `dr-robert-li/dgx-toolbox`. Not a code bug; a workflow bug.
Any patch from Chains A, B, or G must be (a) committed upstream,
(b) the submodule pointer in wp-finetune bumped, (c) re-verified before
declaring fixed. Captured here so the executor does not fix dgx-toolbox
in isolation and leave wp-finetune still hitting the old code.

### Cross-cascade dependency matrix

| Fix                                | Depends on                                | Unblocks                            |
|------------------------------------|-------------------------------------------|-------------------------------------|
| Chain B (#10 + #11 unified patch)  | Chain F (`require_files`) — already done  | Live studio launch on aarch64       |
| Chain A (HF pin propagation)       | `install-deps.py -r` support — done       | Reproducible adapter loads          |
| Chain G (OOM banner + pre-flight)  | Chain A (pins trim memory footprint)      | Headless exit clarity               |
| Chain H (submodule bump)           | A and B landed upstream                   | wp-finetune reproducibility         |

### Sequencing rationale

1. **Chain B first** — actively firing on the user's host (see live
   trace above). Unblocks studio entirely.
2. **Chain A second** — quietly corrupts adapter loads; affects
   reproducibility for every downstream consumer.
3. **Chain G third** — operational hygiene; depends on A's memory
   profile.
4. **Chain H last** — process step; coordinates wp-finetune submodule
   bump after A + B land upstream.

Chains C, D, E, F are out of the critical path; status notes only.

---


### Watch Log — 2026-05-12T22:18:58Z

- **DRIFT**: submodule HEAD `e42656493645` differs from `https://github.com/dr-robert-li/dgx-toolbox.git` `main` (`35ec390a01f9`). Inspect: `git -C deps/dgx-toolbox log --oneline e4265649..35ec390a`.

## How to add issues

Append new entries ABOVE this footer, numbered sequentially, and bump
the `Last updated` date at the top. Keep each entry under ~250 words,
include a file path + line number citation for the offending code, and
state both the reproduction steps and the suggested fix. Anything
project-specific (e.g. a single consumer's training hyperparameters or
dataset paths) belongs in that project's own tracker, not here — the
purpose of this file is to surface issues that affect *any* dgx-toolbox
user running HF / PyTorch / Unsloth workflows.
