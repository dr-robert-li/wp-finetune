VERDICT: PASS

# W5 deploy: independent evaluation

Scope: Dockerfile, docker-compose.yml, docker/llama-entrypoint.sh, docker/allinone.sh, .dockerignore, judged against
SPEC R1.1/R3.1(A2)/R3.2/A6, PLAN W5 locked entry, binding changes 7, 11, 12, research/r4_deploy.md and build_notes/W5.md.

## Conformance (all met)
- Pinned `ghcr.io/ggml-org/llama.cpp:server-cuda13-b11515` in compose (l.4) and allinone FROM (Dockerfile l.13). Never `server-cuda`. (A6)
- Download: `curl -fL -C - --retry 5` to `$M.part`, `mv` only on success, server starts only if `$M` exists (A2, R3.1).
  The HF URL resolves: HEAD on both IQ4_XS and small-IQ3_XXS-experts gives `302`, `accept-ranges: bytes`,
  `x-linked-size` 16579196960 / 12869398560. The repo file list matches.
- `WPJUDGE_MODEL_FILE` is set in both services with the same default (binding 7). The filename guard rejects `/`, `..*` and empty values.
- llama flags: `-ngl 999 --jinja --chat-template-kwargs '{"enable_thinking": false}' --parallel 4 -c 32768`, with no `-ctk/-ctv` (R3.2, small-file caveat).
  `llama-server --help` in b11515 lists `--chat-template-kwargs`, `-np/--parallel`, `--jinja` and `-ngl`.
- Healthcheck: `curl -fsS :8080/health` with `start_period: 40m` and `start_interval: 10s`. The harness has `depends_on: llama: service_healthy`. The publish is `127.0.0.1:8000:8000` (binding 12). The llama port is not published.
- The harness target uses `COPY . /app/` with build context `.` (binding 11). allinone uses `/opt/wpjudge` because `/app` holds llama-server. That is a justified deviation.
- `WPJUDGE_WORKERS`, `WPJUDGE_API_KEY` (empty means auth is off) and `WPJUDGE_JOBS_DIR=/data/jobs` all map to real `config.py` fields. I checked this with `config.load()` inside both images.
- `.dockerignore` does not drop anything needed. `wpjudge/` and `pyproject.toml` are present (pyproject has no `readme`, so `*.md` is safe). The docker/*.sh files are present. The compose entrypoint is a bind mount, so it is unaffected.

## Defects
1. minor, docker/llama-entrypoint.sh:8-12: nothing checks integrity. A `.part` that is stale or corrupt, for example after the HF file is re-uploaded, gets resumed and stitched into a bad GGUF. A `.part` larger than the remote file gets a 416, curl returns rc 0, and the file is moved into place. Fix if wanted: compare against the size or `X-Linked-ETag` sha256 before `mv`, or `rm` the `.part` when llama fails to load. The current behaviour is acceptable for v1.
2. minor, docker/allinone.sh:4-9 and Dockerfile:26: `sh` is PID 1 during the wait loop and ignores SIGTERM, so `docker stop` takes 10 s and then SIGKILL. After `exec`, uvicorn is PID 1 and llama-server is an orphaned child that never gets a SIGTERM. Fix: document `docker run --init`, or add `tini`.
3. minor, Dockerfile:24-25: the allinone HEALTHCHECK probes llama only. If llama-server dies after the harness has started, the container stays up and goes unhealthy, and plain `docker run` does not restart it. Fix: make allinone.sh `wait` on either process, or document it.
4. minor, docker/allinone.sh:4: allinone forwards no extra args to llama-entrypoint.sh, so `docker run ... wpjudge:allinone --foo` drops `--foo` without a message. In allinone, `LLAMA_PARALLEL` is also not tied to `WPJUDGE_WORKERS` the way compose ties them. Fix: `"$@"` passthrough plus `WPJUDGE_WORKERS=${WPJUDGE_WORKERS:-${LLAMA_PARALLEL:-4}}`, or document both.
5. minor, docker-compose.yml:6-8: `LLAMA_SLOT_CTX` exists in the entrypoint but compose does not pass it through, so per-slot ctx cannot be configured from `.env` (R3.2 says "both configurable"). Fix: add `LLAMA_SLOT_CTX: ${LLAMA_SLOT_CTX:-8192}`.
6. minor, docker-compose.yml:17: if the first download outlasts 40 min (below about 7 MB/s), llama turns unhealthy. `compose up` then fails the harness dependency and does not retry after llama recovers. This is already a known risk in r4. Document "re-run `docker compose up -d`" in the README (W7).
7. minor, Dockerfile:4-10: the harness image runs as root and also ships `tests/` (316 KB). Hardening: add a non-root `USER` that owns `/data/jobs`. Optional: add `tests` to .dockerignore. Not required by SPEC.

There are no blockers and no majors. Over-engineering: none found. The files are short, there is one entrypoint shared by both deploy paths, and no custom llama image.

## What I ran
- `sh -n` on both scripts: OK. `docker compose config -q`: OK.
- Rebuilt `wpjudge:harness` and `wpjudge:allinone`: OK (cached). `docker ps -a` is empty afterwards. No GPU containers were started.
- In the harness image, `import wpjudge.cli, wpjudge.api, tree_sitter_php` works and `config.load()` gives `jobs_dir=/data/jobs, workers=4`. In allinone, `config.load()` gives `api_key=''`, base_url `http://127.0.0.1:8080/v1`, jobs `/data/jobs`, and `/opt/venv/bin/wpjudge` is on PATH.
- In the llama image (dash `/bin/sh`, curl 8.5.0), `false & kill -0 $!` detects the dead child, so the allinone loop exits when llama dies instead of hanging.
- Partial-download probe with the image's curl against the real HF URL:
  - A sparse `.part` at the full size gets rc 0 and is unchanged. This is the "killed between curl and mv" case.
  - A `.part` 960 B short gets rc 0 and resumes to exactly 16579196960 B. Resume works through the Xet redirect.
- `curl -sI` on both GGUF resolve URLs, and the HF API sibling list.
