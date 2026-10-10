# Build from harness/: docker build --target harness|allinone -t wpjudge:<target> .

# --- thin harness (compose talks to the llama service) ---
FROM python:3.12-slim AS harness
WORKDIR /app
COPY . /app/
RUN pip install --no-cache-dir /app
ENV WPJUDGE_JOBS_DIR=/data/jobs
EXPOSE 8000
CMD ["wpjudge", "serve", "--host", "0.0.0.0", "--port", "8000"]

# --- all-in-one: llama-server + harness; GGUF mounted at /models, never baked ---
FROM ghcr.io/ggml-org/llama.cpp:server-cuda13-b11515 AS allinone
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv \
 && rm -rf /var/lib/apt/lists/* && python3 -m venv /opt/venv
# /app holds llama-server here, so the harness source goes to /opt/wpjudge.
COPY . /opt/wpjudge/
RUN /opt/venv/bin/pip install --no-cache-dir /opt/wpjudge
COPY docker/llama-entrypoint.sh /llama-entrypoint.sh
COPY docker/allinone.sh /allinone.sh
ENV PATH=/opt/venv/bin:$PATH WPJUDGE_BASE_URL=http://127.0.0.1:8080/v1 WPJUDGE_JOBS_DIR=/data/jobs
EXPOSE 8000
# The harness API has no /health; llama readiness gates the harness start in allinone.sh.
HEALTHCHECK --start-period=40m --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/health || exit 1
ENTRYPOINT ["/bin/sh", "/allinone.sh"]
