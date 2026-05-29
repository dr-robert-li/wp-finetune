"""Shared vLLM boot/wait/generate/stop helper for Phase 4.4 smoke gate.

Used by PR2b (baseline output generation) and PR2d (Stage 2 served smoke).
Boots vLLM on a given MODEL_DIR via scripts/serve_30_70_vllm.sh, polls the
OpenAI-compatible /v1/models endpoint until healthy (900s Pitfall-3 guard),
generates manifest prompts at temperature=0, then stops the container.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVE_SCRIPT = str(PROJECT_ROOT / "scripts" / "serve_30_70_vllm.sh")
BOOT_TIMEOUT_SEC = 900   # Pitfall 3: 30B bf16 weight load can exceed 600s on GB10


class VllmBootTimeout(RuntimeError):
    pass


def boot_vllm(model_dir: str, name: str, port: int, gpu_mem_util: float = 0.55) -> None:
    """Launch vLLM container (detached) for model_dir."""
    env = {
        "CONTAINER_NAME": name,
        "PORT": str(port),
        "MODEL_DIR": str(PROJECT_ROOT / model_dir) if not str(model_dir).startswith("/") else model_dir,
        "GPU_MEM_UTIL": str(gpu_mem_util),
    }
    import os
    full_env = {**os.environ, **env}
    print(f"[vllm] booting {name} on :{port} model={model_dir} gpu_mem_util={gpu_mem_util}")
    subprocess.run(["bash", SERVE_SCRIPT], env=full_env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def wait_healthy(port: int, name: str, timeout: int = BOOT_TIMEOUT_SEC) -> str:
    """Poll /v1/models until the server answers; return served model name.

    Raises VllmBootTimeout if not healthy within timeout (Mode D diagnosis).
    """
    import openai
    client = openai.OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout:
        # bail early if container died
        alive = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True,
        ).stdout
        if name not in alive:
            raise VllmBootTimeout(f"container {name} exited during boot (see docker logs {name})")
        try:
            models = client.models.list()
            served = models.data[0].id
            print(f"[vllm] healthy after {time.time()-t0:.0f}s; served model={served}")
            return served
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(10)
    raise VllmBootTimeout(f"vLLM {name} not healthy within {timeout}s (last: {last_err})")


def generate(port: int, served_model: str, prompts: list[dict], max_tokens: int = 512) -> list[str]:
    """Generate completions for manifest prompts at temperature=0."""
    import openai
    client = openai.OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")
    outs = []
    for p in prompts:
        try:
            resp = client.chat.completions.create(
                model=served_model,
                messages=[{"role": "user", "content": p["instruction"]}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            outs.append(resp.choices[0].message.content or "")
        except Exception as e:  # noqa: BLE001
            print(f"[vllm] gen error idx {p.get('source_val_idx')}: {e}")
            outs.append("")
    return outs


def stop_vllm(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                   stderr=subprocess.STDOUT, check=False)
    print(f"[vllm] stopped {name}")
