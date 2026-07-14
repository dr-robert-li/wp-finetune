#!/usr/bin/env python
"""Phase 21 Plan 06 Task 2 -- JUDGE-03 literal path: merge the promoted judge
seed (Task 1's best_single_seed) onto the local new base, serve it via
serve_base20_vllm.sh at an 8192-token generation cap, capture the wp_judge val
prompts over the vLLM endpoint, and score with the unmodified
scripts/relabel/eval_relabel.py.

Reuses, unchanged:
  - the fused-expert download+merge pattern (scripts/build_gen03_merge.py,
    proven in 21-05) -- same _download_promoted_adapter/_run_merge structure,
    parameterized for the judge seed instead of the gen adapter.
  - boot_vllm/wait_healthy/generate/stop_vllm (scripts/_p0_vllm_smoke_serve.py)
    for the real-generation warm-up gate + base-vs-merged diff.
  - scripts/sieve_capture_judge_http.py's capture() -- the EXISTING HTTP judge
    capture (index-aligned wp_judge_startswith filter, _judge_create's RC-A
    enable_thinking=False guard, user-message-only prompt -- no injected
    system rubric, matching the exact training/Tinker-capture prompt shape).
  - scripts/relabel/eval_relabel.py (unmodified) for the final rho + CI.

MAX_MODEL_LEN is raised to 16384 for this serve (longest wp_judge val prompt
measured at 2288 tokens; the default 8192 would leave no room for an 8192-
token completion on top of the prompt -- this would silently re-truncate
exactly the failure mode Pitfall 4 warns about, just moved from the capture
side to the serve side).

PRESERVES all 3 seed manifests (no deletion) -- the 3-MERGED-checkpoint vLLM-
served ensemble is deferred to packaging (Phase 27).

Must run under the Tinker venv for the merge step (tinker_cookbook), and can
run under the project/conda env for everything else -- mirrors the two-venv
split established across 21-01..21-05 (invoked here as .venv-tinker/bin/python
scripts/build_judge03_merge_serve.py, since merge_adapter.py is a subprocess
call regardless and boot/serve/capture/score are pure openai+subprocess calls
that work under either env, but PYTHONPATH sys.path insert covers the local
scripts/eval package needed here).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CAPTURE_RHO_PATH = PROJECT_ROOT / "output" / "base21" / "judge03_capture_rho.json"
OUT_DIR = PROJECT_ROOT / "output" / "base21"
CONFIG_PATH = "config/train_config_v4.yaml"
EXPECTED_MODULES_MANIFEST = OUT_DIR / "moe_merge_probe.json"  # same MoE-only architecture as gen (21-05)
DATASET = "data/reasoning_dataset/openai_val.jsonl"
DIFF_PROMPT = "<wp_judge> Evaluate this WordPress code:\n\n```php\nfunction f() { return 1; }\n```"
SERVE_SCRIPT = str(PROJECT_ROOT / "scripts" / "serve_base20_vllm.sh")
PORT = 8025
GPU_MEM_UTIL = 0.80
MAX_MODEL_LEN = 16384  # 2288-token longest wp_judge prompt + 8192-token completion cap + margin
MAX_TOKENS = 8192
BOOT_TIMEOUT_SEC = 1200  # 67 GiB base, Pitfall 3 lesson


def _adapter_dir(seed: int) -> Path:
    return OUT_DIR / f"judge03_s{seed}_adapter"


def _merged_dir(seed: int) -> str:
    return f"models/Qwen3.6-35B-A3B-judge-v4-s{seed}-merged"


def _download_promoted_adapter(seed: int) -> str:
    """Mirrors build_gen03_merge.py's WR-10 member-validated archive download
    + extraction, parameterized for the promoted judge seed's manifest."""
    import tinker

    manifest_path = PROJECT_ROOT / "output" / "tinker" / f"wp-judge-v4-s{seed}-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    promoted_name = manifest["promoted"]
    sampler_path = next(c["sampler_path"] for c in manifest["checkpoints"] if c["name"] == promoted_name)
    print(f"[judge03-merge] seed {seed} promoted checkpoint: {promoted_name} -> {sampler_path}", flush=True)

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    print("[judge03-merge] requesting archive URL ...", flush=True)
    resp = rc.get_checkpoint_archive_url_from_tinker_path(sampler_path).result()
    url = getattr(resp, "url", None) or getattr(resp, "archive_url", None)
    if not url:
        raise RuntimeError(f"no archive URL in response: {resp!r}")

    adapter_dir = _adapter_dir(seed)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tar_path = adapter_dir / "checkpoint.tar"
    print(f"[judge03-merge] downloading archive -> {tar_path}", flush=True)
    r = subprocess.run(
        ["curl", "-L", "--fail", "-C", "-", "--retry", "5", "--retry-delay", "10",
         "-o", str(tar_path), url],
        timeout=3600,
    )
    if r.returncode != 0:
        raise RuntimeError(f"curl download failed (exit {r.returncode}) for {tar_path}")
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            if name in ("adapter_config.json", "adapter_model.safetensors") and m.isfile():
                m.name = name
                tf.extract(m, adapter_dir)

    for required in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_dir / required).exists():
            raise RuntimeError(f"tinker archive missing {required} after extraction")

    return sampler_path


def _run_merge(seed: int) -> dict:
    merged_dir = _merged_dir(seed)
    guard_receipt_path = OUT_DIR / f"_judge03_s{seed}_merge_guard_result.json"
    cmd = [
        sys.executable, "scripts/merge_adapter.py",
        "--config-path", CONFIG_PATH,
        "--adapter-dir", str(_adapter_dir(seed)),
        "--output-dir", merged_dir,
        "--expected-modules-manifest", str(EXPECTED_MODULES_MANIFEST),
        "--guard-receipt-path", str(guard_receipt_path),
    ]
    print(f"[judge03-merge] running: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600)
    print(r.stdout[-10000:])
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"merge_adapter.py exited {r.returncode}")
    print("[judge03-merge] subprocess exited 0", flush=True)
    return json.loads(guard_receipt_path.read_text())


def _dir_size_gib(path: str) -> float:
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return round(total / (1024 ** 3), 2)


def _run_base_vs_merged_diff(merged_dir: str) -> bool:
    from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, generate, stop_vllm

    def _serve(model_dir: str, container: str, allow_empty: bool) -> str:
        try:
            boot_vllm(model_dir, container, PORT, GPU_MEM_UTIL,
                      serve_script=SERVE_SCRIPT,
                      extra_env={"LANGUAGE_MODEL_ONLY": "1", "MAX_MODEL_LEN": str(MAX_MODEL_LEN)})
            served = wait_healthy(PORT, container, timeout=BOOT_TIMEOUT_SEC)
            out = generate(PORT, served,
                            [{"instruction": DIFF_PROMPT, "source_val_idx": "judge03_merge_diff"}],
                            max_tokens=256)
            text = (out[0] or "").strip()
            if not text and not allow_empty:
                raise RuntimeError(f"real-generation returned empty output for {container}")
            return text
        finally:
            stop_vllm(container)

    print("[judge03-serve] merged judge model ...", flush=True)
    merged_out = _serve(merged_dir, "judge03-merged-diff", allow_empty=True)
    print(f"[judge03-serve] merged output: {merged_out[:200]!r}", flush=True)

    print("[judge03-serve] raw base model (for diff) ...", flush=True)
    base_out = _serve("models/Qwen3.6-35B-A3B", "judge03-base-diff", allow_empty=False)
    print(f"[judge03-serve] base output: {base_out[:200]!r}", flush=True)

    return merged_out != base_out


def _capture_and_score(merged_dir: str, seed: int) -> dict:
    """Serve the merged judge model, capture wp_judge val prompts at
    max_tokens=8192 via the existing (unmodified) sieve_capture_judge_http
    capture(), score with the existing (unmodified) eval_relabel.py, stop the
    serve in a finally block (sole GB10 residency for this step)."""
    from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, generate, stop_vllm
    from scripts.sieve_capture_judge_http import capture as http_capture

    container = "judge03-served-eval"
    cap_path = OUT_DIR / f"judge_capture_vllm_s{seed}.jsonl"
    try:
        boot_vllm(merged_dir, container, PORT, GPU_MEM_UTIL,
                  serve_script=SERVE_SCRIPT,
                  extra_env={"LANGUAGE_MODEL_ONLY": "1", "MAX_MODEL_LEN": str(MAX_MODEL_LEN)})
        served = wait_healthy(PORT, container, timeout=BOOT_TIMEOUT_SEC)
        warm = generate(PORT, served,
                         [{"instruction": "Reply with exactly one word: OK", "source_val_idx": "warmup"}],
                         max_tokens=16)
        if not warm or not warm[0].strip():
            raise RuntimeError(f"real-generation warm-up returned empty output: {warm!r}")
        print(f"[warmup] real-generation OK (served_model={served!r}): {warm[0].strip()[:80]!r}", flush=True)

        cap_stats = http_capture(base_url=f"http://localhost:{PORT}/v1", model=served,
                                  dataset=DATASET, out=str(cap_path),
                                  max_tokens=MAX_TOKENS, temperature=0.0)
        print(f"[judge03-serve] capture stats: {cap_stats}", flush=True)
    finally:
        stop_vllm(container)

    r = subprocess.run([sys.executable, "scripts/relabel/eval_relabel.py", str(cap_path)],
                        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=600)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"eval_relabel failed on vLLM-served capture (exit {r.returncode})")
    summary = json.loads((OUT_DIR / "eval_summary.json").read_text())
    import re
    m = re.search(r"parse_fail=(\d+)", r.stdout)
    parse_fail = int(m.group(1)) if m else summary.get("parse_fail")

    return {
        "seed": seed,
        "rho": summary["rho_new"],
        "ci_lower": summary["ci"][0],
        "ci_upper": summary["ci"][1],
        "n": summary["n"],
        "parse_fail": parse_fail,
        "max_tokens": MAX_TOKENS,
        "served_model_dir": merged_dir,
        "capture_path": str(cap_path),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    capture_rho = json.loads(CAPTURE_RHO_PATH.read_text())
    seed = capture_rho["best_single_seed"]["seed"]
    print(f"[judge03-merge] Task-1 promotion candidate: seed {seed} "
          f"(cheap-path rho={capture_rho['best_single_seed']['rho']})", flush=True)

    sampler_path = _download_promoted_adapter(seed)
    guard = _run_merge(seed)
    merged_dir = _merged_dir(seed)

    merged_target_module_count = guard["merged_target_module_count"]
    expected_target_module_count = guard["expected_target_module_count"]
    merge_ok = (merged_target_module_count == expected_target_module_count
                and merged_target_module_count > 0)

    print("[judge03-merge] serving merged vs base for a real-generation diff ...", flush=True)
    base_vs_merged_differs = _run_base_vs_merged_diff(merged_dir)
    if not (merge_ok and base_vs_merged_differs):
        raise RuntimeError(f"merge guard failed: merge_ok={merge_ok} "
                            f"base_vs_merged_differs={base_vs_merged_differs} -- "
                            f"refusing to serve+score an unverified merge")

    vllm_served = _capture_and_score(merged_dir, seed)

    result = {
        "promoted_sampler_path": sampler_path,
        "merged_dir": merged_dir,
        "merge_ok": merge_ok,
        "merged_target_module_count": merged_target_module_count,
        "expected_target_module_count": expected_target_module_count,
        "base_vs_merged_differs": base_vs_merged_differs,
        "merged_size_gib": _dir_size_gib(merged_dir),
        "vllm_served_single_seed": vllm_served,
        "ensemble_vllm_served": "deferred_to_packaging",
        "max_model_len_served": MAX_MODEL_LEN,
    }
    out_path = OUT_DIR / "judge03_rho.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[judge03-merge] wrote {out_path}", flush=True)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
