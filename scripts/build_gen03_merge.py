#!/usr/bin/env python
"""Phase 21 Plan 05 Task 1 -- GEN-03 merge: download the PROMOTED gen sampler
checkpoint from output/tinker/wp-gen-v4-manifest.json and merge it onto the
local new base via the fused-expert path proven in 21-01
(merge_adapter.py --config-path config/train_config_v4.yaml routes
mlp.experts.* adapters through tinker_cookbook.weights.build_hf_model).

Mirrors scripts/build_base21_moe_probe_adapter.py's archive-download +
merge + base-vs-merged-diff pattern exactly, with ONE difference: this
downloads a REAL promoted training checkpoint (GEN-02's wp-gen-v4-ep3),
not a fresh throwaway probe run -- no Tinker training spend here, just a
REST archive fetch of an already-trained sampler checkpoint.

Writes output/base21/gen03_merge.json: promoted_sampler_path, adapter_dir,
merged_dir, merge_ok, merged_target_module_count, expected_target_module_count,
base_vs_merged_differs, merged_size_gib.

Must run under the Tinker venv (tinker + tinker_cookbook + torch/transformers/
peft all present there; vLLM itself runs inside the docker serve stack, not
this venv):
    .venv-tinker/bin/python scripts/build_gen03_merge.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "output" / "base21"
MANIFEST_PATH = PROJECT_ROOT / "output" / "tinker" / "wp-gen-v4-manifest.json"
ADAPTER_DIR = OUT_DIR / "gen03_gen_adapter"
OUT_PATH = OUT_DIR / "gen03_merge.json"
GUARD_RECEIPT_PATH = OUT_DIR / "_gen03_merge_guard_result.json"
# Same architecture (train_mlp=True/train_attn=False/train_unembed=False, 40
# layers) as the 21-01 MoE probe -- the attached-module SET is architecture-
# driven, not weight-value-driven, so the probe's receipt is a valid
# --expected-modules-manifest for this real adapter too (240 = 120 routed-
# expert + 120 shared_expert).
EXPECTED_MODULES_MANIFEST = OUT_DIR / "moe_merge_probe.json"
CONFIG_PATH = "config/train_config_v4.yaml"
MERGED_MODEL_DIR = "models/Qwen3.6-35B-A3B-gen-v4-merged"
DIFF_PROMPT = "Write a WordPress function that returns the current logged-in user's display name."


def _download_promoted_adapter() -> str:
    """Download the promoted gen sampler checkpoint's archive and extract the
    adapter files, mirroring build_base21_moe_probe_adapter.py's WR-10
    member-validated extraction (regular-file check neutralizes symlink/
    hardlink/device-entry path-traversal tricks in the tar)."""
    import tinker

    manifest = json.loads(MANIFEST_PATH.read_text())
    promoted_name = manifest["promoted"]
    sampler_path = next(
        c["sampler_path"] for c in manifest["checkpoints"] if c["name"] == promoted_name
    )
    print(f"[gen03] promoted checkpoint: {promoted_name} -> {sampler_path}", flush=True)

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    print("[gen03] requesting archive URL (server packs the archive)...", flush=True)
    resp = rc.get_checkpoint_archive_url_from_tinker_path(sampler_path).result()
    url = getattr(resp, "url", None) or getattr(resp, "archive_url", None)
    if not url:
        raise RuntimeError(f"no archive URL in response: {resp!r}")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = ADAPTER_DIR / "checkpoint.tar"
    print(f"[gen03] downloading archive -> {tar_path}", flush=True)
    # Resumable + retrying download: plain urllib.request.urlretrieve has no
    # retry/resume and died mid-transfer with ContentTooShortError on this
    # 2 GiB archive. curl -C - resumes any existing partial file; --retry
    # covers transient network resets; --fail surfaces HTTP errors as a
    # non-zero exit (curl exit 18 = server closed before Content-Length was
    # reached, so a silently-truncated file cannot pass this check).
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
            # WR-10: require a regular file -- m.name already neutralizes path
            # traversal, but a symlink/hardlink/device entry named
            # adapter_config.json would otherwise still be extracted as such.
            if name in ("adapter_config.json", "adapter_model.safetensors") and m.isfile():
                m.name = name
                tf.extract(m, ADAPTER_DIR)

    for required in ("adapter_config.json", "adapter_model.safetensors"):
        if not (ADAPTER_DIR / required).exists():
            raise RuntimeError(f"tinker archive missing {required} after extraction")

    return sampler_path


def _run_merge() -> None:
    cmd = [
        sys.executable, "scripts/merge_adapter.py",
        "--config-path", CONFIG_PATH,
        "--adapter-dir", str(ADAPTER_DIR),
        "--output-dir", MERGED_MODEL_DIR,
        "--expected-modules-manifest", str(EXPECTED_MODULES_MANIFEST),
        "--guard-receipt-path", str(GUARD_RECEIPT_PATH),
    ]
    print(f"[merge] running: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600)
    print(r.stdout[-10000:])
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise RuntimeError(f"merge_adapter.py exited {r.returncode} (see stdout/stderr above)")
    print("[merge] subprocess exited 0", flush=True)


def _dir_size_gib(path: str) -> float:
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return round(total / (1024 ** 3), 2)


def _run_base_vs_merged_diff() -> bool:
    """Serve merged vs raw base on the same fixed prompt (greedy) and diff --
    reuses the exact boot_vllm/wait_healthy/generate/stop_vllm harness proven
    in 21-01 (sole-GB10-residency discipline: each serve fully tears down
    before the next boots)."""
    from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, generate, stop_vllm

    serve_script = str(PROJECT_ROOT / "scripts" / "serve_base20_vllm.sh")
    port = 8023

    def _serve(model_dir: str, container: str, allow_empty: bool) -> str:
        try:
            boot_vllm(model_dir, container, port, 0.80,
                      serve_script=serve_script, extra_env={"LANGUAGE_MODEL_ONLY": "1"})
            served = wait_healthy(port, container, timeout=1200)
            out = generate(port, served,
                            [{"instruction": DIFF_PROMPT, "source_val_idx": "gen03_merge_diff"}],
                            max_tokens=128)
            text = (out[0] or "").strip()
            if not text and not allow_empty:
                raise RuntimeError(f"real-generation returned empty output for {container}")
            return text
        finally:
            stop_vllm(container)

    print("[serve] merged gen model ...", flush=True)
    merged_out = _serve(MERGED_MODEL_DIR, "gen03-merged-diff", allow_empty=True)
    print(f"[serve] merged output: {merged_out[:200]!r}", flush=True)

    print("[serve] raw base model (for diff) ...", flush=True)
    base_out = _serve("models/Qwen3.6-35B-A3B", "gen03-base-diff", allow_empty=False)
    print(f"[serve] base output: {base_out[:200]!r}", flush=True)

    return merged_out != base_out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sampler_path = _download_promoted_adapter()
    _run_merge()

    guard = json.loads(GUARD_RECEIPT_PATH.read_text())
    merged_target_module_count = guard["merged_target_module_count"]
    expected_target_module_count = guard["expected_target_module_count"]

    print("[diff] serving merged vs base for a real-generation diff ...", flush=True)
    base_vs_merged_differs = _run_base_vs_merged_diff()

    merge_ok = (merged_target_module_count == expected_target_module_count
                and merged_target_module_count > 0)

    result = {
        "promoted_sampler_path": sampler_path,
        "adapter_dir": str(ADAPTER_DIR),
        "merged_dir": MERGED_MODEL_DIR,
        "merge_ok": merge_ok,
        "merged_target_module_count": merged_target_module_count,
        "expected_target_module_count": expected_target_module_count,
        "base_vs_merged_differs": base_vs_merged_differs,
        "merged_size_gib": _dir_size_gib(MERGED_MODEL_DIR),
        "config_path": CONFIG_PATH,
        "merge_engine": "tinker_cookbook.weights.build_hf_model (routed MoE-expert path, 21-01 gap-closure)",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[gen03] wrote {OUT_PATH}", flush=True)
    print(json.dumps(result, indent=2))
    return 0 if merge_ok and base_vs_merged_differs else 1


if __name__ == "__main__":
    sys.exit(main())
