#!/usr/bin/env python
"""Phase 21 Task 2 -- MoE (train_mlp=True) merge probe: prove the fused-expert
merge path on the new base BEFORE any real GEN-02/JUDGE-02 Tinker spend.

Mirrors scripts/build_base20_probe_adapter.py's PRIMARY (source=tinker) path
exactly, with ONE change: train_mlp=True/train_attn=False (Phase 20's probe was
attention-only by design, MoE experts explicitly deferred as "already-solved" --
21-RESEARCH.md Pitfall 1 shows that assumption needs re-verifying for this
base's FUSED mlp.experts.{gate_up_proj,down_proj} nn.Parameter tensors).

Writes output/base21/moe_merge_probe.json TWICE:
  1. After the Tinker probe run + archive download/extract -- attached_expert_modules,
     fused_convention_observed, source, sampler_path (this file doubles as the
     --expected-modules-manifest merge_adapter.py reads for its module-count guard).
  2. After merge_adapter.py + the smoke_vl_merge_base20.py re-run -- merge_ok,
     merged_target_module_count, expected_target_module_count,
     base_vs_merged_differs, smoke_vl_merge_rerun_ok, cost_note.

Must run under the Tinker venv:
    .venv-tinker/bin/python scripts/build_base21_moe_probe_adapter.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

from tinker_reasoning_data_v4 import BASE_MODEL, RENDERER_NAME  # noqa: E402

OUT_DIR = PROJECT_ROOT / "output" / "base21"
ADAPTER_DIR = OUT_DIR / "base21_moe_probe_adapter"
OUT_PATH = OUT_DIR / "moe_merge_probe.json"
GUARD_RECEIPT_PATH = OUT_DIR / "_merge_guard_result.json"
CONFIG_PATH = "config/train_config_v4.yaml"
MERGED_MODEL_DIR = "models/Qwen3.6-35B-A3B-base21-moe-probe-merged"

# Trivial, overfit-friendly probe example -- same discipline as the Phase 20
# attention probe: distinctive target completion so a tiny/cheap LoRA nudge is
# easy to observe in the base-vs-merged diff.
PROBE_PROMPT = "Reply with exactly one word: OK"
PROBE_COMPLETION = "PROBEXYZ"
PROBE_RANK = 8
PROBE_LR = 0.05
PROBE_STEPS = 8


def _attached_modules_from_adapter(adapter_dir: Path) -> list[str]:
    """Read the ACTUAL lora_A/lora_B (or target_parameters) key names from an
    exported PEFT adapter and return the sorted unique module base-paths
    (don't assume the fused-expert convention from config, log the real list)."""
    from safetensors import safe_open

    modules = set()
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            for marker in (".lora_A.", ".lora_B."):
                if marker in k:
                    modules.add(k.rsplit(marker, 1)[0])
                    break
            else:
                # target_parameters-style keys have no lora_A/lora_B marker --
                # keep the raw key so the fused-expert convention is still visible.
                modules.add(k)
    return sorted(modules)


def _fused_convention_observed(attached_modules: list[str]) -> str:
    """Describe the ACTUAL exported key convention for the MoE expert tensors,
    per Open Question 4: same 3-tensor per-expert convention as merge_tinker_v3.py,
    a different fused convention, or something else entirely."""
    expert_keys = [m for m in attached_modules if "expert" in m.lower()]
    if not expert_keys:
        return "NO expert modules attached -- unexpected for train_mlp=True (see notes)"
    fused = [m for m in expert_keys if "gate_up_proj" in m or "down_proj" in m]
    per_expert_3tensor = [m for m in expert_keys if any(
        s in m for s in ("gate_proj", "up_proj")) and "gate_up_proj" not in m]
    if fused and not per_expert_3tensor:
        return (
            f"FUSED convention observed ({len(fused)} gate_up_proj/down_proj-style keys) -- "
            f"matches config/train_config_v4.yaml's target_parameters mechanism, NOT "
            f"merge_tinker_v3.py's unfused 3-tensor per-expert convention. Sample: {fused[:3]}"
        )
    if per_expert_3tensor and not fused:
        return (
            f"UNFUSED 3-tensor per-expert convention observed ({len(per_expert_3tensor)} "
            f"gate_proj/up_proj/down_proj-style keys) -- matches merge_tinker_v3.py's OLD-base "
            f"convention, NOT the new base's fused target_parameters mechanism. "
            f"Sample: {per_expert_3tensor[:3]}"
        )
    return f"MIXED/unclear convention -- sample expert keys: {expert_keys[:5]}"


def run_tinker_probe() -> dict:
    """PRIMARY path: minimal real Tinker LoRA run, MoE-experts-only target."""
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    train_row = {
        "messages": [
            {"role": "user", "content": PROBE_PROMPT},
            {"role": "assistant", "content": PROBE_COMPLETION},
        ]
    }
    train_path = OUT_DIR / "_moe_probe_train.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w") as f:
        f.write(json.dumps(train_row) + "\n")

    tok = get_tokenizer(BASE_MODEL)
    cc = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=BASE_MODEL,
        renderer_name=RENDERER_NAME,
        max_length=512,
        batch_size=1,
        train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    train_ds, _ = FromConversationFileBuilder(file_path=str(train_path), test_size=0, common_config=cc)()

    print(f"[probe] tinker run: base={BASE_MODEL} renderer={RENDERER_NAME} rank={PROBE_RANK} "
          f"lr={PROBE_LR} steps={PROBE_STEPS} train_mlp=True train_attn=False train_unembed=False",
          flush=True)

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=PROBE_RANK,
        train_mlp=True,
        train_attn=False,
        train_unembed=False,
    )

    losses = []
    for step in range(PROBE_STEPS):
        batch = train_ds.get_batch(0)
        fb = tc.forward_backward(data=batch, loss_fn="cross_entropy")
        tc.optim_step(tinker.AdamParams(learning_rate=PROBE_LR))
        out = fb.result() if hasattr(fb, "result") else fb
        loss = None
        try:
            d = out.model_dump() if hasattr(out, "model_dump") else getattr(out, "__dict__", {})
            for v in d.values() if isinstance(d, dict) else []:
                if isinstance(v, (int, float)):
                    loss = float(v)
                    break
        except Exception:  # noqa: BLE001 -- loss logging is diagnostic only
            pass
        losses.append(loss)
        print(f"[probe] step {step} loss={loss}", flush=True)

    sampler_res = tc.save_weights_for_sampler(name="base21-moe-probe", ttl_seconds=None)
    sampler_path = sampler_res.result().path if hasattr(sampler_res, "result") else sampler_res.path
    print(f"[probe] sampler checkpoint: {sampler_path}", flush=True)

    rc = sc.create_rest_client()
    print("[probe] requesting archive URL (server packs the archive)...", flush=True)
    resp = rc.get_checkpoint_archive_url_from_tinker_path(sampler_path).result()
    url = getattr(resp, "url", None) or getattr(resp, "archive_url", None)
    if not url:
        raise RuntimeError(f"no archive URL in response: {resp!r}")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = ADAPTER_DIR / "checkpoint.tar"
    print(f"[probe] downloading archive -> {tar_path}", flush=True)
    urllib.request.urlretrieve(url, tar_path)
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            # WR-10 (reused from Phase 20): require a regular file -- m.name
            # already neutralizes path traversal, but a symlink/hardlink/device
            # entry named adapter_config.json would otherwise still be
            # extracted as such.
            if name in ("adapter_config.json", "adapter_model.safetensors") and m.isfile():
                m.name = name
                tf.extract(m, ADAPTER_DIR)

    for required in ("adapter_config.json", "adapter_model.safetensors"):
        if not (ADAPTER_DIR / required).exists():
            raise RuntimeError(f"tinker archive missing {required} after extraction")

    attached_modules = _attached_modules_from_adapter(ADAPTER_DIR)
    if not attached_modules:
        raise RuntimeError("tinker adapter exported zero LoRA modules")

    return {
        "source": "tinker",
        "adapter_dir": str(ADAPTER_DIR),
        "attached_modules": attached_modules,
        "attached_expert_modules": [m for m in attached_modules if "expert" in m.lower()],
        "fused_convention_observed": _fused_convention_observed(attached_modules),
        "sampler_path": sampler_path,
        "rank": PROBE_RANK,
        "learning_rate": PROBE_LR,
        "steps": PROBE_STEPS,
        "losses": losses,
        "cost_note": (
            "Minimal real Tinker LoRA run (rank=8, train_mlp-only, 8 steps, 1 "
            "overfit-friendly example) -- order-of-magnitude cents, matches the "
            "20-04 attention-only probe's cost profile."
        ),
    }


def run_local_fallback_probe(reason: str) -> dict:
    """FALLBACK path: only if TINKER_API_KEY is absent or the run is blocked
    for an account/API reason. Mirrors 20-04's local_zero_init fallback, but a
    real zero-init MoE-expert LoRA against target_parameters requires PEFT
    support this probe cannot exercise offline in the same way as the
    tinker-primary path -- record source + reduced confidence rather than
    fabricating a pass, per plan instruction."""
    print(f"[probe] FALLBACK: local fallback probe ({reason})", flush=True)
    return {
        "source": "local_fallback",
        "attached_modules": [],
        "attached_expert_modules": [],
        "fused_convention_observed": "N/A -- local fallback cannot exercise Tinker's export-side convention",
        "sampler_path": None,
        "reason": reason,
        "cost_note": "no Tinker spend incurred (fallback path)",
    }


def run_merge(adapter_dir: str) -> None:
    """Run merge_adapter.py as a subprocess (blocking until it fully exits --
    releases the CPU-resident copy before any further step)."""
    cmd = [
        sys.executable, "scripts/merge_adapter.py",
        "--config-path", CONFIG_PATH,
        "--adapter-dir", adapter_dir,
        "--output-dir", MERGED_MODEL_DIR,
        "--expected-modules-manifest", str(OUT_PATH),
        "--guard-receipt-path", str(GUARD_RECEIPT_PATH),
    ]
    print(f"[merge] running: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600)
    print(r.stdout[-10000:])
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise RuntimeError(f"merge_adapter.py exited {r.returncode} (see stdout/stderr above)")
    print("[merge] subprocess exited 0", flush=True)


def run_base_vs_merged_diff() -> bool:
    """Serve merged vs base on the same fixed prompt (greedy) and diff, reusing
    the exact boot_vllm/wait_healthy/generate/stop_vllm harness Phase 20 already
    proved for this base -- no re-implementation needed."""
    from scripts._p0_vllm_smoke_serve import boot_vllm, wait_healthy, generate, stop_vllm

    serve_script = str(PROJECT_ROOT / "scripts" / "serve_base20_vllm.sh")
    port = 8022

    def _serve(model_dir: str, container: str, allow_empty: bool) -> str:
        try:
            boot_vllm(model_dir, container, port, 0.80,
                      serve_script=serve_script, extra_env={"LANGUAGE_MODEL_ONLY": "1"})
            served = wait_healthy(port, container, timeout=1200)
            out = generate(port, served,
                            [{"instruction": PROBE_PROMPT, "source_val_idx": "moe_merge_probe"}],
                            max_tokens=64)
            text = (out[0] or "").strip()
            if not text and not allow_empty:
                raise RuntimeError(f"real-generation returned empty output for {container}")
            return text
        finally:
            stop_vllm(container)

    print("[serve] merged model ...", flush=True)
    merged_out = _serve(MERGED_MODEL_DIR, "base21-moe-probe-merged", allow_empty=True)
    print(f"[serve] merged output: {merged_out!r}", flush=True)

    print("[serve] base model (for diff) ...", flush=True)
    base_out = _serve("models/Qwen3.6-35B-A3B", "base21-moe-probe-base", allow_empty=False)
    print(f"[serve] base output: {base_out!r}", flush=True)

    return merged_out != base_out


def run_smoke_vl_merge_rerun() -> bool:
    """Re-run scripts/smoke_vl_merge_base20.py once to re-exercise merge_adapter.py's
    post-review-fix code (WR-02/WR-03/WR-04, 20-VERIFICATION carry-forward 2)
    end-to-end on the current repo (discharges that carry-forward)."""
    cmd = [sys.executable, "scripts/smoke_vl_merge_base20.py"]
    print(f"[smoke-rerun] running: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600)
    print(r.stdout[-6000:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
        return False
    return "BASE-04 SMOKE PASSED" in r.stdout


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tinker_key_present = bool(os.environ.get("TINKER_API_KEY"))

    if tinker_key_present:
        try:
            result = run_tinker_probe()
        except Exception as exc:  # noqa: BLE001 -- fall back only for account/API-shaped failures
            print(f"[probe] tinker run raised: {exc!r}", flush=True)
            result = run_local_fallback_probe(reason=f"tinker run blocked: {exc}")
    else:
        result = run_local_fallback_probe(reason="TINKER_API_KEY not set in environment")

    # First write: this file doubles as merge_adapter.py's --expected-modules-manifest.
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[probe] wrote {OUT_PATH} (interim -- attached_modules only)", flush=True)

    if result["source"] != "tinker" or not result["attached_modules"]:
        result["merge_ok"] = False
        result["notes"] = "no mergeable adapter produced (fallback path) -- merge step skipped"
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print("MOE MERGE PROBE: fallback path, merge NOT attempted")
        return 1

    run_merge(result["adapter_dir"])

    guard = json.loads(GUARD_RECEIPT_PATH.read_text())
    merged_target_module_count = guard["merged_target_module_count"]
    expected_target_module_count = guard["expected_target_module_count"]

    print("[diff] serving merged vs base for a real-generation diff ...", flush=True)
    base_vs_merged_differs = run_base_vs_merged_diff()

    print("[carry-forward] re-running smoke_vl_merge_base20.py ...", flush=True)
    smoke_vl_merge_rerun_ok = run_smoke_vl_merge_rerun()

    merge_ok = (merged_target_module_count == expected_target_module_count
                and merged_target_module_count > 0)

    result.update({
        "merge_ok": merge_ok,
        "merged_target_module_count": merged_target_module_count,
        "expected_target_module_count": expected_target_module_count,
        "base_vs_merged_differs": base_vs_merged_differs,
        "smoke_vl_merge_rerun_ok": smoke_vl_merge_rerun_ok,
    })
    del result["losses"]  # noisy, not needed in the final receipt

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[probe] wrote final {OUT_PATH}", flush=True)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
