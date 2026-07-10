#!/usr/bin/env python
"""Phase 04.3-04 Task 1 launcher — train the 9 MoE-only rank x replay candidates on
Tinker, export each promoted sampler to a local checkpoint.tar, and assemble the
combined output/tinker/grid_manifest.json.

Drives scripts/tinker_reasoning_sft.py (MoE-only by default after Plan 01) and
scripts/tinker_export_checkpoint.py (Plan-prior export tool). RESUMABLE: a candidate
whose per-candidate manifest has 3 epochs promoted AND whose checkpoint.tar exists is
skipped. Cheapest-first (rank 8) so a wiring bug surfaces on the ~$1 first cell.

HARD GUARD (T-04.3-08): every per-candidate manifest MUST record train_attn==False and
train_unembed==False. A True (accidental all-linear) aborts the whole launcher — an
all-linear candidate would re-introduce the net-harmful attention deltas RC-B identified.

Run detached via scripts/_run_grid_train.sh (sources .env for TINKER_API_KEY).
"""
import json
import os
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(ROOT, ".venv-tinker", "bin", "python")

REPLAY_PATHS = {
    15: "data/reasoning_dataset/openai_train.augmented.jsonl",          # base = 15% (Plan 02)
    30: "data/reasoning_dataset/openai_train.augmented.replay30.jsonl",
    50: "data/reasoning_dataset/openai_train.augmented.replay50.jsonl",
}
RANKS = [8, 16, 32]
REPLAYS = [15, 30, 50]

GRID_MANIFEST = "output/tinker/grid_manifest.json"


def log(msg):
    print(f"[grid-train] {msg}", flush=True)


def candidate_configs():
    # rank-minor, replay-minor: cheapest (rank 8) first so cell 1 de-risks the chain.
    for r in RANKS:
        for p in REPLAYS:
            yield {
                "candidate_tag": f"r{r}-rp{p}",
                "rank": r,
                "replay_pct": p,
                "train_path": REPLAY_PATHS[p],
                "manifest": f"output/tinker/wp-reasoning-v4-r{r}-rp{p}-manifest.json",
                "out_dir": f"models/tinker_export/wp-reasoning-v4-r{r}-rp{p}",
            }


def trained_ok(manifest_path, epochs=3):
    """True if the per-candidate manifest shows all epochs promoted + MoE-only."""
    if not os.path.exists(manifest_path):
        return False
    m = json.load(open(manifest_path))
    if m.get("train_attn") is not False or m.get("train_unembed") is not False:
        raise SystemExit(
            f"FATAL T-04.3-08: {manifest_path} is NOT MoE-only "
            f"(train_attn={m.get('train_attn')} train_unembed={m.get('train_unembed')}); "
            f"an all-linear candidate re-introduces net-harmful attn deltas. Aborting grid."
        )
    cps = m.get("checkpoints", [])
    return m.get("promoted") and len(cps) >= epochs and all(c.get("sampler_path") for c in cps)


def tar_ok(tar_path):
    if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 1_000_000:
        return False
    try:
        with tarfile.open(tar_path) as t:
            names = t.getnames()
        return any("adapter_model" in n for n in names)
    except Exception:
        return False


def run(cmd):
    log("RUN " + " ".join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        raise SystemExit(f"FATAL: command failed rc={rc}: {' '.join(cmd)}")


def promoted_sampler_path(manifest_path):
    m = json.load(open(manifest_path))
    promoted = m.get("promoted")
    for c in m.get("checkpoints", []):
        if c.get("name") == promoted:
            return c.get("sampler_path")
    raise SystemExit(f"no promoted sampler_path in {manifest_path}")


def main():
    os.chdir(ROOT)
    grid = []
    configs = list(candidate_configs())
    for idx, cfg in enumerate(configs):
        tag = cfg["candidate_tag"]
        manifest_path = cfg["manifest"]
        tar_path = os.path.join(cfg["out_dir"], "checkpoint.tar")
        log(f"=== candidate {idx + 1}/9: {tag} (rank={cfg['rank']} replay={cfg['replay_pct']}%) ===")

        # --- Train (resumable) ---
        if trained_ok(manifest_path):
            log(f"{tag}: training already complete (manifest has 3 promoted epochs) — skip")
        else:
            run([
                VENV_PY, "scripts/tinker_reasoning_sft.py",
                "--stage", f"v4-{tag}",
                "--rank", str(cfg["rank"]),
                "--train-path", cfg["train_path"],
                "--epochs", "3",
                "--per-epoch-eval-n", "8",
                "--manifest", manifest_path,
            ])
            if not trained_ok(manifest_path):
                raise SystemExit(f"FATAL: {tag} training did not produce a complete MoE-only manifest")

        # --- Export promoted sampler -> checkpoint.tar (resumable) ---
        if tar_ok(tar_path):
            log(f"{tag}: export tar already present + well-formed — skip")
        else:
            os.makedirs(cfg["out_dir"], exist_ok=True)
            run([
                VENV_PY, "scripts/tinker_export_checkpoint.py",
                "--manifest", manifest_path,
                "--out-dir", cfg["out_dir"],
            ])
            if not tar_ok(tar_path):
                raise SystemExit(f"FATAL: {tag} export did not produce a valid checkpoint.tar at {tar_path}")

        grid.append({
            "candidate_tag": tag,
            "rank": cfg["rank"],
            "replay_pct": cfg["replay_pct"],
            "train_path": cfg["train_path"],
            "manifest": manifest_path,
            "adapter_tar": tar_path,
            "sampler_path": promoted_sampler_path(manifest_path),
        })
        # Write grid manifest incrementally (crash-safe).
        os.makedirs("output/tinker", exist_ok=True)
        json.dump(grid, open(GRID_MANIFEST, "w"), indent=2)
        log(f"{tag}: DONE — grid_manifest now has {len(grid)} candidate(s)")

        if idx == 0:
            log("CHECKPOINT: candidate 1 (r8-rp15) trained+exported+MoE-only-verified. "
                "Live Tinker->export chain validated. Continuing remaining 8.")

    assert len(grid) == 9, f"expected 9 candidates, assembled {len(grid)}"
    log(f"ALL 9 candidates trained + exported. grid_manifest.json written ({len(grid)} entries).")
    log("DONE")


if __name__ == "__main__":
    main()
