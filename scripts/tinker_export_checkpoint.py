#!/usr/bin/env python
"""P4 export — download a Tinker checkpoint archive to a local artifact (.venv-tinker).

Reads the manifest written by tinker_reasoning_sft.py, takes the durable training
checkpoint (`state_path`, a tinker://<run>/weights/<id>), requests a signed archive
URL via get_checkpoint_archive_url_from_tinker_path, downloads the .tar.gz, and
verifies it is a non-trivial, well-formed gzip tar. Export is decoupled from REVL —
this only produces the local weights artifact (downstream RL / MoE-Sieve / packaging).

Usage:
  python scripts/tinker_export_checkpoint.py \
      --manifest output/tinker/wp-reasoning-v2-manifest.json \
      --out-dir models/tinker_export/wp-reasoning-v2
"""
import argparse
import json
import os
import sys
import tarfile
import urllib.request

import tinker


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="output/tinker/wp-reasoning-v2-manifest.json")
    ap.add_argument("--tinker-path", default=None,
                    help="override; default = manifest promoted SAMPLER path "
                         "(the archive endpoint only supports sampler_weights checkpoints)")
    ap.add_argument("--out-dir", default="models/tinker_export/wp-reasoning-v2")
    args = ap.parse_args()

    with open(args.manifest) as f:
        m = json.load(f)
    tinker_path = args.tinker_path
    if not tinker_path:
        # The archive endpoint ONLY supports sampler_weights/ checkpoints (the
        # save_state weights/ checkpoint 400s), so export from the promoted sampler.
        promoted = m.get("promoted")
        tinker_path = next((c["sampler_path"] for c in m.get("checkpoints", [])
                            if c.get("name") == promoted), None)
    if not tinker_path:
        raise SystemExit(f"no sampler path for promoted checkpoint in {args.manifest}")
    print(f"[export] checkpoint: {tinker_path}", flush=True)

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    print("[export] requesting archive URL (server packs the archive — may take a while)...", flush=True)
    resp = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path).result()
    url = getattr(resp, "url", None) or getattr(resp, "archive_url", None)
    expires = getattr(resp, "expires_at", None)
    if not url:
        raise SystemExit(f"no URL in archive response: {resp!r}")
    print(f"[export] signed URL acquired (expires={expires})", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_tar = os.path.join(args.out_dir, "checkpoint.tar")
    print(f"[export] downloading -> {out_tar}", flush=True)
    urllib.request.urlretrieve(url, out_tar)
    size = os.path.getsize(out_tar)
    print(f"[export] downloaded {size / 1e6:.1f} MB", flush=True)

    # Verify: well-formed tar (Tinker ships an UNcompressed POSIX tar despite any
    # .gz convention), non-trivial, lists members. "r:*" auto-detects compression.
    if size < 1024:
        raise SystemExit(f"[export] FAIL: archive suspiciously small ({size} bytes)")
    try:
        with tarfile.open(out_tar, "r:*") as tf:
            members = tf.getnames()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"[export] FAIL: archive not a valid tar: {e}")
    print(f"[export] archive OK: {len(members)} members; sample={members[:8]}", flush=True)

    manifest_export = {
        "tinker_path": tinker_path, "archive": out_tar, "size_bytes": size,
        "n_members": len(members), "members_sample": members[:20],
        "expires_at_url": str(expires),
    }
    meta_path = os.path.join(args.out_dir, "export_manifest.json")
    with open(meta_path, "w") as f:
        json.dump(manifest_export, f, indent=2)
    print(f"[export] DONE -> {out_tar} (+ {meta_path})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
