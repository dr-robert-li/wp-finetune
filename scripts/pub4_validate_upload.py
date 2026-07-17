"""Standalone post-upload round-trip validation driver (PUB4-01, Wave 0).

Proves the UPLOADED bytes work, not just the local pre-upload artifact: lists
the published repo via the HF API, downloads the ship-tier GGUF from HF (not a
local copy), loads it via llama-server, re-asserts the 224-expert topology on
the DOWNLOADED bytes (T-27-01's last integrity gate), and runs one judge smoke
prompt. Emits the Phase-18 PUB-03 receipt schema minus the retired generation-smoke
block (judge-only ship per CONTEXT.md's scope correction).

Usage:
    .venv-tinker/bin/python -m scripts.pub4_validate_upload --repo iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf --manifest output/pkg-v4/pub4_upload_manifest.json
    .venv-tinker/bin/python -m scripts.pub4_validate_upload --self-check   # no network, no HF creds, no GGUF, no GPU
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Documented workaround for this host (_pub03_upload.sh precedent): upload-large-folder
# and some download paths deadlock/misbehave with Xet enabled.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.output_parsers import parse_judge_scores, load_dim_map  # noqa: E402

LLAMA_SERVER = str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-server")
SMOKE_PROMPTS_PATH = "data/phase4_4/smoke_prompts.json"
EXPECTED_EXPERT_COUNT = 224  # the pruned v4 judge's post-surgery expert count (T-27-01 last gate)


# --------------------------------------------------------------------------
# Pure logic (no I/O) -- shared by the real path and --self-check.
# --------------------------------------------------------------------------

def compare_listing(listing: dict, manifest: dict) -> dict:
    """listing:  {repo_id: {"public": bool, "files": {remote_name: size_bytes}}}
    manifest: {"repos": [{"repo_id": str, "files": [{"repo_path": str, "size_bytes": int}]}]}

    Pure -- no file I/O, no network. The real path resolves local sizes into this
    `manifest` shape BEFORE calling this function; --self-check fabricates both
    dicts directly, so both callers exercise the exact same comparison logic."""
    result = {"ok": True, "repos": {}}
    for repo in manifest.get("repos", []):
        repo_id = repo["repo_id"]
        repo_listing = listing.get(repo_id, {})
        remote_files = repo_listing.get("files", {})
        files_ok = {}
        repo_ok = True
        for f in repo["files"]:
            name = f["repo_path"]
            expected = f["size_bytes"]
            actual = remote_files.get(name)
            match = actual is not None and actual == expected
            files_ok[name] = match
            repo_ok = repo_ok and match
        result["repos"][repo_id] = {
            "public": repo_listing.get("public"),
            "files": files_ok,
            "matches_manifest": repo_ok,
        }
        result["ok"] = result["ok"] and repo_ok
    return result


def _self_check() -> None:
    manifest = {
        "repos": [
            {
                "repo_id": "iamchum/fake-v4-repo",
                "files": [
                    {"repo_path": "README.md", "size_bytes": 4096},
                    {"repo_path": "wp-judge-v4-pruned-k224.Q8_0.gguf", "size_bytes": 36_000_000_000},
                ],
            }
        ]
    }

    listing_match = {
        "iamchum/fake-v4-repo": {
            "public": True,
            "files": {"README.md": 4096, "wp-judge-v4-pruned-k224.Q8_0.gguf": 36_000_000_000},
        }
    }
    r1 = compare_listing(listing_match, manifest)
    assert r1["ok"] is True, f"self-check FAILED: matching listing did not resolve ok=True: {r1}"
    assert r1["repos"]["iamchum/fake-v4-repo"]["matches_manifest"] is True

    # One file off by a single byte -- a size comparison that cannot go red proves nothing.
    listing_mismatch = {
        "iamchum/fake-v4-repo": {
            "public": True,
            "files": {"README.md": 4096, "wp-judge-v4-pruned-k224.Q8_0.gguf": 36_000_000_001},
        }
    }
    r2 = compare_listing(listing_mismatch, manifest)
    assert r2["ok"] is False, f"self-check FAILED: 1-byte mismatch did not resolve ok=False: {r2}"
    assert r2["repos"]["iamchum/fake-v4-repo"]["matches_manifest"] is False

    print("self-check OK")


# --------------------------------------------------------------------------
# Real path (network / subprocess / file I/O)
# --------------------------------------------------------------------------

def _manifest_with_local_sizes(raw_manifest: dict) -> dict:
    """The ONLY place this driver reads local file sizes -- resolves the raw
    upload manifest (path + repo_path) into the size-bearing shape compare_listing
    expects, before handing off to the pure function."""
    out_repos = []
    for repo in raw_manifest["repos"]:
        files = [
            {"repo_path": f["repo_path"], "size_bytes": os.path.getsize(f["path"])}
            for f in repo["files"]
        ]
        out_repos.append({"repo_id": repo["repo_id"], "files": files})
    return {"repos": out_repos}


def fetch_api_listing(repo_id: str) -> dict:
    from huggingface_hub import HfApi

    api = HfApi()  # picks up the `hf` CLI credential store on its own -- never read a token into a variable
    info = api.repo_info(repo_id, files_metadata=True)
    files: dict = {}
    for s in (info.siblings or []):
        size = getattr(s, "size", None)
        lfs = getattr(s, "lfs", None)
        if size is None and lfs is not None:
            size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
        files[s.rfilename] = size
    return {repo_id: {"public": not bool(info.private), "files": files}}


def download_ship_gguf(repo_id: str, raw_manifest: dict, scratch_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    gguf_entry = None
    for repo in raw_manifest["repos"]:
        if repo["repo_id"] != repo_id:
            continue
        for f in repo["files"]:
            if f["repo_path"].endswith(".gguf"):
                gguf_entry = f
                break
    if gguf_entry is None:
        raise RuntimeError(f"No .gguf entry in manifest for repo {repo_id}")
    os.makedirs(scratch_dir, exist_ok=True)
    return hf_hub_download(repo_id=repo_id, filename=gguf_entry["repo_path"], local_dir=scratch_dir)


def serve_and_probe(gguf_path: str, port: int, alias: str) -> subprocess.Popen:
    """Boot llama-server and poll readiness with a REAL 1-token chat completion --
    NOT the health-check endpoint, which returns ok while a 30B+ is still loading
    (documented: scripts/_pkg_gguf_eval_run.sh:21-32)."""
    import requests

    proc = subprocess.Popen(
        [
            LLAMA_SERVER, "-m", gguf_path,
            "--host", "127.0.0.1", "--port", str(port),
            "-ngl", "999", "-c", "12288", "--jinja", "-a", alias,
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    ready = False
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server DIED (rc={proc.returncode})")
        try:
            r = requests.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": alias, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                timeout=30,
            )
            if r.ok and "content" in r.text:
                ready = True
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    if not ready:
        proc.terminate()
        raise RuntimeError("llama-server never warmed (real-generation readiness probe failed)")
    return proc


def _read_field_int(reader, suffix: str):
    for f in reader.fields:
        if f.endswith(suffix):
            fld = reader.fields[f]
            return int(fld.parts[fld.data[0]][0])
    return None


def _read_field_str(reader, suffix: str):
    for f in reader.fields:
        if f.endswith(suffix):
            fld = reader.fields[f]
            arr = fld.parts[fld.data[0]]
            return bytes(bytearray(int(x) for x in arr)).decode("utf-8", errors="replace")
    return None


def read_gguf_header(gguf_path: str) -> dict:
    from gguf import GGUFReader

    r = GGUFReader(gguf_path)
    version_fld = r.fields["GGUF.version"]
    version = int(version_fld.parts[version_fld.data[0]][0])
    return {
        "magic": "GGUF",
        "version": version,
        "n_tensors": len(r.tensors),
        "arch": _read_field_str(r, "general.architecture"),
        "expert_count": _read_field_int(r, ".expert_count"),
        "block_count": _read_field_int(r, ".block_count"),
        "file_type": _read_field_int(r, "general.file_type"),
        "size_bytes": Path(gguf_path).stat().st_size,
    }


def run_judge_smoke(port: int, alias: str, out_dir: str) -> dict:
    import requests

    prompts = json.load(open(SMOKE_PROMPTS_PATH))
    prompt = prompts[0]
    resp = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={
            "model": alias,
            "messages": [{"role": "user", "content": prompt["instruction"]}],
            "max_tokens": 2048,
            "temperature": 0.0,
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    response_file = os.path.join(out_dir, "judge_smoke_response.json")
    Path(response_file).write_text(json.dumps(data, indent=2))

    parsed = parse_judge_scores(content, "auto")
    parsed_ok = bool(parsed and parsed.get("dimension_scores"))
    overall = None
    if parsed_ok:
        overall = parsed.get("overall")
        if overall is None:
            from eval.eval_judge import _derive_prose_overall

            dm = load_dim_map()
            dw = {k: v for k, v in dm["dimension_weights"].items() if not k.startswith("_")}
            overall = _derive_prose_overall(parsed["dimension_scores"], dw)

    return {
        "judge_smoke_parsed": parsed_ok,
        "judge_smoke": {
            "prompt_source": f"{SMOKE_PROMPTS_PATH} idx 0",
            "parse_format": parsed.get("_format") if parsed else None,
            "prose_rubric_dims": len(parsed["dimension_scores"]) if parsed_ok else 0,
            "overall_score": overall,
            "response_file": response_file,
        },
    }


def _write_receipt(out_path: str, downloaded_from_hf: bool, scratch_paths: dict,
                    api_listing: dict, gguf_load: dict) -> None:
    receipt = {
        "requirement": "PUB4-01",
        "title": "Post-upload validation — round-trip from DOWNLOADED HF artifacts",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
        "downloaded_from_hf": downloaded_from_hf,
        "scratch_paths": scratch_paths,
        "api_listing": api_listing,
        "gguf_load": gguf_load,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(receipt, indent=2))
    print(f"[pub4_validate_upload] receipt written to {out_path}")


def real_run(repo: str, manifest_path: str, out_path: str, scratch_dir: str, port: int) -> int:
    raw_manifest = json.load(open(manifest_path))
    manifest_sized = _manifest_with_local_sizes(raw_manifest)

    listing = fetch_api_listing(repo)
    api_result = compare_listing(listing, manifest_sized)
    if not api_result["ok"]:
        print(f"API LISTING MISMATCH vs manifest:\n{json.dumps(api_result, indent=2)}", file=sys.stderr)
        _write_receipt(
            out_path, downloaded_from_hf=True,
            scratch_paths={"note": "not reached -- API listing did not match the upload manifest"},
            api_listing=api_result, gguf_load={"ok": False},
        )
        return 1

    gguf_local = download_ship_gguf(repo, raw_manifest, scratch_dir)
    alias = "wp_judge_v4_roundtrip"
    proc = None
    gguf_load = {"ok": False}
    try:
        proc = serve_and_probe(gguf_local, port, alias)
        header = read_gguf_header(gguf_local)
        assert header["expert_count"] == EXPECTED_EXPERT_COUNT, (
            f"EXPERT COUNT MISMATCH on DOWNLOADED bytes: "
            f"gguf={header['expert_count']} vs expected={EXPECTED_EXPERT_COUNT}"
        )
        smoke = run_judge_smoke(port, alias, os.path.dirname(out_path) or ".")
        gguf_load = {
            "ok": True,
            "engine": "llama.cpp llama-server (~/llama.cpp/build/bin), -ngl 999 -c 12288 --jinja",
            "header": header,
            **smoke,
        }
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    scratch_paths = {
        "judge_gguf": gguf_local,
        "note": "scratch cleaned after validation; re-download to reproduce",
    }
    _write_receipt(out_path, downloaded_from_hf=True, scratch_paths=scratch_paths,
                    api_listing=api_result, gguf_load=gguf_load)
    return 0 if gguf_load.get("ok") else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", help="HF repo id, e.g. iamchum/wp-qwen3.6-35b-a3b-wp-judge-v4-gguf")
    ap.add_argument("--manifest", default="output/pkg-v4/pub4_upload_manifest.json")
    ap.add_argument("--out", default="output/pkg-v4/pub4_validation_receipt.json")
    ap.add_argument("--scratch", default="models/_hf_dl_scratch/judge_v4")
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        _self_check()
        return 0

    if not args.repo:
        ap.error("--repo is required unless --self-check")

    return real_run(args.repo, args.manifest, args.out, args.scratch, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
