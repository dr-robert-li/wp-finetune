#!/usr/bin/env python3
"""Periodic dgx-toolbox systemic-issue watcher.

Scans `deps/dgx-toolbox/` for:
  1. submodule drift vs upstream `origin/main`
  2. known anti-patterns from existing #N entries in DGX_TOOLBOX_ISSUES.md
  3. recent (last hour) docker container exits with non-zero status

Appends NEW findings (deduped via state file) to DGX_TOOLBOX_ISSUES.md
ABOVE the "How to add issues" footer, bumps Last-updated header, and
prints a single summary line to stdout (event line for Monitor wrap).

State file: $XDG_STATE_HOME/wp-finetune/dgx_toolbox_watch.state
  -> one finding-signature per line; only unseen signatures fire.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACKER = ROOT / "DGX_TOOLBOX_ISSUES.md"
SUBMODULE = ROOT / "deps" / "dgx-toolbox"
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "wp-finetune"
STATE_FILE = STATE_DIR / "dgx_toolbox_watch.state"
FOOTER_MARKER = "## How to add issues"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state() -> set[str]:
    if not STATE_FILE.exists():
        return set()
    return {ln.strip() for ln in STATE_FILE.read_text().splitlines() if ln.strip()}


def save_state(state: set[str]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text("\n".join(sorted(state)) + "\n")


def run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return 1, str(e)


def check_submodule_drift() -> list[dict]:
    if not SUBMODULE.exists():
        return []
    rc, cur = run(["git", "rev-parse", "HEAD"], cwd=SUBMODULE)
    if rc != 0:
        return []
    cur = cur.strip().splitlines()[0]
    rc, url_out = run(["git", "config", "--get", "remote.origin.url"], cwd=SUBMODULE)
    url = url_out.strip().splitlines()[0] if rc == 0 and url_out.strip() else "unknown"
    run(["git", "fetch", "--quiet", "origin"], cwd=SUBMODULE)
    rc, ls = run(["git", "ls-remote", "--heads", "origin", "main"], cwd=SUBMODULE)
    if rc != 0 or not ls.strip():
        return []
    upstream = ls.split()[0]
    if upstream == cur:
        return []
    return [{
        "sig": f"drift:{cur[:12]}->{upstream[:12]}",
        "kind": "DRIFT",
        "body": (
            f"submodule HEAD `{cur[:12]}` differs from `{url}` `main` (`{upstream[:12]}`). "
            f"Inspect: `git -C deps/dgx-toolbox log --oneline {cur[:8]}..{upstream[:8]}`."
        ),
    }]


def check_antipatterns() -> list[dict]:
    out: list[dict] = []
    studio = SUBMODULE / "containers" / "unsloth-studio.sh"
    if studio.exists():
        text = studio.read_text()
        # Issue #4: launcher must auto-bootstrap missing studio venv before calling
        # `unsloth studio setup`. Bootstrap path = `curl install.sh | sh` per upstream
        # docs. If install.sh ref is missing AND launcher invokes studio setup, the
        # `&&` chain aborts on fresh containers and drops user back to host shell.
        has_setup = bool(re.search(r"unsloth\s+studio\s+setup", text))
        has_bootstrap = "install.sh" in text
        if has_setup and not has_bootstrap:
            out.append({
                "sig": f"ap:unsloth-studio-no-bootstrap:{hash_first(text, 12)}",
                "kind": "ANTI-PATTERN",
                "body": (
                    "`containers/unsloth-studio.sh` calls `unsloth studio setup` without "
                    "first bootstrapping the studio venv via `install.sh`. Fresh "
                    "nvcr.io/nvidia/pytorch:25.11-py3 containers ship without the venv "
                    "at `/root/.unsloth/studio/unsloth_studio`; `unsloth studio setup` "
                    "fails and the `&&` chain drops the user back to host shell. "
                    "Root cause of issue #4 (P1)."
                ),
            })
    headless = SUBMODULE / "containers" / "unsloth-headless.sh"
    if headless.exists():
        text = headless.read_text()
        if re.search(r":latest\b", text):
            out.append({
                "sig": f"ap:unsloth-headless-latest:{hash_first(text, 12)}",
                "kind": "ANTI-PATTERN",
                "body": (
                    "`containers/unsloth-headless.sh` references `:latest` tag — root cause "
                    "of issue #3 (auto-pull of incompatible upstream unsloth). Pin a "
                    "versioned tag instead."
                ),
            })
    return out


def hash_first(text: str, n: int) -> str:
    import hashlib
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def check_recent_container_exits() -> list[dict]:
    rc, _ = run(["docker", "--version"])
    if rc != 0:
        return []
    rc, ps = run([
        "docker", "ps", "-a",
        "--format", "{{.Names}}|{{.Status}}|{{.ID}}",
        "--filter", "status=exited",
    ])
    if rc != 0:
        return []
    out: list[dict] = []
    for line in ps.splitlines():
        if line.count("|") < 2:
            continue
        name, status, cid = line.split("|", 2)
        if not re.search(r"unsloth|ngc-pytorch|sparkrun|jupyter", name):
            continue
        if re.search(r"Exited \(0\)", status):
            continue
        if not re.search(r"(second|minute|hour)s? ago", status):
            continue
        # Only the most recent half-day; older state files keep history out
        if re.search(r"(\d+)\s*days?\s*ago", status):
            continue
        # Pull exit code if present
        m = re.search(r"Exited \((-?\d+)\)", status)
        code = m.group(1) if m else "?"
        sig = f"exit:{cid}:{code}"
        body = (
            f"container `{name}` ({cid}) exited with code **{code}** ({status}). "
            f"Inspect: `docker logs --tail 200 {cid}`."
        )
        if code == "137":
            body += " Code 137 = SIGKILL (OOM or manual kill); check host memory pressure."
        elif code == "1":
            body += " Code 1 = generic error; often the `&&` chain in `unsloth-studio.sh` (issue #4)."
        out.append({"sig": sig, "kind": "EXIT", "body": body})
    return out


def insert_into_tracker(entries: list[dict]) -> None:
    if not entries:
        return
    text = TRACKER.read_text()
    ts = now_iso()
    lines = [f"\n### Watch Log — {ts}\n"]
    for e in entries:
        lines.append(f"- **{e['kind']}**: {e['body']}")
    lines.append("")
    block = "\n".join(lines) + "\n"
    if FOOTER_MARKER in text:
        text = text.replace(FOOTER_MARKER, block + FOOTER_MARKER, 1)
    else:
        text = text + block
    text = re.sub(
        r"^(Last updated: ).+$",
        lambda m: m.group(1) + ts[:10],
        text, count=1, flags=re.MULTILINE,
    )
    TRACKER.write_text(text)


def main() -> int:
    state = load_state()
    candidates: list[dict] = []
    candidates += check_submodule_drift()
    candidates += check_antipatterns()
    candidates += check_recent_container_exits()
    new = [e for e in candidates if e["sig"] not in state]
    if not new:
        print(f"{now_iso()}: dgx-toolbox watch — no new systemic findings "
              f"(seen={len(state)})")
        return 0
    insert_into_tracker(new)
    for e in new:
        state.add(e["sig"])
    save_state(state)
    kinds = ", ".join(e["kind"] for e in new)
    print(f"{now_iso()}: dgx-toolbox watch — appended {len(new)} finding(s) "
          f"[{kinds}] to DGX_TOOLBOX_ISSUES.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
