# V11 WS-A: T1 webhook SSRF (builder report)

Branch `worktree-agent-a1513c1e362c941af`, worktree `/home/rli/Desktop/projects/wp-finetune/.claude/worktrees/agent-a1513c1e362c941af`.
Code commit `2a5e9fb` (this report is committed after it). Implemented R1 option C as written.

## What changed
- `wpjudge/config.py` `Config`: new field `webhook_allow_nets: tuple = ()` (env `WPJUDGE_WEBHOOK_ALLOW_NETS`, comma list of CIDRs). The existing tuple cast handles it, and an empty string gives `()`.
- `wpjudge/api.py`:
  - imports: `urllib.request` removed; `http.client`, `ipaddress`, `socket` added. New module constant `NAT64`.
  - new `webhook_target(url)`: parses once with `urlsplit`, then `getaddrinfo`. It unwraps `ipv4_mapped` and NAT64, then blocks when `(not is_global or is_multicast) and not in allowlist`, with the parenthesised condition. Returns `(parts, port, ip)`, preferring IPv4, or raises ValueError. A bad port and a bad CIDR also raise ValueError.
  - `webhook()`: re-resolves and re-checks at send time, then connects with `http.client` to the checked IP through `_create_connection`. Host, SNI and the cert check use the name. Redirects are never followed. A block (ValueError) returns at once with no retry; any other error gets one retry, as before.
  - `create_job`: `await run_in_threadpool(webhook_target, webhook_url)`; a ValueError becomes a 400 carrying the message.
- `tests/test_api.py`:
  - `test_webhook_delivered` now sets `webhook_allow_nets = ("127.0.0.0/8",)`.
  - new tests:
    - `test_webhook_internal_rejected`: R1's 19-URL BLOCKED list gives 400, and no job dir is left behind.
    - `test_webhook_public_allowed`: a public v4 and v6 literal pass `webhook_target`.
    - `test_webhook_allow_nets`: 127.0.0.1 and `::ffff:127.0.0.1` give 202; metadata, multicast and 10/8 still give 400. This test fails if the parentheses are dropped.
    - `test_webhook_rebind_pinned`: the 1st DNS answer is public and the 2nd is loopback. The send connects only to the public IP, the retry re-resolves and is blocked, and `_create_connection` is pinned.
    - `test_webhook_no_redirect`: listener A answers 307 to B; B gets 0 requests.
- `tests/e2e.sh`:
  - `compose_override` takes `$2`, written as `WPJUDGE_WEBHOOK_ALLOW_NETS`.
  - `sec_api` phase 1 runs with `"" "127.0.0.0/8"` and adds a check that the metadata URL gives 400.
  - `sec_api` phase 2 runs with `"$KEY" ""` and adds a loop expecting 400 for `127.0.0.1`, `2130706433`, `[::ffff:127.0.0.1]`, `localhost` and `llama:8080/health`.
  - `sec_allinone` adds a check that `http://127.0.0.1:8080/health` gives 400.

## Tests
- `.venv/bin/python -m pytest tests/`: **40 passed** (the worktree venv; `wpjudge.__file__` is in this worktree).
- `bash -n tests/e2e.sh`: OK.

## Live run: `flock /tmp/wpjudge-gpu.lock tests/e2e.sh api allinone` (GGUF = shrink-iq4xs-im_f16)
Total: 58 passed, 7 failed. Log: `scratchpad/e2e/e2e.log` in this session.

All the new SSRF checks passed:
```
PASS api: metadata webhook -> 400 with allowlist 127/8 (got 400)
PASS api ssrf: webhook http://127.0.0.1:8766/hook -> 400 (... non-public address (127.0.0.1))
PASS api ssrf: webhook http://2130706433:8766/hook -> 400 (... (127.0.0.1))
PASS api ssrf: webhook http://[::ffff:127.0.0.1]:8766/hook -> 400 (... (127.0.0.1))
PASS api ssrf: webhook http://localhost:8766/hook -> 400 (... (::1))
PASS api ssrf: webhook http://llama:8080/health -> 400 (... (172.18.0.2))
PASS allinone: webhook to the in-container llama server -> 400 (got 400)
PASS api: file:// webhook -> 400
```
The phase-1 opt-in delivery works in the container: the listener wrote `hook.json` with `schema_version 1.0` and the full report, identical to `api_report.json`.

**The 7 failures are not caused by WS-A.** They all come from one environment gap in the worktree:
- `harness/tests/fixtures/vuln-plugin/build/index.asset.php` is matched by `.gitignore:10: build/`, so it is untracked. It exists in the main checkout but not in any git worktree.
- Without it, the fixture has 6 files instead of 7 and no `build` skip row. That gives the 3 `skip rows`/`skip counts`/`files_scanned` FAILs for each of `api_report.json` and `aio_report.json`.
- It also fails the webhook check, which asserts `(.functions | length) == 15`. The delivered report has 14 rows, the same count as the API report.

Fix (outside WS-A ownership, for the orchestrator): `git add -f harness/tests/fixtures/vuln-plugin/build/index.asset.php` on main, or a `!harness/tests/fixtures/**/build/` rule in `.gitignore`. Other worktree evaluators (B to D) will hit the same 3 to 4 FAILs in any fixture check.

## Deviations
- README not edited: WS-F owns `README.md`. WS-F needs to add:
  - the 400 row text;
  - the rewritten Webhook section (public addresses only, resolved and pinned, no redirects, `HTTP(S)_PROXY` ignored);
  - the `WPJUDGE_WEBHOOK_ALLOW_NETS` config row;
  - the removal of "Webhook URLs are not filtered by host".
- Extra test `test_webhook_public_allowed`: a no-network sanity check that the guard does not block public literals.
- Committed on the worktree branch, as the builder brief says. The plan's "orchestrator commits" rule is superseded by the brief.
- The running llama-server on port 8197 is not from this run; it was left alone. Docker cleanup was done by e2e.sh (`dc down -v`, `dk rm -f`).
