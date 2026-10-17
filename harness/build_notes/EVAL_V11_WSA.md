# EVAL v1.1 WS-A: T1 webhook SSRF (independent evaluator)

Worktree `agent-a1513c1e362c941af`. Diff `5e5ea1a..HEAD` (commits 2a5e9fb and 4d06344). venv Python 3.12.3. `wpjudge.__file__` is in the worktree. The image is python 3.12.15.

## Spec conformance
Every PLAN_v1.1 WS-A item is implemented. The code matches R1's design almost verbatim.
- `config.webhook_allow_nets` comes from env through the existing tuple cast. An empty env value gives `()`.
- `api.py`:
  - `webhook_target` parses once with urlsplit, then calls getaddrinfo.
  - It unwraps `ipv4_mapped` and NAT64 addresses, then applies the parenthesised condition `(not is_global or is_multicast) and not allowed`.
  - `webhook()` re-checks at send time.
  - The connection goes through `http.client` to the pinned IP, using `_create_connection`. Host, SNI and the cert check use the name.
  - Redirects are not followed, and a ValueError is not retried.
  - `create_job` runs the check through `run_in_threadpool`, and a ValueError becomes a 400.
  - `urllib.request` has been removed.
- The unit tests and e2e checks are all present as specified.
- One deviation is justified: the README edits are left to WS-F, which owns README per the plan table. Until WS-F lands, the README still says "Webhook URLs are not filtered by host".

## Findings
1. MINOR: IPv4-compatible `::a.b.c.d` and SIIT `::ffff:0:a.b.c.d` pass the guard.
   - Evidence: `webhook_target("http://[::7f00:1]/")` and `("http://[::ffff:0:7f00:1]/")` both return ALLOW, on the host and in the image. Python's `is_global` is True for these.
   - Impact today: none found. `connect()` gives ENETUNREACH both on the host and inside the container (tested). The kernel does not translate these forms without a sit tunnel or a SIIT gateway.
   - This is the same class as NAT64, which the code does unwrap.
   - Fix (two lines): also block `::/96` and `::ffff:0:0/96`, or unwrap them like NAT64. Add both to BLOCKED.
2. MINOR: the 400 detail leaks internal addresses and acts as a DNS oracle.
   - Example: `llama:8080` gives "(172.18.0.2)", and the response tells a caller whether an internal name resolves. A bad `WPJUDGE_WEBHOOK_ALLOW_NETS` value also shows its parse error to every caller.
   - Fix: return a generic "webhook_url not allowed", and log the details on the server.
3. MINOR: a CIDR with host bits set fails closed for every webhook, including public ones. `127.0.0.1/8` raises "has host bits set".
   - R1 accepted this risk, but it is an easy operator mistake.
   - Fix: `ip_network(n, strict=False)`, or validate at startup.
4. MINOR, accepted in R1: getaddrinfo has no timeout. Many slow-DNS hostnames can hold threadpool threads, both at submit and in the worker. It is noted only.

I found no bypasses. Everything below is blocked unless marked otherwise (`probe.py` in the scratchpad):
- encoded loopback: `127.1`, `0`, `0x7f000001`, `017700000001`, `0177.1`, fullwidth `１２７.０.０.１`, and `%31%32%37...`, which does not resolve;
- IPv6 forms: `[::]`, `[::ffff:7f00:1]`, `[::ffff:169.254.169.254]`, 6to4, Teredo, `64:ff9b:1::/48`, zone ids, `ff02::1`, `100::1`, `2001:db8::1`;
- names: `localhost.`, `LOCALHOST`, `nip.io`, `localtest.me`;
- userinfo and parser tricks: `evil.com\@127.0.0.1`, `93.184.215.14@127.0.0.1`. `127.0.0.1:80@93.184.215.14` resolves to the public address, which is correct;
- other ranges: broadcast, 198.18/15, 240/4, the docker gateway 172.17.0.1;
- ports: `:0` and an empty port, which fall back to 80;
- public mapped and NAT64 public addresses are allowed, which is correct;
- a cross-version allowlist (`::/0`) does not let IPv4 through.

Rebinding: the connect goes only to the checked IP. Redirects: none are followed. TLS: pinning `example.com` to 8.8.8.8 gives SSLCertVerificationError Hostname mismatch with a single attempt and no retry. A real HTTPS POST to example.com connected to the resolved IP with Host/SNI by name (status 405).

## Test quality
`pytest tests/`: 40 passed. I mutation-tested a scratch copy, running `-k webhook`:
- dropping the parentheses: 3 tests fail;
- removing the pin (`_create_connection`): `rebind_pinned` fails;
- no check at send time (resolve and connect unchecked): `rebind_pinned` fails;
- dropping the multicast check: 2 fail;
- dropping the NAT64 unwrap: 1 fails;
- dropping the `ipv4_mapped` unwrap: 1 fails;
- retrying a block: 0 fail. This does not matter, because the loop only makes 2 attempts and the second one is blocked anyway.

The tests are meaningful.

## Live e2e
`sg docker -c "flock /tmp/wpjudge-gpu.lock env GGUF=.../shrink-iq4xs-im_f16.gguf WORK=<scratch>/e2e bash tests/e2e.sh api allinone"`. Before the run I copied the gitignored `tests/fixtures/vuln-plugin/build/` from main, as the coordinator instructed. It is an environment artefact.

Result: **65 passed, 0 failed, exit 0**.
- Phase 1 (allowlist 127/8): loopback delivery works. `hook.json` has schema 1.0 and 15 functions, and is byte-identical to `api_report.json` after `jq -S`. The metadata URL gives 400.
- Phase 2 (no allowlist): each of these gives 400 with the resolved IP in the detail:
  - `127.0.0.1`
  - `2130706433`
  - `[::ffff:127.0.0.1]`
  - `localhost` (::1)
  - `llama:8080` (172.18.0.2)
- allinone: `http://127.0.0.1:8080/health` (the in-container llama server) gives 400.
- In-image probe (harness target, docker run): loopback, `localhost`, the docker gateway, mapped and decimal forms are all blocked. `example.com` and the public literal are allowed.
- Cleanup: compose and the aio container were removed by e2e.sh, and I removed the probe image. The llama-server on 8097 predates my run (another evaluator's) and was left alone.

## Functional sanity
The guard does what users need: internal targets are refused at submit with a clear 400, public hooks still deliver, and an operator can opt specific internal nets in without opening metadata. The only feature regression is that `HTTP(S)_PROXY` is no longer honoured. This was accepted in R1, and WS-F must document it.

VERDICT: PASS
