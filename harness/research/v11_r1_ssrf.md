# R1: SSRF guard for `webhook_url` (v1.1 T1)

## Problem
`api.create_job` checks only `scheme in (http, https)` and a non-empty netloc. `api.webhook` POSTs with `urllib.request.urlopen`. That call:
- reaches any host the container can reach: loopback, the compose `llama` service, private LANs, and cloud metadata (169.254.169.254, fd00:ec2::254);
- resolves DNS again at send time, so a host that was public at submit time can resolve to loopback later (DNS rebinding);
- follows redirects (a 307/308 re-POSTs the report to the new `Location`);
- honours `HTTP(S)_PROXY`.

The README says this openly ("Webhook URLs are not filtered by host"). T1 removes that gap.

## Options
| | Approach | Encodings | Rebinding | Redirects | Size |
|---|---|---|---|---|---|
| A | Check `ipaddress.ip_address(hostname)` on literals only | partial: `2130706433`, `0x7f.1` and `0177.0.0.1` are not valid `ipaddress` input, but libc still connects to them | no | no | tiny |
| B | Resolve at submit, check every IP, keep `urlopen` | yes | **no**: urlopen resolves again (shown below) | no, unless a custom redirect handler is added | small |
| C | Resolve once at send, check every IP, connect to **that** IP with `http.client` (Host/SNI/cert by name); same check at submit for an early 400 | yes | **yes** | **yes**: `http.client` never follows them | ~25 lines, stdlib |
| D | Egress proxy / network policy (smokescreen, iptables, compose `internal` network) | yes | yes | yes | ops work. The harness must reach the internet for webhooks, so this cannot be the only control in compose |

**Recommendation: C.**

## Evidence (prototype in the session scratchpad, `harness/.venv/bin/python` 3.12.3, the same as the `python:3.12-slim` image)
Files: `scratchpad/r1/guard.py` (the guard), `attack.py` (static cases), `dyn.py` (redirect, rebinding, TLS).

### Static attack cases: `python attack.py` gives `FAILURES 0`
All of these are blocked (resolved via `getaddrinfo`, so we check exactly what libc would connect to):
- Loopback: `127.0.0.1`, `localhost`, `localhost.`, `127.1`, `2130706433` (decimal), `0x7f000001` and `0x7f.0.0.1` (hex), `0177.0.0.1` and `017700000001` (octal).
- Unspecified: `0`, `0.0.0.0`, `[::]`.
- IPv6 loopback and mapped forms: `[::1]`, `[::ffff:127.0.0.1]`, `[::ffff:7f00:1]`, `[0:0:0:0:0:ffff:169.254.169.254]`.
- ULA and link-local: `[fd00::1]`, `[fc00::1]`, `[fe80::1]`, `[fd00:ec2::254]` (AWS IPv6 metadata).
- Private v4: `10/8`, `172.16/12`, `192.168/16`. Metadata `169.254.169.254`. CGNAT `100.64/10`. Benchmarking `198.18/15`. Broadcast. Multicast `224.0.0.1`.
- Translation prefixes: 6to4 `[2002:7f00:1::]`, Teredo `[2001::1]`, NAT64 `[64:ff9b::7f00:1]`.
- Userinfo tricks: `evil@127.0.0.1`, `127.0.0.1:80@127.0.0.1`.
- DNS names that point at loopback: `localtest.me` (gives ::1), `127.0.0.1.nip.io`.
- Bad port `:99999`. `file:`, `gopher:` and `http:///x`. A zone id `[::1%25lo]` fails to resolve. An unresolvable `metadata.google.internal` (on this host).

Allowed: `93.184.215.14`, `https://example.com/hook`, `[2606:4700:4700::1111]`.

Two gaps in Python 3.12 `is_global`, found by the first run (it reported 2 LEAKs). The prototype fixes both:
- `224.0.0.1.is_global` is True. Fixed with `or ip.is_multicast`.
- `64:ff9b::/96` is "global", but the NAT64 gateway forwards it to the embedded v4 address. Fixed by unwrapping it to the embedded IPv4 before the check.

`::ffff:a.b.c.d` is unwrapped with `ipv4_mapped` first. In 3.12, `IPv6Address("::ffff:127.0.0.1").is_loopback` is False, so unwrapping is needed.

Allowlist (`127.0.0.0/8`):
- `127.0.0.1`, `localhost` and `::ffff:127.0.0.1` pass.
- `169.254.169.254` is still blocked.

One bug caught while prototyping: `not g or m and not allowed` parses as `not g or (m and not allowed)`, which breaks the allowlist. The condition needs parentheses. Keep the allowlist unit test.

### Dynamic cases: `python dyn.py`
```
redirect: status 302 hits {'A': 1}            # guard: 302 returned, target B never contacted
urllib today: HTTPError 501 ... ('GET')       # current code follows the 302 to B (as GET; 307/308 would re-POST the body)
pinned: connected to [('93.184.215.14', ...)] loopback hits 0     # rebinding: answer 1 public, answer 2 loopback
naive check-then-urlopen: connected to [('rebind.test', ...)] loopback hits 1   # option B is bypassed
https pinned (SNI+cert by name): 405          # TLS works through the pinned IP
```
- TLS name check still holds: with `example.com` pinned to 8.8.8.8, the send fails with `SSLCertVerificationError: Hostname mismatch`.
- Pinning cannot be abused to skip TLS verification.

## Recommended design (C)
`wpjudge/config.py`, one field:
```python
    webhook_allow_nets: tuple = ()   # CIDRs exempt from the public-address rule, e.g. 127.0.0.0/8 (env WPJUDGE_WEBHOOK_ALLOW_NETS, comma list)
```
A CIDR list instead of a bool `ALLOW_PRIVATE`:
- The existing tuple cast handles it for free.
- An operator can allow one internal hook host without also opening metadata and the `llama` service.

`wpjudge/api.py`: drop `urllib.request`, add `http.client, ipaddress, socket`. Replace `webhook` and the scheme check:
```python
NAT64 = ipaddress.ip_network("64:ff9b::/96")


def webhook_target(url: str):
    """Resolve once; every address must be public (or in webhook_allow_nets). -> (parts, port, ip). ValueError if not."""
    u = urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("webhook_url must be an http(s) URL")
    port = u.port or (443 if u.scheme == "https" else 80)  # bad port -> ValueError
    try:
        infos = socket.getaddrinfo(u.hostname, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as e:
        raise ValueError(f"webhook host does not resolve: {e}")
    allow = [ipaddress.ip_network(n) for n in CFG.webhook_allow_nets]
    ips = {ipaddress.ip_address(i[4][0].split("%")[0]) for i in infos}
    for ip in ips:
        ip = getattr(ip, "ipv4_mapped", None) or ip
        if ip in NAT64:  # 64:ff9b::a.b.c.d reaches a.b.c.d through the gateway
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if (not ip.is_global or ip.is_multicast) and not any(ip in n for n in allow):
            raise ValueError(f"webhook_url resolves to a non-public address ({ip})")
    return u, port, str(min(ips, key=lambda a: a.version))


def webhook(url: str, report: dict):
    body = json.dumps(report).encode()
    for _ in (1, 2):  # best effort: one retry, never raises
        try:
            u, port, ip = webhook_target(url)  # re-checked at send time: DNS may have changed since submit
            C = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
            c = C(u.hostname, port, timeout=5)  # Host header, SNI and cert check use the name
            c._create_connection = lambda a, *r: socket.create_connection((ip, a[1]), *r)  # connect to the checked IP only
            try:
                c.request("POST", (u.path or "/") + (f"?{u.query}" if u.query else ""), body,
                          {"Content-Type": "application/json"})
                c.getresponse()  # http.client never follows redirects
            finally:
                c.close()
            return
        except ValueError:
            return  # blocked: retrying will not help
        except Exception:
            time.sleep(1)
```
In `create_job`, the check becomes:
```python
    if webhook_url:
        try:
            await run_in_threadpool(webhook_target, webhook_url)  # getaddrinfo blocks: keep it off the event loop
        except ValueError as e:
            raise HTTPException(400, str(e))
```

Behaviour notes:
- The URL is parsed once with `urlsplit`, and the same `hostname` is used for both the check and the connect. There is no parser differential between the checker and the client, which is the classic `http://a@b` / backslash bypass class.
- A previously public URL that now resolves to a private address is blocked at send time. Nothing is sent and the request is not retried.
- The status code is ignored, as before.

## Files to change
1. `harness/wpjudge/config.py`: add the `webhook_allow_nets` field.
2. `harness/wpjudge/api.py`:
   - add `webhook_target`;
   - rewrite `webhook`;
   - change the `create_job` check;
   - change the imports: `urllib.request` out; `http.client`, `ipaddress` and `socket` in.
3. `harness/tests/test_api.py`: the tests below; `test_webhook_delivered` sets the allowlist.
4. `harness/tests/e2e.sh`: `compose_override` takes a second arg for `WPJUDGE_WEBHOOK_ALLOW_NETS`, plus the new `sec_api` checks.
5. `harness/README.md`:
   - In the `POST /jobs` 400 row, add "or `webhook_url` resolves to a non-public address".
   - Rewrite the Webhook section: public addresses only; the address is resolved and pinned; no redirects; proxy env ignored; `WPJUDGE_WEBHOOK_ALLOW_NETS`.
   - Replace the line "Webhook URLs are not filtered by host".
   - Add the env var to the config table.

## Unit tests (`tests/test_api.py`, no network needed)
```python
BLOCKED = ["http://127.0.0.1/", "http://localhost/", "http://2130706433/", "http://0x7f.1/", "http://0177.0.0.1/",
           "http://0.0.0.0/", "http://[::1]/", "http://[::ffff:127.0.0.1]/", "http://[fd00::1]/", "http://[fe80::1]/",
           "http://10.0.0.1/", "http://192.168.1.1/", "http://169.254.169.254/latest/meta-data/", "http://[fd00:ec2::254]/",
           "http://100.64.0.1/", "http://224.0.0.1/", "http://[64:ff9b::7f00:1]/", "http://evil@127.0.0.1/", "http://h:99999/"]

def test_webhook_internal_rejected(env):
    with TestClient(api.app) as c:
        for url in BLOCKED:
            assert submit(c, webhook_url=url).status_code == 400, url
        assert os.listdir(api.CFG.jobs_dir) == []

def test_webhook_allow_nets(env):
    api.CFG.webhook_allow_nets = ("127.0.0.0/8",)
    with TestClient(api.app) as c:
        assert submit(c, webhook_url="http://127.0.0.1:9/h").status_code == 202
        assert submit(c, webhook_url="http://169.254.169.254/").status_code == 400  # allowlist is not "allow all"

def test_webhook_rebind_pinned(monkeypatch):
    """answer 1 public, answer 2 loopback: the send must connect to the IP it checked."""
    real, n, seen = socket.getaddrinfo, [0], []
    def gai(h, p, *a, **k):
        n[0] += 1
        return real("93.184.215.14" if n[0] == 1 else "127.0.0.1", p, *a, **k)
    def conn(addr, *a):
        seen.append(addr[0]); raise OSError("stop")
    monkeypatch.setattr(api.socket, "getaddrinfo", gai)
    monkeypatch.setattr(api.socket, "create_connection", conn)
    monkeypatch.setattr(api.time, "sleep", lambda s: None)
    api.webhook("http://rebind.test/h", {})
    assert seen and set(seen) <= {"93.184.215.14"}  # 2nd attempt re-resolves -> loopback -> blocked, not sent

def test_webhook_no_redirect(env):
    api.CFG.webhook_allow_nets = ("127.0.0.0/8",)
    # listener A answers 307 Location: B; assert B got 0 requests after api.webhook(A, {})
```
- `test_webhook_delivered`: add `api.CFG.webhook_allow_nets = ("127.0.0.0/8",)` before submitting. Without it the test must now fail with 400, which also proves the default blocks loopback.
- `test_webhook_scheme_rejected` stays as is.
- `test_webhook_rebind_pinned` depends on `webhook` using `api.socket` at call time. `env` is not needed because `CFG.webhook_allow_nets` defaults to `()`.

## E2E test (`tests/e2e.sh sec_api`, real compose stack)
- Phase 1, auth off. Run `compose_override "" "127.0.0.0/8"`. The existing loopback listener test (`http://127.0.0.1:8766/hook`) must still deliver the report. This proves the opt-in works in the container.
- In the same phase, add:
  `c=$(code -F file=@... -F webhook_url=http://169.254.169.254/latest/meta-data/ "$API/jobs")` must give 400. This proves the allowlist does not open metadata.
- Phase 2, the existing auth-on restart. Run `compose_override "$KEY" ""` (no allowlist), then each `webhook_url` below must give **400**:
  - `http://127.0.0.1:8766/hook`
  - `http://2130706433:8766/hook`
  - `http://[::ffff:127.0.0.1]:8766/hook`
  - `http://localhost:8766/hook`
  - `http://llama:8080/health`: the compose service name resolves to a private bridge IP. This is the real SSRF target in this deployment.
- Then submit with a public-looking name that rebinds? Not feasible in compose without a DNS server. Unit tests cover rebinding and redirects. The E2E covers resolution and the env opt-in through the real container's resolver.
- Optional positive check with no allowlist: `-F webhook_url=https://example.com/` must give 202. It needs internet from the container, so skip it when offline.

## Risks
- **`HTTP(S)_PROXY` is no longer honoured** (urllib honoured it; http.client does not). A deployment whose egress needs a proxy loses webhooks. Accepted: honouring a proxy means the proxy, not us, decides the final IP, which defeats the check. Document it.
- **`_create_connection` is a private `http.client` attribute.** It is present and used by `HTTPConnection.connect` in 3.12; the image pins `python:3.12-slim`. If a future Python removes it, `test_webhook_rebind_pinned` fails loudly. The fallback is a 4-line `HTTPConnection` subclass that overrides `connect`.
- **One pinned IP, no happy-eyeballs fallback.** The guard prefers IPv4. If that address is down, the send fails even though another record might work. This matters little for a best-effort webhook.
- **Submit-time DNS.** A transient resolver failure gives a 400 and the client must resubmit. `getaddrinfo` can take seconds, so it runs in the threadpool.
- **Bad CIDR in `WPJUDGE_WEBHOOK_ALLOW_NETS`.** It raises ValueError inside the check, so every webhook submit returns 400 with the parse error. That is loud and fails closed. Validating at startup would be slightly nicer, but is optional.
- **`is_global` semantics vary across Python versions** (3.13 changed how mapped and private addresses are handled). Unwrapping mapped and NAT64 addresses first, plus the explicit multicast check, keeps the result the same. The unit list pins it.
- **Out of scope:**
  - The webhook body is the full report sent to a user-chosen public host. That is by design.
  - Response size and slowloris: the timeout is 5 s and the body is never read.
