"""wpjudge scan <dir|zip|tar.gz> -o report.json | wpjudge serve."""
import argparse
import json
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="wpjudge")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="judge a plugin/theme dir or archive")
    s.add_argument("path")
    s.add_argument("-o", "--output", help="report path (default: stdout)")
    s.add_argument("--workers", type=int)
    s.add_argument("--base-url")
    s.add_argument("--model-file")
    v = sub.add_parser("serve", help="run the async-job API (one worker process)")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("wpjudge.api:app", host=a.host, port=a.port, workers=1)  # in-memory queue: one process only
        return 0

    from . import scan

    def progress(j, done, total):
        if j is None:
            print(f"{total} units, ETA ~{total / 5.5:.1f} min", file=sys.stderr, flush=True)

    try:
        report = scan(a.path, on_result=progress, workers=a.workers, base_url=a.base_url, model_file=a.model_file)
    except ValueError as e:  # extract.Rejected: missing path, not an archive, unsafe archive
        print(f"wpjudge: {e}", file=sys.stderr)
        return 2
    m = report["model"]
    if not m["reproducible"]:
        print(f"wpjudge: WARNING: workers={m['workers']} on a server with {m['server_slots'] or 'unknown'} slots "
              "decodes concurrently; scores and cap_reasons may change between runs. "
              "Use --workers 1 or a --parallel 1 server for a reproducible report.", file=sys.stderr, flush=True)
    text = json.dumps(report, indent=2)
    if a.output:
        with open(a.output, "w") as f:
            f.write(text + "\n")
    else:
        print(text)
    return 0  # unjudged units are reported in coverage, not as a failure exit


if __name__ == "__main__":
    sys.exit(main())
