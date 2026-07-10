"""Throwaway: exercise the wp-bench execution grader WITHOUT a model.

Calls WordPressEnvironment.execute_code() with a trivial PHP snippet + empty
verification spec to confirm the `wp bench verify` path (npx shim -> wp-env run
cli) returns parseable JSON against the restored WP site. No vLLM needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "wp-bench" / "python"))

from wp_bench.config import GraderConfig  # noqa: E402
from wp_bench.environment import WordPressEnvironment  # noqa: E402

RUNTIME = PROJECT_ROOT / "wp-bench" / "runtime"


def main() -> int:
    cfg = GraderConfig(kind="docker", wp_env_dir=RUNTIME)
    env = WordPressEnvironment(cfg)
    print("[grader] setup() (wp-env start, idempotent) ...", file=sys.stderr)
    env.setup()
    code = "<?php echo 'wp-bench-grader-ok';"
    spec = {"static_checks": [], "runtime_checks": [], "judge_config": None}
    print("[grader] execute_code(trivial php) ...", file=sys.stderr)
    res = env.execute_code(code, spec)
    print(f"[grader] success={res.success}", file=sys.stderr)
    print(f"[grader] raw={res.raw}", file=sys.stderr)
    print(f"[grader] stdout(tail)={res.stdout[-400:]!r}", file=sys.stderr)
    print(f"[grader] stderr(tail)={res.stderr[-400:]!r}", file=sys.stderr)
    # success may be False if empty checks => no assertions; what matters is that
    # raw is valid JSON (grader executed), not a parse/transport failure.
    ok = isinstance(res.raw, dict) and "fatal_error" not in res.raw
    print(f"[grader] {'PASS' if ok else 'FAIL'}: grader path "
          f"{'returned parseable JSON' if ok else 'broke'}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
