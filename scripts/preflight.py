"""Pre-flight validation script for the wp-finetune pipeline.

Checks that all required external tools and credentials are in place
before any pipeline script runs. Exits with code 1 if any check fails.

Usage:
    python scripts/preflight.py
    # or call run_preflight() from another script
"""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def run_preflight() -> None:
    """Run all pre-flight checks.

    Checks:
    1. ANTHROPIC_API_KEY environment variable is set and non-empty
    2. php --version returns exit code 0
    3. phpcs --version returns exit code 0
    4. phpcs -i output includes "WordPress-Extra"

    Prints a summary of failures to stderr and exits with code 1 if any fail.
    Prints "Pre-flight: all checks passed." on success.
    """
    failures = []

    # Check 1: ANTHROPIC_API_KEY
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        failures.append("ANTHROPIC_API_KEY environment variable is not set")

    # Check 2: php available
    try:
        php_result = subprocess.run(
            ["php", "--version"],
            capture_output=True,
            text=True,
        )
        php_ok = php_result.returncode == 0
    except FileNotFoundError:
        php_ok = False
    if not php_ok:
        failures.append("php is not installed or not in PATH")

    # Check 3: phpcs available
    try:
        phpcs_result = subprocess.run(
            ["phpcs", "--version"],
            capture_output=True,
            text=True,
        )
        phpcs_ok = phpcs_result.returncode == 0
    except FileNotFoundError:
        phpcs_ok = False

    if not phpcs_ok:
        failures.append("phpcs is not installed or not in PATH")
    else:
        # Check 4: WordPress-Extra standard installed (only if phpcs works)
        try:
            standards_result = subprocess.run(
                ["phpcs", "-i"],
                capture_output=True,
                text=True,
            )
            if "WordPress-Extra" not in standards_result.stdout:
                failures.append("WordPress-Extra coding standard is not installed in phpcs")
        except FileNotFoundError:
            failures.append("WordPress-Extra coding standard is not installed in phpcs")

    if failures:
        print(
            "ERROR: Pre-flight failed. Missing: " + ", ".join(failures),
            file=sys.stderr,
        )
        sys.exit(1)

    print("Pre-flight: all checks passed.")


if __name__ == "__main__":
    run_preflight()
