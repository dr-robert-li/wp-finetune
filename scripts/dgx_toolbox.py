"""DGX Toolbox — Execution engine for container-based ML pipelines.

Architecture: Skill (intent) → dgx_toolbox.py (resolve + validate + execute) → Docker commands

This module is the resilience layer between skills/agents and Docker containers.
It resolves paths from config, validates preconditions, generates commands dynamically,
executes with error handling, and exposes structured status for telemetry agents.

Usage:
    from scripts.dgx_toolbox import get_toolbox

    dgx = get_toolbox()

    # Validate before doing anything
    result = dgx.validate(["toolbox", "training_data", "config", "memory:70"])
    if not result.ok:
        print(result.report())
        sys.exit(1)

    # Ensure container is running with project mounted + deps installed
    dgx.ensure_ready("unsloth_studio")

    # Execute inside container (dynamically generated command)
    dgx.execute("unsloth_studio", "python", "-m", "scripts.train_model")

    # Get structured status for telemetry agents
    status = dgx.status_report()
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path.cwd() / "config" / "dgx_toolbox.yaml"


# ─── Data classes ────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of all validation checks."""
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def report(self) -> str:
        lines = []
        for c in self.checks:
            icon = "✓" if c.passed else "✗"
            lines.append(f"  {icon} {c.name}: {c.message}")
        return "\n".join(lines)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


@dataclass
class ExecResult:
    """Result of a container execution."""
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    container: str
    skipped: bool = False
    skip_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 or self.skipped

    def summary(self) -> str:
        if self.skipped:
            return f"SKIPPED: {self.skip_reason}"
        status = "OK" if self.ok else f"FAILED (exit {self.returncode})"
        return f"{status} [{self.duration_s:.1f}s] {' '.join(self.command[-3:])}"


# ─── Main class ──────────────────────────────────────────────────────────────


class DGXToolbox:
    """Execution engine for DGX Toolbox container pipelines.

    Resolves paths, validates state, generates commands dynamically,
    and provides structured status for telemetry agents.
    """

    def __init__(self, config_path: Path | None = None):
        self._config = self._load_config(config_path or CONFIG_PATH)
        self._base = self._resolve_base()
        self._project_root = Path(self._config.get("project_root", ".")).resolve()
        self._containers: dict[str, dict] = self._config.get("containers", {})
        self._exec_log: list[ExecResult] = []

    # ── Config ───────────────────────────────────────────────────────────

    def _load_config(self, path: Path) -> dict:
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
        return {}

    def _resolve_base(self) -> Path:
        env_path = os.environ.get("DGX_TOOLBOX_PATH")
        if env_path:
            return Path(env_path).expanduser().resolve()
        config_path = self._config.get("dgx_toolbox_path")
        if config_path:
            return Path(config_path).expanduser().resolve()
        return Path("~/dgx-toolbox").expanduser().resolve()

    @property
    def path(self) -> Path:
        """Root path of the dgx-toolbox project."""
        return self._base

    @property
    def available(self) -> bool:
        """Whether the dgx-toolbox directory exists."""
        return self._base.is_dir()

    @property
    def project_root(self) -> Path:
        """Root path of the project using this toolbox."""
        return self._project_root

    # ── Component resolution ─────────────────────────────────────────────

    def resolve(self, component: str) -> Path:
        """Resolve a component name to its full script path."""
        if not self.available:
            raise FileNotFoundError(
                f"DGX Toolbox not found at {self._base}. "
                f"Set DGX_TOOLBOX_PATH or update config/dgx_toolbox.yaml"
            )
        components = self._config.get("components", {})
        rel_path = components.get(component)
        if not rel_path:
            raise KeyError(
                f"Unknown component '{component}'. "
                f"Available: {', '.join(components.keys())}"
            )
        full_path = self._base / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Component not found: {full_path}")
        return full_path

    def port(self, component: str) -> int | None:
        """Get the configured port for a component."""
        return self._config.get("ports", {}).get(component)

    @property
    def pinned_versions(self) -> dict[str, str]:
        """Pinned dependency versions for container setup."""
        return self._config.get("pinned_versions", {})

    def vllm_endpoint(self) -> str:
        port = self.port("vllm") or 8020
        return f"http://localhost:{port}/v1"

    def litellm_endpoint(self) -> str:
        port = self.port("litellm") or 4000
        return f"http://localhost:{port}/v1"

    # ── Validation engine ────────────────────────────────────────────────

    def validate(self, checks: list[str]) -> ValidationResult:
        """Run named validation checks. Returns structured result.

        Available checks:
            "toolbox"        — dgx-toolbox directory exists
            "training_data"  — training data file exists (path from config)
            "config"         — config file exists (path from config)
            "memory:N"       — at least N GB available memory
            "container:name" — named container is running
            "mounted:name"   — project is mounted in named container
            "gpu"            — GPU is accessible (via container)
            "deps:name"      — pinned deps installed in container
        """
        result = ValidationResult()
        for check in checks:
            if ":" in check:
                name, arg = check.split(":", 1)
            else:
                name, arg = check, ""
            result.checks.append(self._run_check(name, arg))
        return result

    def _run_check(self, name: str, arg: str) -> CheckResult:
        dispatch = {
            "toolbox": self._check_toolbox,
            "training_data": self._check_training_data,
            "config": self._check_config,
            "memory": self._check_memory,
            "container": self._check_container,
            "mounted": self._check_mounted,
            "gpu": self._check_gpu,
            "deps": self._check_deps,
        }
        fn = dispatch.get(name)
        if not fn:
            return CheckResult(name, False, f"Unknown check: {name}")
        try:
            return fn(arg)
        except Exception as e:
            return CheckResult(name, False, f"Check error: {e}")

    def _check_toolbox(self, _: str) -> CheckResult:
        if self.available:
            return CheckResult("toolbox", True, f"Found at {self._base}")
        return CheckResult("toolbox", False, f"Not found at {self._base}")

    def _check_training_data(self, _: str) -> CheckResult:
        rel = self._config.get("validation_paths", {}).get("training_data", "data/final_dataset/openai_train.jsonl")
        path = self._project_root / rel
        if path.exists():
            lines = sum(1 for _ in open(path))
            return CheckResult("training_data", True, f"{lines} examples", {"path": str(path), "lines": lines})
        return CheckResult("training_data", False, f"Not found: {path}")

    def _check_config(self, _: str) -> CheckResult:
        rel = self._config.get("validation_paths", {}).get("config", "config/train_config.yaml")
        path = self._project_root / rel
        if path.exists():
            return CheckResult("config", True, f"Found: {path}")
        return CheckResult("config", False, f"Not found: {path}")

    def _check_memory(self, arg: str) -> CheckResult:
        min_gb = float(arg) if arg else 70.0
        try:
            meminfo = Path("/proc/meminfo").read_text()
            mem = {l.split(":")[0].strip(): int(l.split(":")[1].strip().split()[0])
                   for l in meminfo.splitlines() if ":" in l}
            avail_gb = mem.get("MemAvailable", 0) / (1024 * 1024)
            if avail_gb >= min_gb:
                return CheckResult("memory", True, f"{avail_gb:.1f} GB available (need {min_gb})",
                                   {"available_gb": avail_gb, "required_gb": min_gb})
            # Get top memory consumers for diagnostics
            ps = subprocess.run(["ps", "aux", "--sort=-rss"], capture_output=True, text=True, timeout=5)
            top_procs = []
            for line in ps.stdout.strip().split("\n")[1:6]:
                parts = line.split(None, 10)
                if len(parts) >= 11:
                    rss_mb = int(parts[5]) // 1024 if parts[5].isdigit() else 0
                    if rss_mb > 50:
                        top_procs.append(f"{parts[0]}:{rss_mb}MB:{parts[10][:40]}")
            docker = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=5)
            containers = docker.stdout.strip().split("\n") if docker.stdout.strip() else []
            return CheckResult("memory", False,
                               f"{avail_gb:.1f} GB available, need {min_gb} GB",
                               {"available_gb": avail_gb, "required_gb": min_gb,
                                "top_processes": top_procs, "running_containers": containers})
        except Exception as e:
            return CheckResult("memory", False, f"Cannot check: {e}")

    def _check_container(self, name: str) -> CheckResult:
        cname = self._containers.get(name, {}).get("container_name", name)
        result = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                                capture_output=True, text=True, timeout=5)
        running = result.stdout.strip().split("\n")
        if cname in running:
            # Check for restart loop — container shows as running but can't be exec'd
            inspect = subprocess.run(
                ["docker", "inspect", cname, "--format",
                 "{{.State.Restarting}} {{.RestartCount}}"],
                capture_output=True, text=True, timeout=5)
            if inspect.returncode == 0:
                parts = inspect.stdout.strip().split()
                restarting = parts[0] == "true" if parts else False
                restart_count = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                if restarting or restart_count > 2:
                    return CheckResult(
                        f"container:{name}", False,
                        f"{cname} is in a restart loop (restarts={restart_count})",
                        {"restart_count": restart_count, "restarting": restarting})
            return CheckResult(f"container:{name}", True, f"{cname} is running")
        return CheckResult(f"container:{name}", False, f"{cname} is not running")

    def _check_mounted(self, name: str) -> CheckResult:
        mapping = self._containers.get(name, {})
        cname = mapping.get("container_name", name)
        workdir = mapping.get("workdir", "/workspace/wp-finetune")
        check_file = self._config.get("mount_check_file", "config/train_config.yaml")
        check = subprocess.run(
            ["docker", "exec", cname, "test", "-f", f"{workdir}/{check_file}"],
            capture_output=True, timeout=5)
        if check.returncode == 0:
            return CheckResult(f"mounted:{name}", True, f"Project visible at {workdir}")
        return CheckResult(f"mounted:{name}", False,
                           f"Project not mounted in {cname}. Restart with EXTRA_MOUNTS.")

    def _check_gpu(self, name: str) -> CheckResult:
        cname = self._containers.get(name, {}).get("container_name", name) if name else "unsloth-headless"
        result = subprocess.run(
            ["docker", "exec", cname, "nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return CheckResult("gpu", True, result.stdout.strip())
        return CheckResult("gpu", False, "No GPU access in container")

    def _check_deps(self, name: str) -> CheckResult:
        cname = self._containers.get(name, {}).get("container_name", name)
        workdir = self._containers.get(name, {}).get("workdir", "/workspace")
        # Check critical imports
        imports = self._config.get("required_imports", ["unsloth", "trl", "peft", "datasets", "mlflow", "yaml", "scipy"])
        check_script = "import " + ",".join(imports) + ";print('OK')"
        result = subprocess.run(
            ["docker", "exec", "-w", workdir, cname, "python", "-c", check_script],
            capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and "OK" in result.stdout:
            return CheckResult(f"deps:{name}", True, "All pinned deps available")
        missing = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown"
        return CheckResult(f"deps:{name}", False, f"Missing deps: {missing}")

    # ── Container lifecycle ──────────────────────────────────────────────

    def ensure_ready(self, component: str, wait: int = 45) -> ValidationResult:
        """Ensure a container is running, project mounted, and deps installed.

        This is the main entry point for preparing a container for work.
        It handles: start → wait → mount check → dep install → validate.

        Args:
            component: Key from containers config (e.g., "unsloth_studio")
            wait: Seconds to wait after starting container

        Returns:
            ValidationResult with all checks.
        """
        mapping = self._containers.get(component)
        if not mapping:
            r = ValidationResult()
            r.checks.append(CheckResult(component, False, f"Unknown component: {component}"))
            return r

        cname = mapping["container_name"]
        workdir = mapping.get("workdir")

        # Step 1: Is container running (and not in a restart loop)?
        container_check = self._check_container(component)
        if not container_check.passed:
            # If restart loop detected, clean up first
            if container_check.details.get("restart_count", 0) > 0:
                print(f"  {cname} is in a restart loop — removing and recreating...")
                subprocess.run(["docker", "stop", cname], capture_output=True, timeout=30)
                subprocess.run(["docker", "rm", cname], capture_output=True, timeout=10)
            print(f"  Starting {cname} via dgx-toolbox...")
            self._start_container(component)
            print(f"  Waiting {wait}s for setup...")
            time.sleep(wait)

        # Step 2: Is project mounted?
        mount_check = self._check_mounted(component)
        if not mount_check.passed:
            print(f"  Project not mounted. Restarting {cname} with EXTRA_MOUNTS...")
            subprocess.run(["docker", "stop", cname], capture_output=True, timeout=30)
            subprocess.run(["docker", "rm", cname], capture_output=True, timeout=10)
            self._start_container(component)
            print(f"  Waiting {wait}s for setup...")
            time.sleep(wait)

        # Step 3: Are deps installed?
        deps_check = self._check_deps(component)
        if not deps_check.passed:
            print(f"  Installing pinned dependencies...")
            self._install_deps(component)

        # Step 4: Final validation
        return self.validate([
            f"container:{component}",
            f"mounted:{component}",
            f"gpu:{cname}",
            f"deps:{component}",
        ])

    def _start_container(self, component: str) -> None:
        """Start a container via dgx-toolbox with EXTRA_MOUNTS."""
        mapping = self._containers[component]
        script = self.resolve(mapping["component"])
        env = os.environ.copy()
        env["EXTRA_MOUNTS"] = f"{self._project_root}:{mapping['workdir']}"
        subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)

    def _install_deps(self, component: str) -> None:
        """Install pinned deps inside a container."""
        cname = self._containers[component]["container_name"]
        versions = self.pinned_versions
        if not versions:
            return
        # Build pip install command from pinned versions
        pkgs = [f"{pkg}=={ver}" for pkg, ver in versions.items()]
        extras = self._config.get("extra_deps", ["pyyaml", "python-dotenv", "scipy", "mlflow", "peft", "hf_transfer"])
        cmd = ["docker", "exec", cname, "pip", "install", "--no-deps"] + pkgs
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        # Install extras (with deps)
        cmd2 = ["docker", "exec", cname, "pip", "install"] + extras
        subprocess.run(cmd2, capture_output=True, text=True, timeout=300)

    # ── Execution engine ─────────────────────────────────────────────────

    def execute(
        self,
        component: str,
        *cmd: str,
        capture: bool = False,
        timeout: int | None = None,
        idempotency_check: str | None = None,
    ) -> ExecResult:
        """Execute a command inside a container. The core execution method.

        Args:
            component: Key from containers config (e.g., "unsloth_studio")
            *cmd: Command and args (e.g., "python", "-m", "scripts.train_model")
            capture: Capture stdout/stderr (default: stream to terminal)
            timeout: Timeout in seconds
            idempotency_check: Path to check inside container — if exists, skip execution

        Returns:
            ExecResult with structured output.
        """
        mapping = self._containers.get(component)
        if not mapping:
            return ExecResult(list(cmd), 1, "", f"Unknown component: {component}", 0, "")

        cname = mapping["container_name"]
        workdir = mapping.get("workdir")

        # Idempotency: skip if output already exists
        if idempotency_check:
            check = subprocess.run(
                ["docker", "exec", cname, "test", "-e", idempotency_check],
                capture_output=True, timeout=5)
            if check.returncode == 0:
                result = ExecResult(list(cmd), 0, "", "", 0, cname,
                                    skipped=True, skip_reason=f"Already exists: {idempotency_check}")
                self._exec_log.append(result)
                return result

        # Build and run command
        full_cmd = ["docker", "exec"]
        if workdir:
            full_cmd += ["-w", workdir]
        full_cmd += [cname] + list(cmd)

        start = time.time()
        proc = subprocess.run(
            full_cmd,
            capture_output=capture,
            text=True,
            timeout=timeout,
        )
        duration = time.time() - start

        result = ExecResult(
            command=list(cmd),
            returncode=proc.returncode,
            stdout=proc.stdout if capture else "",
            stderr=proc.stderr if capture else "",
            duration_s=duration,
            container=cname,
        )
        self._exec_log.append(result)
        return result

    def run_service(self, component: str, *args: str) -> ExecResult:
        """Start a dgx-toolbox service (vLLM, LiteLLM, etc.) — not exec'd into.

        Args:
            component: Key from components config (e.g., "vllm")
            *args: Additional arguments (e.g., model name)
        """
        script = self.resolve(component)
        env = os.environ.copy()
        mount_target = self._containers.get(component, {}).get("workdir", "/workspace")
        env["EXTRA_MOUNTS"] = f"{self._project_root}:{mount_target}"
        start = time.time()
        proc = subprocess.run(
            ["bash", str(script)] + list(args),
            env=env, capture_output=True, text=True, timeout=60,
        )
        duration = time.time() - start
        cname = self._containers.get(component, {}).get("container_name", component)
        result = ExecResult(
            command=["bash", str(script)] + list(args),
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_s=duration,
            container=cname,
        )
        self._exec_log.append(result)
        return result

    # ── Telemetry / status ───────────────────────────────────────────────

    def status_report(self) -> dict:
        """Structured status for telemetry agents to consume.

        Returns a dict with container states, execution log, resource usage,
        and pipeline progress — everything a background observer needs.
        """
        # Container states
        ps = subprocess.run(["docker", "ps", "--format", "json"],
                            capture_output=True, text=True, timeout=5)
        containers = []
        if ps.stdout.strip():
            for line in ps.stdout.strip().split("\n"):
                try:
                    containers.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        # Memory
        try:
            meminfo = Path("/proc/meminfo").read_text()
            mem = {l.split(":")[0].strip(): int(l.split(":")[1].strip().split()[0])
                   for l in meminfo.splitlines() if ":" in l}
            memory = {
                "total_gb": round(mem.get("MemTotal", 0) / (1024 * 1024), 1),
                "available_gb": round(mem.get("MemAvailable", 0) / (1024 * 1024), 1),
            }
        except Exception:
            memory = {}

        # Pipeline artifacts (config-driven)
        artifact_defs = self._config.get("status_artifacts", {})
        artifacts = {}
        for name, defn in artifact_defs.items():
            rel_path = defn.get("path", "")
            full = self._project_root / rel_path
            if defn.get("type") == "glob":
                pattern = defn.get("pattern", "*")
                artifacts[name] = len(list(full.glob(pattern))) if full.exists() else 0
            else:
                artifacts[name] = full.exists()

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dgx_toolbox_path": str(self._base),
            "dgx_toolbox_available": self.available,
            "project_root": str(self._project_root),
            "containers": containers,
            "memory": memory,
            "artifacts": artifacts,
            "execution_log": [
                {"command": r.command, "status": r.summary(), "duration_s": r.duration_s}
                for r in self._exec_log[-20:]  # Last 20 executions
            ],
            "endpoints": {
                "vllm": self.vllm_endpoint(),
                "litellm": self.litellm_endpoint(),
            },
        }

    # ── Convenience ──────────────────────────────────────────────────────

    def info(self) -> dict:
        """Basic toolbox info for diagnostics."""
        return {
            "path": str(self._base),
            "available": self.available,
            "config_file": str(CONFIG_PATH),
            "vllm_endpoint": self.vllm_endpoint(),
            "litellm_endpoint": self.litellm_endpoint(),
            "components": list(self._config.get("components", {}).keys()),
            "containers": list(self._containers.keys()),
        }


# ─── Singleton ───────────────────────────────────────────────────────────────

_instance: DGXToolbox | None = None


def get_toolbox() -> DGXToolbox:
    """Get the singleton DGXToolbox instance."""
    global _instance
    if _instance is None:
        _instance = DGXToolbox()
    return _instance


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"
    dgx = DGXToolbox()

    if cmd == "info":
        if not dgx.available:
            print(f"DGX Toolbox NOT FOUND at {dgx.path}", file=sys.stderr)
            print(f"Set DGX_TOOLBOX_PATH or update config/dgx_toolbox.yaml", file=sys.stderr)
            sys.exit(1)
        print(f"DGX Toolbox: {dgx.path}")
        print(f"Endpoints: vLLM={dgx.vllm_endpoint()}, LiteLLM={dgx.litellm_endpoint()}")
        print(f"\nComponents:")
        for name, rel in dgx._config.get("components", {}).items():
            full = dgx.path / rel
            icon = "✓" if full.exists() else "✗"
            print(f"  {icon} {name:20s} → {rel}")
        print(f"\nContainers:")
        for name, mapping in dgx._containers.items():
            print(f"  {name:20s} → {mapping['container_name']:20s} ({mapping['purpose'][:50]})")

    elif cmd == "validate":
        checks = sys.argv[2:] or ["toolbox", "training_data", "config", "memory:70"]
        result = dgx.validate(checks)
        print(result.report())
        sys.exit(0 if result.ok else 1)

    elif cmd == "status":
        print(json.dumps(dgx.status_report(), indent=2, default=str))

    else:
        print(f"Usage: python scripts/dgx_toolbox.py [info|validate|status]")
