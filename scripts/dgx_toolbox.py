"""DGX Toolbox path resolver.

Resolves paths to dgx-toolbox scripts and components. The toolbox location
is configurable via (in priority order):
1. DGX_TOOLBOX_PATH environment variable
2. config/dgx_toolbox.yaml → dgx_toolbox_path
3. Default: ~/dgx-toolbox

Usage:
    from scripts.dgx_toolbox import DGXToolbox

    dgx = DGXToolbox()
    dgx.run("vllm", "Qwen/Qwen3-30B-A3B")          # start-vllm.sh
    dgx.run("eval_toolbox")                           # eval-toolbox.sh
    dgx.run("unsloth_studio")                         # unsloth-studio.sh
    print(dgx.resolve("vllm"))                        # full path to script
    print(dgx.port("vllm"))                           # 8020
"""

import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "dgx_toolbox.yaml"


class DGXToolbox:
    """Resolve and invoke DGX Toolbox scripts."""

    def __init__(self, config_path: Path | None = None):
        self._config = self._load_config(config_path or CONFIG_PATH)
        self._base = self._resolve_base()

    def _load_config(self, path: Path) -> dict:
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
        return {}

    def _resolve_base(self) -> Path:
        """Resolve the dgx-toolbox root directory."""
        # Priority 1: Environment variable
        env_path = os.environ.get("DGX_TOOLBOX_PATH")
        if env_path:
            return Path(env_path).expanduser().resolve()

        # Priority 2: Config file
        config_path = self._config.get("dgx_toolbox_path")
        if config_path:
            return Path(config_path).expanduser().resolve()

        # Priority 3: Default
        return Path("~/dgx-toolbox").expanduser().resolve()

    @property
    def path(self) -> Path:
        """Root path of the dgx-toolbox project."""
        return self._base

    @property
    def available(self) -> bool:
        """Whether the dgx-toolbox directory exists."""
        return self._base.is_dir()

    def resolve(self, component: str) -> Path:
        """Resolve a component name to its full script path.

        Args:
            component: Key from config/dgx_toolbox.yaml components section.
                       E.g., "vllm", "eval_toolbox", "unsloth_studio".

        Returns:
            Full path to the script.

        Raises:
            FileNotFoundError: If dgx-toolbox or component script not found.
        """
        if not self.available:
            raise FileNotFoundError(
                f"DGX Toolbox not found at {self._base}. "
                f"Set DGX_TOOLBOX_PATH env var or update config/dgx_toolbox.yaml"
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
            raise FileNotFoundError(
                f"Component script not found: {full_path}. "
                f"Is dgx-toolbox installed at {self._base}?"
            )
        return full_path

    def port(self, component: str) -> int | None:
        """Get the configured port for a component."""
        return self._config.get("ports", {}).get(component)

    def shared_dir(self, name: str) -> Path:
        """Get a shared directory path."""
        dirs = self._config.get("shared_dirs", {})
        path = dirs.get(name)
        if not path:
            raise KeyError(f"Unknown shared dir '{name}'. Available: {', '.join(dirs.keys())}")
        return Path(path).expanduser().resolve()

    def run(self, component: str, *args: str, capture: bool = False) -> subprocess.CompletedProcess:
        """Run a DGX Toolbox script.

        Args:
            component: Component key (e.g., "vllm", "eval_toolbox").
            *args: Additional arguments passed to the script.
            capture: If True, capture stdout/stderr.

        Returns:
            CompletedProcess result.
        """
        script = self.resolve(component)
        cmd = ["bash", str(script)] + list(args)
        return subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
        )

    @property
    def container_workdir(self) -> str:
        """Get the working directory inside the container."""
        return self._config.get("container_workdir", "/workspace/wp-finetune")

    @property
    def extra_mounts_env(self) -> str:
        """Build EXTRA_MOUNTS env var value for dgx-toolbox container scripts.

        Replaces {project_root} with the actual project root path.
        """
        mounts = self._config.get("extra_mounts", {})
        specs = []
        for _name, spec in mounts.items():
            resolved = spec.replace("{project_root}", str(PROJECT_ROOT))
            specs.append(resolved)
        return ",".join(specs)

    def container_exec(
        self,
        *cmd: str,
        container: str = "unsloth-studio",
        capture: bool = False,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess:
        """Execute a command inside a running DGX Toolbox container.

        Args:
            *cmd: Command and arguments to run inside the container.
            container: Container name (default: unsloth-studio).
            capture: If True, capture stdout/stderr.
            timeout: Timeout in seconds.

        Returns:
            CompletedProcess result.
        """
        full_cmd = [
            "docker", "exec",
            "-w", self.container_workdir,
            container,
        ] + list(cmd)
        return subprocess.run(
            full_cmd,
            capture_output=capture,
            text=True,
            timeout=timeout,
        )

    def ensure_container(
        self,
        container: str = "unsloth-studio",
        component: str = "unsloth_studio",
    ) -> bool:
        """Ensure a DGX Toolbox container is running with project mounted.

        If not running, launches it via the component script with EXTRA_MOUNTS set.

        Returns:
            True if container is running (or was started), False on failure.
        """
        # Check if already running
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        if container in result.stdout.strip().split("\n"):
            # Verify our project is mounted
            check = subprocess.run(
                ["docker", "exec", container, "test", "-f",
                 f"{self.container_workdir}/config/train_config.yaml"],
                capture_output=True,
            )
            if check.returncode == 0:
                return True
            # Container running but project not mounted — need restart
            subprocess.run(["docker", "stop", container], capture_output=True)
            subprocess.run(["docker", "rm", container], capture_output=True)

        # Launch with EXTRA_MOUNTS
        env = os.environ.copy()
        env["EXTRA_MOUNTS"] = self.extra_mounts_env
        script = self.resolve(component)
        result = subprocess.run(
            ["bash", str(script)],
            env=env,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    @property
    def pinned_versions(self) -> dict[str, str]:
        """Get pinned dependency versions for container setup."""
        return self._config.get("pinned_versions", {})

    def status(self) -> str:
        """Run dgx-toolbox status.sh and return output."""
        result = self.run("status", capture=True)
        return result.stdout

    def vllm_endpoint(self) -> str:
        """Get the vLLM API endpoint URL."""
        port = self.port("vllm") or 8020
        return f"http://localhost:{port}/v1"

    def litellm_endpoint(self) -> str:
        """Get the LiteLLM proxy endpoint URL."""
        port = self.port("litellm") or 4000
        return f"http://localhost:{port}/v1"

    def info(self) -> dict:
        """Return toolbox info for diagnostics."""
        return {
            "path": str(self._base),
            "available": self.available,
            "config_file": str(CONFIG_PATH),
            "vllm_endpoint": self.vllm_endpoint(),
            "litellm_endpoint": self.litellm_endpoint(),
            "components": list(self._config.get("components", {}).keys()),
        }


# Module-level singleton
_instance: DGXToolbox | None = None


def get_toolbox() -> DGXToolbox:
    """Get the singleton DGXToolbox instance."""
    global _instance
    if _instance is None:
        _instance = DGXToolbox()
    return _instance


if __name__ == "__main__":
    dgx = DGXToolbox()
    if not dgx.available:
        print(f"DGX Toolbox NOT FOUND at {dgx.path}", file=sys.stderr)
        print(f"Set DGX_TOOLBOX_PATH or update config/dgx_toolbox.yaml", file=sys.stderr)
        sys.exit(1)

    print(f"DGX Toolbox: {dgx.path}")
    print(f"vLLM endpoint: {dgx.vllm_endpoint()}")
    print(f"LiteLLM endpoint: {dgx.litellm_endpoint()}")
    print(f"Components: {', '.join(dgx._config.get('components', {}).keys())}")
    print()

    # Check which components exist
    for name, rel in dgx._config.get("components", {}).items():
        full = dgx.path / rel
        status = "✓" if full.exists() else "✗"
        print(f"  {status} {name:20s} → {rel}")
