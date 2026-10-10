"""Every harness knob. Defaults below; override any field with env WPJUDGE_<FIELD upper>."""
import os
from dataclasses import dataclass, field, fields


@dataclass
class Config:
    # model / server
    base_url: str = "http://127.0.0.1:8080/v1"
    model_file: str = "wp-judge-v4-pruned-k224.IQ4_XS.gguf"
    max_tokens: int = 2048
    workers: int = 4
    # aggregation (R2.1, R2.3)
    cap_dims: tuple = ("security", "sql_safety")
    cap_threshold: int = 4      # a cap dim scored <= this triggers the cap
    cap_value: float = 60
    coverage_threshold: float = 90
    # extraction (A3, binding change 15)
    max_chars: int = 15000
    templates: bool = True
    # archive intake
    max_files: int = 20000
    max_total_mb: int = 200
    max_ratio: int = 100
    # api (R2.4)
    max_upload_mb: int = 50
    jobs_dir: str = "jobs"
    retention_days: float = 7
    api_key: str = ""           # empty = auth off


def _cast(default, raw):
    if isinstance(default, bool):
        return raw.strip().lower() not in ("0", "false", "no", "off", "")
    if isinstance(default, tuple):
        return tuple(s.strip() for s in raw.split(",") if s.strip())
    return type(default)(raw)


def load(**overrides) -> Config:
    """Defaults <- WPJUDGE_* env <- explicit overrides (None values ignored)."""
    cfg = Config()
    for f in fields(cfg):
        raw = os.environ.get(f"WPJUDGE_{f.name.upper()}")
        if raw is not None:
            setattr(cfg, f.name, _cast(getattr(cfg, f.name), raw))
    for k, v in overrides.items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise TypeError(f"unknown config field: {k}")
        setattr(cfg, k, v)
    return cfg
