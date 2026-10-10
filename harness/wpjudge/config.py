"""Every harness knob. Defaults below; override any field with env WPJUDGE_<FIELD upper>."""
import os
from dataclasses import dataclass, field, fields


@dataclass
class Config:
    # model / server
    base_url: str = "http://127.0.0.1:8080/v1"
    model_file: str = "wp-judge-v4-pruned-k224.IQ4_XS.gguf"
    max_tokens: int = 2048
    workers: int = 1           # >1 needs server --parallel >1: faster, but scores are not reproducible (T5)
    # aggregation (R2.1, R2.3)
    cap_dims: tuple = ("security", "sql_safety")
    cap_threshold: int = 4      # a cap dim scored <= this triggers the cap; also the low_rate/anti-pattern bound.
                                # Keep 4: 7 of 8 planted-bug cap reasons sit at exactly 4, so 3 misses most (v11_r2 §a)
    pattern_min: int = 8        # rank(): a dim scored >= this counts as a pattern
    cap_value: float = 60
    coverage_threshold: float = 90
    min_verdict_p: float = 0.9  # T4: verdict-token p below this marks a row low_confidence; 0 disables
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
    webhook_allow_nets: tuple = ()  # CIDRs exempt from the public-address rule, e.g. 127.0.0.0/8


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
