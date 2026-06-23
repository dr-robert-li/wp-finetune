# Phase 09 — RL Run Logging / Telemetry Requirements (for FUTURE runs)

**Status:** REQUIREMENTS (not yet implemented). Captured 2026-06-24 after the first
full run's flat-reward episode exposed blind spots. Implement on the NEXT run — do
NOT hot-edit the live run.

**Motivation:** the current `rl_metrics.jsonl` logs only the COMPOSITE reward
(`reward_breakdown = {n_samples, reward_min, reward_max}`). It cannot answer "is
fix_correctness or consistency flat?", "are groups all-zero (GSPO gradient
starvation)?", or "is the policy collapsing (entropy)?". These additions would have
diagnosed the flat reward immediately.

**Touch points:** `scripts/rl_train.py` `_log_step` (metric assembly + write) and
`build_loss_step`/optimizer (grad_norm, lr, loss, entropy); `scripts/reward_pipeline.py`
(component means); `scripts/rl_rollouts.py` (Transition.logs already carry
fix_correctness/consistency per rollout + the `_group_id` grouping — aggregate them);
`scripts/rl_judge_dispatch.py` (consistency score distribution / latency).

---

## 1. Reward-signal diagnostics (HIGHEST PRIORITY) — per step in rl_metrics.jsonl

```jsonc
{
  "step": 100,
  "reward_mean": 0.274, "reward_std": 0.18, "reward_min": 0.0, "reward_max": 1.0,

  // Component breakdown — the critical missing piece (from Transition.logs)
  "fix_correctness_mean": 0.31, "fix_correctness_std": 0.22,
  "consistency_mean": 0.19,     "consistency_std": 0.15,

  // Group-level variance — direct GSPO/GRPO health signal
  "group_reward_std_mean": 0.12,   // mean of per-group std across batch
  "frac_groups_all_zero": 0.34,    // fraction of groups where ALL samples scored 0
  "frac_groups_all_one": 0.08,     // fraction where all scored 1
  "frac_groups_nonuniform": 0.58,  // fraction with real intra-group variance

  // Reward saturation sentinel
  "frac_reward_gt_0.9": 0.11,
  "frac_reward_lt_0.1": 0.29
}
```

**Primary kill/continue signal:** `frac_groups_all_zero` > 0.5 at steady state ==
GSPO gradient starvation (uniform groups → zero advantage → no gradient). Currently
NO visibility. Compute from the per-group reward arrays already built in
`compute_rollout_advantages` (per-prompt `_group_id` grouping).

---

## 2. Policy / optimizer health — per step

```jsonc
{
  "kl_v1": 0.0,
  "kl_ref": null,            // add if ref-KL penalty ever enabled
  "e_frac": 0.958,
  "e_frac_trend_10": -0.001, // rolling 10-step slope — pre-halt early warning
  "e_max_violation": 6.3,
  "grad_norm": 1.24,         // spikes = instability; log clip threshold too
  "grad_norm_clipped": false,
  "lr": 3e-6,                // confirm scheduler is actually ticking
  "loss": 0.412,
  "entropy": 2.31            // policy entropy — collapse <1.5 = stall/mode-exploit
}
```

**`entropy` is the single most informative add.** Low entropy + flat reward = policy
exploiting a degenerate mode (not learning). High entropy + flat reward = reward
sparsity. `e_frac_trend_10` (rolling slope) is the real pre-halt early-warning vs the
point-in-time e_frac.

---

## 3. Per-container telemetry

### Tinker trainer process (have pid/MemAvailable/AnonPages; ADD)
```
VRAM_used_GB        // torch.cuda.memory_allocated()/1e9
VRAM_reserved_GB    // torch.cuda.memory_reserved()/1e9
tokens_per_sec      // throughput — degrade = silent bottleneck
step_wall_time_s    // per-step wall — drift = OOM pressure/swap
checkpoint_wrote    // bool per step — silent write failure is possible
```

### wp_judge (vLLM :8000)
```
judge_latency_p50_ms, judge_latency_p99_ms, judge_throughput_req_s
judge_error_rate          // 5xx/timeout fraction (currently only binary "0 errors")
judge_score_distribution  // histogram [0-.2,.2-.4,.4-.6,.6-.8,.8-1]
judge_good_minus_wrong    // MARGIN, not bool (good 1.0 - wrong 0.2 = 0.8)
```

### wp_consistency (vLLM :8001)
```
consistency_latency_p50_ms
consistency_score_distribution   // same histogram
consistency_vllm_queue_depth     // backpressure proxy — growth = scoring bottleneck
```

**Histograms are key:** a bimodal 0/1 `*_score_distribution` confirms binary reward
saturation; a SHIFT of the histogram peak over steps is the EARLIEST positive learning
signal — earlier than reward_mean moving. (vLLM exposes much of this at `/metrics`.)

---

## 4. Window-mean computation — automate in the monitoring tick

```python
import pandas as pd

def compute_window_means(metrics_path: str, windows: list[tuple[int,int]]) -> dict:
    df = pd.read_json(metrics_path, lines=True)
    result = {}
    for lo, hi in windows:
        mask = (df["step"] >= lo) & (df["step"] <= hi)
        result[f"window_{lo}_{hi}"] = {
            "reward_mean":           df.loc[mask, "reward_mean"].mean(),
            "fix_correctness_mean":  df.loc[mask, "fix_correctness_mean"].mean(),
            "consistency_mean":      df.loc[mask, "consistency_mean"].mean(),
            "group_reward_std_mean": df.loc[mask, "group_reward_std_mean"].mean(),
            "frac_groups_all_zero":  df.loc[mask, "frac_groups_all_zero"].mean(),
            "entropy_mean":          df.loc[mask, "entropy"].mean(),
        }
    return result
```

Call at every decision-point tick (50/100/150/200/250...) and append automatically.
`frac_groups_all_zero` window-mean is the primary kill/continue signal next run.

---

## 5. Structured status-tick JSON (replace free-text D/E sections)

```jsonc
{
  "tick_utc": "2026-06-24T06:00:00Z", "step": 225,
  "containers": {"wp_judge": "Up 30h", "wp_consistency": "Up 29h"},
  "halt_guards": {"kl_v1": 0.0, "e_frac": 0.952, "halt": null},
  "reward": {"mean": 0.258, "std": 0.19, "min": 0.0, "max": 1.0,
             "fix_correctness": 0.29, "consistency": 0.21,
             "group_std_mean": 0.09, "frac_all_zero": 0.41},
  "policy": {"entropy": 2.18, "grad_norm": 1.31, "lr": 3e-6},
  "mem": {"anonpages_mb": 10045, "vram_used_gb": 11.2},
  "judge_quality": {"good": 1.0, "wrong": 0.0, "margin": 1.0},
  "window_means": {"0_50": 0.266, "51_100": 0.278, "101_150": 0.274, "151_200": 0.247},
  "anomalies": []
}
```

Status doc becomes auto-generatable from the tick JSON — eliminates the manual
transcription lag between metric observation and log entry.

---

## 6. Kill/continue decision rule — codify the protocol

```python
def should_flag_for_review(window_means: dict, current: dict) -> tuple[bool, str]:
    recent = window_means.get("window_151_200", {}).get("reward_mean")
    early  = window_means.get("window_0_50",    {}).get("reward_mean")
    frac_zero = window_means.get("window_151_200", {}).get("frac_groups_all_zero", 0)

    if recent is not None and early is not None:
        if recent < early - 0.01:                 # decisive window below baseline
            return True, f"DECISIVE_FLAT: w151_200={recent:.3f} < early={early:.3f}"
    if frac_zero > 0.5:
        return True, f"GROUP_COLLAPSE: {frac_zero:.1%} groups all-zero"
    if current.get("entropy", 99) < 1.5:
        return True, "ENTROPY_COLLAPSE: policy narrowing"
    return False, "OK"
```

Automated equivalent of the manual `F-STEP200` flag — fires the moment the condition
is met, not when a monitoring tick happens to be written.

---

## Implementation priority order
1. **Component means** (fix_correctness_mean, consistency_mean) — answers the flat-reward question.
2. **frac_groups_all_zero + group_reward_std_mean** — GSPO gradient-starvation diagnosis.
3. **entropy** — learning-vs-mode-collapse discriminator.
4. **score-distribution histograms** (judge + consistency) — earliest learning signal.
5. Structured tick JSON + automated window means + codified kill/continue rule.
6. Per-container latency/throughput/VRAM telemetry.

> Provenance: requirements dictated by Dr. Robert Li 2026-06-24 in response to the
> flat-reward episode on the first full GSPO run (composite-only logging hid which
> component / whether groups were collapsing).
