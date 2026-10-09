# Phase 3 review (shrink_run_phase3.sh, Combinations block, shrink_mem.sh)

PASS.

Checked:
1. Server args: rt_kvq8 and all c_*_kvq8 use `-fa on -ctk q8_0 -ctv q8_0`; rt_cpumoe and c_eiq3xxs_cpumoe use `--n-cpu-moe 99`; mem calls match their eval variants. Live eval process confirms c_iq4xs_kvq8 args.
2. Paths: all 7 GGUFs used exist in models/_gguf. TAG is unset, so run() keeps the c_* names unsuffixed; the mem names (q5km_f16, iq4xs_im_f16, mix_*_f16) match existing eval dirs.
3. nvidia-smi on GB10 does report per-process used_memory (live: 1881045, 15928 MiB for IQ4_XS+kvq8, plausible). No memory.json exists yet. eval_review_mem.md fixes 1-5 are all applied.
4. Isolation: phase 3 runs in sequence. Eval uses :8097 and mem uses :8098. Both scripts trap kill+wait the server, so no server or GPU memory is left over. Peak is per-PID, so other GPU users do not skew it.
5. Comparability: same CTX (5120), same flags as eval, same first wp_judge val prompt, temp 0, 2048 cap. The context cache is fixed at load, so output length barely moves the peak.

Notes (not faults, for the write-up):
- For cpumoe variants, expert weights sit in RAM, so gpu_peak_mib drops. On unified memory, compare gpu_peak_mib + ram_peak_mib as the footprint.
- With --no-mmap, VmHWM for full-GPU variants may include the transient load staging. It is consistent across variants but not pure steady state.
