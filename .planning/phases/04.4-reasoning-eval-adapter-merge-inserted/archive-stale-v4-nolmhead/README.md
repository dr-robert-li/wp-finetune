# Archived: superseded 04.4 plan set (v3 merge + v4-nolmhead merge-fix)

Archived 2026-06-13 when phase 04.4 was replanned for the **v4-winner** (Tinker MoE-only
r32-rp30) post-merge re-gate (CONTEXT iteration 2, D-V4-01..08).

These plans belong to two dead iterations, both superseded:
- **01–03** — v3 merge + fidelity + REVL-04. Merged model FAILED REVL-04 (reasoning 0.3716 <
  baseline 0.4537); 19% parse failures traced to lm_head-on-extended-vocab collision.
- **06–09** — v4-nolmhead merge-fix (exclude lm_head, keep q_proj). Also FAILED REVL-04; the
  D-IT-02 attribution probe traced the damage to the **MoE deltas (RC-B)**, which re-opened
  Phase 04.3 for a MoE-only retrain → produced the v4-winner.
- **09** — promote of the v4-nolmhead candidate. Never executed (no SUMMARY); self-blocks on the
  failed nolmhead gates. Superseded by the iteration-2 promote (D-V4-08 → `...-reasoning-merged-v4`).

SUMMARY files retained as the execution record. Not consumed by execute-phase (out of active dir).
Earlier sets live in `../archive-stale-v2-prereval/` and `../archive-stale-v3-lmhead/`.
