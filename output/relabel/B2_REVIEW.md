# B2 Oracle-Gate Review — Phase C Go/No-Go

**Reviewer:** adversarial ML-eng review (async)
**Date:** 2026-07-04
**Standing rule (08.1 post-mortem):** never train on a reward that has not demonstrated offline, with CIs, that it ranks checkpoints the same way the target metric (measured judge-rho vs GT) does.

---

## (a) VERDICT: CONDITIONAL-GO

**Single strongest reason:** The in-family **calib-only** stream passes the pre-registered gate (oracle Spearman 0.886, bootstrap CI [0.31, 1.00] > 0), the signal **survives removing the non-independent epoch trajectory** (leave-the-run-out on the 3 seed-independent checkpoints gives calib +1.0), and the RL warm-starts in-family from v1.3 so the cross-family scale confound that sinks the pooled oracle cannot recur across a family boundary. That earns the *calib stream specifically* the right to be the per-step proxy — but ONLY if the anti-correlating defect stream is removed and a live scale-drift trip-wire backs it, because the one Goodhart hole an offline static-capture replay structurally cannot observe is mid-run score-scale/discrimination drift.

The default `combine()` weights (`w_calib=0.4, w_defect=0.5, w_format=0.1`) as written are a **NO-GO** — that blend puts majority weight on a stream that *anti*-correlates with the target (in-family blended oracle −0.257). Phase C proceeds only with the reward reconfigured below.

---

## (b) Exact conditions for CONDITIONAL-GO

1. **Reward = calib-dominant, defect OFF.**
   - `w_defect = 0.0` for Phase C. The defect stream anti-correlates in-family (−0.886, robust to leave-one-run-out at −1.0). Shipping it at 0.5 weight is training on an inverted reward — the exact Goodhart failure the standing rule exists to prevent.
   - `w_calib = 1.0` (or calib-dominant, ≥0.9).
   - `w_format` ≤ 0.1, and treat format as a **parseability floor / gate**, not a score driver — it exists to keep outputs parseable, not to rank quality.
   - Update `reward_v2.combine()` defaults (or the RL train config that calls it) so the shipped weights match this. Do not rely on callers remembering to override.

2. **Add a scale-drift / discrimination-collapse trip-wire** (the thing offline replay cannot see). Per read point, on the frozen val slice, log the policy's **overall-score distribution stats**: `std(pred_overall)` and `mean|pred_overall − gt_overall|`. Halt (or freeze reward and alert) if either:
   - `std(pred_overall)` falls below `0.7 × std_warmstart` (variance collapse — calib can be nudged up by regressing predictions toward the middle of the GT range, which improves mean |error| on easy items while destroying the rank information rho measures), **or**
   - `mean|pred−gt|` improves while measured rho drops between two consecutive reads (calib up, target down = the signature of Goodhart on this reward).
   This is cheap and fails fast *between* the heavier G1 reads.

3. **Keep G1 (measured judge-rho vs GT, warmstart+0.02, CI clear) as the true-north gate, unchanged.** The oracle only licenses calib as the *per-step* proxy; G1 is the real anti-Goodhart backstop. If the two ever disagree (calib rising, G1 rho flat/falling), G1 wins and the kill criteria fire. Do not weaken G1 on the strength of the oracle passing.

4. **Re-run the oracle? No hard requirement, one cheap add recommended.** Do NOT block Phase C on a re-run: the policy stays in-family from step 0, so the pooled-oracle confound is out of scope by construction. But before launch, record in the run doc the **leave-one-run-out calib oracle** (already computed: +1.0 on relabel_1ep/s1/s2; every 5-of-6 LOO in [0.8, 0.9]) so the go decision rests on the independent-point signal, not only the epoch-inflated 6-point number. Optionally, once the RL loop produces ≥3 of its own checkpoints, re-run the oracle *on those in-run checkpoints* to confirm calib still tracks rho inside the run — the only truly in-distribution test.

---

## (c) Strongest argument AGAINST this verdict (steelman for NO-GO)

The in-family pass is weaker evidence than it looks, on three compounding grounds, and arguably fails the *spirit* of the standing rule:

1. **Near-tautology.** `calib = 1 − |pred_overall − gt_overall|/100` and `judge_rho = Spearman(pred_overall, gt_overall)` are the same axis measured two ways (L1-agreement vs rank-agreement on the identical scalar). A checkpoint with lower per-item overall error almost mechanically ranks better. So "calib oracle = 0.886" is close to definitional, not an independent demonstration that optimizing calib won't Goodhart. The post-mortem rule wants evidence the reward *tracks* the target; a reward that *is* a monotone transform of the target on static captures tells you little about optimization dynamics.

2. **Non-independence inflates the CI.** 3 of the 6 in-family points are epochs of one run (full_ep1/2/3), monotone in *both* axes by training dynamics, and the bootstrap resamples **items, not checkpoints** — so the CI [0.31, 1.00] treats 6 correlated points as if they were 6 independent draws. The effective independent config count is ~3. A CI that doesn't model the actual sampling unit (checkpoints) is not the CI the pre-registered gate intended.

3. **The real risk is invisible offline.** The failure mode that matters — the policy's overall-score scale drifting so that mean |error| improves while rank discrimination collapses — is a *within-run dynamic*. Static historical captures can never exhibit it. So no amount of offline replay can clear the actual hazard; passing offline is necessary but not sufficient, and treating a pass as a green light repeats the category error (trusting a proxy the target can diverge from).

**Why it still lands at CONDITIONAL-GO, not NO-GO:** Condition (2) (variance/discrimination trip-wire) and condition (3) (live G1 rho gate) close exactly the hole the steelman identifies — they move the divergence check *inside the run*, where it can be observed and halted. And empirically the non-independence objection is blunted: the calib sign and strength survive deleting the entire epoch trajectory (LORO = +1.0 on the seed-independent points), so the signal is not merely an epoch-monotonicity artifact. The tautology objection actually cuts *for* calib on the training side: because GRPO groups are per-prompt, the within-group calib signal rewards matching *each item's own* gt, which mechanically improves rho rather than trading against it. The residual risk is real but bounded and monitored, not unmodeled.

---

## (d) Flaws in the oracle methodology itself

1. **Bootstrap resamples the wrong unit.** It resamples the 121 items and recomputes both stats over a *fixed* set of 6 checkpoints. This propagates item-level label/measurement noise but treats the 6-checkpoint ranking as the estimand with zero checkpoint-sampling uncertainty. With only 6 points, Spearman is coarse and discrete (0.886 = one inversion; ~0.31 ≈ three inversions), and the true uncertainty over *which checkpoints you happened to have* is unmodeled. A defensible CI would also resample/jackknife checkpoints (e.g. block-bootstrap by run, or report leave-one-run-out spread) — done here post hoc: LOO stays 0.8–0.9, which is reassuring but is not what the pre-registered CI actually measured.

2. **Checkpoint non-independence (the big one).** The 6 in-family points are not 6 independent evaluations of "does reward track rho": full_ep1/2/3 are epochs of a single run (a monotone trajectory in both axes — training longer raises both calib and rho, so they co-move *trivially*), and relabel_1ep/s1_ep3/s2_ep3 are seeds/config-siblings of the v1.3 recipe. Effective independent signal ≈ 2–3 configs, not 6. A 6-point Spearman on such data **overstates** how well the oracle has been demonstrated. Mitigant already in hand: LORO on the 3 seed-independent points gives calib +1.0 / defect −1.0, so the *direction* is not an artifact — but the *confidence* (CI width) implied by n=6 is optimistic.

3. **Oracle validates between-checkpoint mean-reward ranking; RL optimizes within-group z-normalized advantage.** `combine()` mean-reward across checkpoints is a different object from the MO-GRPO per-prompt z-normalized signal the policy actually sees. The oracle is the best available offline proxy but does not directly validate the within-group training signal. (This cuts toward safety for calib specifically — per-prompt calib rewards matching each item's gt — but it is a genuine gap: the thing measured ≠ the thing optimized.)

4. **`_derive_prose_overall` fallback path.** When a capture lacks an explicit `overall`, the harness derives it from dimension_scores via `dw` weights. If any in-family captures hit that path, their "overall" (and thus both calib and rho) is partly a function of the dim-weighting, not a raw model emission — a small confound worth confirming is not silently present in the s1/s2/full captures.

5. **Defect anti-correlation is diagnosable and un-salvaged for Phase C.** s1 has the *lowest* defect (0.882) yet the *highest* rho (0.827). Likely cause: the defect stream scores per-dim L1 against **median-aggregated** GT dims plus a coarse verdict pseudo-dim (PASS/FAIL thresholded at overall≥65). The best-ranking model produces more *discriminating* per-dim spreads that improve overall rank but increase L1 distance to the compressed median labels — and the verdict pseudo-dim is a near-duplicate of the overall signal, double-counting it with a lossy threshold. Net: defect measures proximity-to-noisy-medians, which anti-tracks the target. **No salvage worth attempting before the Phase C smoke** — it needs better (non-median, per-dim-calibrated) GT, which is a separate work item, not a blocker. Ship calib-only; revisit defect with improved dim labels afterward.

---

### Bottom line
Reconfigure the reward to **calib-only** (defect weight 0, format as a floor), add a **score-variance/discrimination trip-wire** at each read, keep **G1 measured-rho** as the hard gate, and record the **leave-one-run-out** oracle in the run doc so the decision rests on the independent-point signal. Under those conditions the pre-registered gate is met on the stream that will actually drive training, the pooled-oracle confound is out of scope by warm-start construction, and the one risk offline replay cannot see is covered by live monitoring. Proceed to the gated smoke.

**VERDICT: CONDITIONAL-GO** — calib-only reward passes the in-family gate and survives leave-one-run-out; defect anti-correlates and must be zeroed, with a live score-drift trip-wire plus the G1 rho gate covering the drift risk offline replay can't see.
