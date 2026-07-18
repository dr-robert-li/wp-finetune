# Manual judging plan — WordPress functional code units, full catalog

Handoff document for a data labelling team. Defines corpus, extraction, rubric, per-item
protocol, QA, logistics, and acceptance criteria for human rubric-scoring of functional
code units drawn from the project's repo catalogs. Written 2026-07-18 against rubric
v1.3 (frozen) and the v4.0 shipped judge.

---

## 1. Objective

Produce human 9-dimension rubric scores + PASS/FAIL verdicts for functional PHP code
units extracted from the cataloged WordPress repos, at a quality bar sufficient to serve
as (a) evaluation ground truth for judge models (v5 eval expansion; ~1pp resolution
target) and (b) training ground truth for future judge fine-tunes.

The catalogs and seed data ship **in this folder** (`docs/`), alongside this plan:

| File (co-located) | Rows | Role |
|---|---|---|
| [`wp_top1000_plugins_final.csv`](wp_top1000_plugins_final.csv) | 1,000 | corpus catalog — high tier (top installs) |
| [`wp_poor_plugins_final.csv`](wp_poor_plugins_final.csv) | 1,000 | corpus catalog — deliberately poor tier |
| [`wp_top100_themes_final.csv`](wp_top100_themes_final.csv) | 100 | corpus catalog — high tier |
| [`wp_poor_themes_final.csv`](wp_poor_themes_final.csv) | 186 | corpus catalog — deliberately poor tier |
| [`ugc_boundary_seeds.csv`](ugc_boundary_seeds.csv) / [`.json`](ugc_boundary_seeds.json) | 25 | boundary-case exemplars (defective code + human critique + corrected code) — the model for §4's boundary enrichment and part of T0 gold |
| [`ugc_boundary_seeds_summary.json`](ugc_boundary_seeds_summary.json) | — | boundary-seed dimension/subtlety coverage summary |
| [`ugc_seeds.json`](ugc_seeds.json) | 93 | general UGC seed exemplars — T0 gold feedstock |
| [`repos_catalog.csv`](repos_catalog.csv) | **2,226** | **NORMALIZED corpus table** — one row per unique repo, union schema of the four catalogs + `repo_type`/`catalog_tier` (48 repos are dual-tier `poor,top`: high-install AND poorly-rated/vulnerable; 12 same-file duplicate rows merged). Regenerate: `python3 normalize_assets.py` |
| [`seed_units.jsonl`](seed_units.jsonl) / [`.csv`](seed_units.csv) | 118 | **NORMALIZED seed table** — both UGC seed files unified (`code`/`annotation`/`corrected_code`; 59 deep_judge_cot + 59 critique_then_fix, 58 boundary / 60 clear-cut). JSONL canonical |
| [`normalize_assets.py`](normalize_assets.py) | — | the normalizer (`--self-check` validates source invariants) |
| **Corpus total** | **2,226 unique repos** (2,286 source rows) | |

Catalog CSV columns useful for stratification (§4): `slug` (the WordPress.org repo to
clone), `active_installs`, `rating_pct`, `last_updated`, and — plugin catalogs only —
`total_known_vulns` / `unpatched_vulns` / `max_cvss` (vulnerability-bearing repos are a
natural boundary-item source). Theme catalogs add `parent_theme` (child themes inherit
much of their surface; extraction records the parent). Downstream stages should consume the two NORMALIZED tables (`repos_catalog.csv`, `seed_units.jsonl`); the four catalog CSVs and two seed JSONs are the immutable sources of record.

## 2. Volume reality — read this before staffing

The v3 pipeline yielded ~148 extractable functional units per repo (34,855 units from
236 repos). Extrapolated:

- **~330,000 functional units** across 2,226 unique repos (projection — measure at extraction).
- Manual 9-dimension scoring runs 10–15 min/unit for a trained annotator (evidence
  quotes included).
- Fully exhaustive manual labeling ≈ **57,000–85,000 annotator-hours ≈ 28–42
  person-years ≈ $1.4M–$4.2M** at $25–50/hr, before QA overlap (+15–20%).

Exhaustive manual labeling of everything is therefore a multi-year program decision,
not a task. This plan is structured in tiers so the extraction is exhaustive (cheap,
automated) while human labeling proceeds in priority order with identical protocol at
every tier. Each tier is independently useful; stopping after any tier leaves a
complete, documented dataset.

| Tier | Items | Purpose | Effort (labeling + QA) |
|---|---|---|---|
| T0 gold | ~500 | calibration/QA anchors (double-labeled + adjudicated) | ~350 hr |
| T1 eval core | 3,500 (boundary-enriched, stratified) | v5 eval GT at ~1pp resolution | ~1,000 hr |
| T2 training core | 25,000 (stratified) | judge SFT ground truth at scale | ~6,300 hr |
| T3 full sweep | ~340,000 | exhaustive coverage | ~70,000 hr (program-level) |

## 3. Corpus extraction (automated, runs before the team starts)

Performed by engineering, not the labeling team. Exhaustive over all 2,286 repos.

1. **Clone** every repo in the four catalogs at a pinned date; record commit SHA per repo.
2. **Extract functional units**: every top-level function, class method, and closure-bearing
   hook callback in PHP files. Reuse the v3 extraction path (`docs/AGENT_PIPELINE.md`);
   unit = the complete function/method body plus its docblock.
3. **Attach context**: 20 lines above/below, file path, containing class, plugin/theme
   header (name, version), catalog tier (top/poor).
4. **Filter mechanically** (logged, not silent): units < 3 lines of body; pure
   getters/setters; generated code (vendored libs, `node_modules`-equivalents, minified);
   exact-duplicate SHA-256 bodies; near-duplicates by MinHash (Jaccard > 0.9 keeps one
   representative, records the cluster).
5. **PHPCS pre-annotation**: run WPCS PHPCS over every unit; store machine findings as
   metadata. These are HINTS for annotators, with a bias control (§9).
6. **Stratified item IDs**: `unit_id = sha1(repo_sha + path + line_start)`. Every unit gets
   a manifest row whether or not it is ever human-labeled — the manifest IS the
   exhaustive coverage record.

Deliverable: `corpus_manifest.jsonl` (one row per unit, ~340k rows) + per-tier sample
files.

## 4. Sampling design (which units humans see first)

- **T0 gold (500):** 250 from top catalogs, 250 from poor; seeded with the co-located
  [`ugc_seeds.json`](ugc_seeds.json) (93) and [`ugc_boundary_seeds.json`](ugc_boundary_seeds.json)
  (25), plus the 27 real-engagement human seeds (calibration-only, §13) — all converted
  to full rubric scores by the two most senior annotators + adjudicator. Includes 100
  deliberately clear-cut items (calibration floor/ceiling) and 400 boundary-tagged.
- **T1 eval core (3,500):** stratified by catalog tier (40% top-plugin, 30% poor-plugin,
  15% top-theme, 15% poor-theme), by unit type (handler / query-builder / output /
  utility), and **enriched toward boundary cases** modeled on the co-located
  [`ugc_boundary_seeds.csv`](ugc_boundary_seeds.csv) exemplars (defect_subtlety-tagged;
  see [`ugc_boundary_seeds_summary.json`](ugc_boundary_seeds_summary.json) for the
  dimension coverage they demonstrate). Vulnerability-bearing plugins
  (`unpatched_vulns > 0` in the plugin catalogs) are oversampled as a boundary-item
  source. Boundary items discriminate between judge models; clear-cut items add n
  without signal. Held out from all training use, permanently.
- **T2 training core (25,000):** proportional stratified sample of the remaining pool.
- **T3:** everything else, batched by repo, only on explicit program approval.

## 5. The rubric (frozen — no local interpretation)

Source of truth: `config/judge_system.md` (rubric v1.3, frozen). Annotators score the
unit **as extracted**, not the whole plugin.

Nine dimensions, each scored 1–10:

| # | Dimension | Weight | Auto-fail rules |
|---|---|---|---|
| 1 | WPCS compliance | 0.10 | critical: wrong naming convention throughout, missing PHPDoc on public API |
| 2 | SQL safety | 0.15 | **any unprepared query with dynamic values = instant FAIL** |
| 3 | Security | 0.20 | **score < 5 = automatic FAIL** regardless of other dims; injection vector, missing nonce on state-changing handler, unescaped user-controlled output |
| 4 | Performance | 0.10 | |
| 5 | WP API usage | 0.10 | |
| 6 | i18n | 0.10 | |
| 7 | Accessibility | 0.08 | |
| 8 | Error handling | 0.10 | |
| 9 | Structure/code quality | 0.07 | |

- **Verdict:** PASS requires ALL applicable dimensions ≥ 8 and no critical failures.
  There is no middle ground; when in doubt, FAIL (rubric's own instruction).
- **Overall score:** weighted sum × 10 (weights above, from
  `eval/rubric_definitions.py::DIMENSION_WEIGHTS`); computed by tooling, not by hand.
- **N/A rule:** a dimension is N/A only when the unit gives it no surface (e.g.
  accessibility on a pure data function; SQL safety with no queries). N/A is recorded
  explicitly (`null`), never scored 10. Weights renormalize over applicable dims.
- Full per-dimension anchors and worked examples: `config/judge_system.md` §1–9 —
  reproduce it verbatim in the annotator handbook. Do not paraphrase it.

## 6. Per-item protocol (what an annotator does)

1. Read the unit + context (budget: 2 min read, 10 min total median; hard stop 25 min →
   escalate, do not guess).
2. For each dimension, in fixed order 2 → 3 → 1 → 4–9 (SQL and security first — they
   carry auto-fail rules): decide applicable? → score 1–10 → **quote the evidence line(s)**
   for any score ≤ 7 and for any 9–10 on security/SQL (both directions need receipts).
3. Record the verdict per the auto-fail rules, then let tooling compute overall.
4. Tag **subtlety**: `clear-cut` or `boundary` (would a competent reviewer plausibly
   flip the verdict? then boundary) — this tag drives future sampling.
5. Tag **abstain** if the unit is unjudgeable (truncated extraction, non-WP framework
   code, generated code that escaped filters). Abstains route back to extraction QA.

Output schema (one JSONL row per judgment):

```json
{"unit_id": "...", "annotator_id": "...", "pass": false,
 "dims": {"D1_wpcs": 6, "D2_security": 3, "D3_sql": null, "D4_perf": 7,
          "D5_wp_api": 8, "D6_i18n": 2, "D7_a11y": null, "D8_errors": 5,
          "D9_structure": 7},
 "auto_fail_triggered": "D2_security<5",
 "evidence": [{"dim": "D2_security", "line": 683, "quote": "..."}],
 "subtlety": "boundary", "abstain": false, "minutes": 11,
 "rubric_version": "1.3", "timestamp": "..."}
```

## 7. Annotator qualification

- Prerequisite: working PHP + WordPress development experience (has shipped a plugin or
  theme; can read `$wpdb`, hooks, nonces cold). This rubric is not learnable from scratch
  during onboarding.
- Training: read handbook → score 40 training items with published rationales → exam of
  30 gold items. Qualification bar: verdict agreement ≥ 85% with gold, per-dimension
  Spearman ≥ 0.75, zero missed auto-fail rules. Two attempts; failure on security/SQL
  auto-fail detection is disqualifying regardless of totals.
- Requalification: monthly 10-gold-item drift check, same bars.

## 8. QA architecture

- **Overlap:** 20% of all items double-labeled (random, blind). T0 gold is 100%
  double-labeled + adjudicated.
- **Agreement bars (computed weekly, per annotator and per batch):** verdict Cohen's
  kappa ≥ 0.75; per-dimension Krippendorff's alpha ≥ 0.65; security dimension
  specifically: missed-auto-fail rate < 2%.
- **Adjudication:** disagreements on verdict or any dimension delta ≥ 3 go to a senior
  adjudicator (target < 10% of overlapped items). Adjudicated labels are final and enter
  the gold pool.
- **Drift control:** 5 gold items salted invisibly into every 100-item batch. Two
  consecutive failed salts → pause annotator, retrain.
- **Label noise estimate:** report overall human-vs-adjudicated confusion per dimension
  quarterly — this number is what downstream model evals subtract as the human ceiling
  (v3 measured ceiling: sqrt(0.969) ≈ 0.984 rho).

## 9. Machine assists and bias controls

- PHPCS findings shown as collapsible hints, **hidden for a random 50% of overlap
  items** — the overlap comparison measures hint-induced bias; if hinted-vs-unhinted
  agreement diverges (kappa gap > 0.05), hints get demoted to post-submit review.
- No LLM pre-scores shown to annotators in T0/T1 (eval GT must be independent of any
  model under test). T2/T3 MAY use model-assisted triage (LLM pre-score routes items to
  queues) provided the human sees no scores — routing only. Document whatever is used
  per batch in the batch manifest.

## 10. Logistics

- Batch = 100 items, single catalog tier per batch, delivered as JSONL + web sheet.
- Throughput planning number: 4 units/hr sustained (includes evidence quoting).
- Team sizing at that rate: T0+T1 (~1,350 hr incl. QA) = 4 annotators + 1 adjudicator
  ≈ 8–9 weeks. T2 (~6,300 hr) = 10 annotators + 2 adjudicators ≈ 4 months. T3: staff as
  a standing program only if the value case survives T2 (it likely will not — see §12).
- Weekly deliverable: labeled JSONL + QA report (agreement stats, salt pass rate,
  abstain rate, per-tier progress vs the exhaustive manifest).

## 11. Acceptance criteria (per batch)

1. Schema-valid JSONL, 100% of assigned unit_ids present (labeled or abstained).
2. Agreement bars met (§8) on the batch's overlap slice.
3. Every auto-fail verdict carries its triggering evidence quote.
4. Abstain rate < 5% (higher → extraction bug, batch bounced to engineering).
5. Rubric version stamped on every row; any rubric ambiguity escalated, never resolved
   locally (ambiguity resolutions go into a versioned FAQ appendix, monthly).

## 12. Program recommendation (opinionated, priced)

Run T0 + T1 now (~$35–70k, ~2 months): this alone delivers the v5 eval expansion —
~1pp-resolution judge evaluation, which is the gating asset for every downstream claim.
Decide T2 after the first v5 training experiment shows whether more training GT moves
the judge at all (the measured bottleneck today is the ~5pp serve-path ceiling, not GT
scale). Treat T3 (exhaustive 340k) as almost certainly not worth manual labeling: its
marginal value over a well-stratified T2 is model-training data, and by T2's end the
then-current judge + adjudication loop will label the tail cheaper than humans with
equal effective noise. The exhaustive part of this plan is the **extraction manifest**
(§3), which costs nearly nothing and preserves the option forever.

## 13. Provenance and licensing

All cataloged repos are WordPress.org-distributed (GPL-compatible). Extracted units
retain `repo`, `commit_sha`, `path`, `line` provenance in the manifest; the labeled
dataset redistributes snippets under GPLv2+ with attribution intact. No private client
code enters T0–T3 (the 27 human seeds from client engagements stay calibration-only
unless separately cleared).
