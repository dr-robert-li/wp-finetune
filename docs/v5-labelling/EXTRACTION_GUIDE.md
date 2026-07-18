# Extraction guide — splitting the cataloged repos into functional code units

For the **engineering team**. This is the exhaustive specification for LABELING_PLAN.md §3:
turning the 2,226 repos in [`repos_catalog.csv`](repos_catalog.csv) into labelable functional
units without duplicating or contaminating the existing v4 judge datasets. The labeling team
never does this work — they receive finished units (LABELING_PLAN.md §6).

## 0. Inputs and outputs

Inputs: `repos_catalog.csv` (what to clone), [`v4-judge-training-dataset/`](v4-judge-training-dataset/)
(the overlap-check reference — see §4), the v3 extraction precedent (`../AGENT_PIPELINE.md`).

Outputs: `corpus_manifest.jsonl` (one row per unit, exhaustive), per-tier item packages for the
labeling team, dedup/contamination receipts (§4), with finished labels landing in
[`labelled/`](labelled/).

## 1. What counts as one functional unit

Parse PHP with a real parser (nikic/php-parser or PHP's `token_get_all` — the v3 pipeline's
extractor already does this; reuse it, don't rewrite). One unit per:

- **Top-level function** — `function foo(...) { ... }` including its docblock.
- **Class method** — each method separately, carrying `class::method` identity; the class-level
  docblock and property declarations go in context, not in the unit.
- **Hook-registered closure** — `add_action('init', function () { ... })`: the unit is the whole
  registration statement (the closure is meaningless without its hook).
- **Trait/interface methods with bodies** — same as class methods; interface signatures without
  bodies are NOT units.

NOT units: bare top-level statement blocks (template files' inline HTML/PHP — skip, they have no
callable boundary), `use`/`require` headers, property/constant declarations, generated code
(§3). Nested named functions inside a function stay inside the parent's unit (splitting them
orphans scope; count = 1 unit).

Boundary rules that make units comparable across the corpus (these matter for dedup and for
overlap-QA — two extractors must produce byte-identical units from the same file):

- Unit text runs from the first character of the docblock (or the `function`/`add_` keyword if
  no docblock) through the closing brace, inclusive. Original whitespace preserved. No trimming
  inside the span.
- Context = up to 20 physical lines above and below the unit span, plus the file's namespace and
  `use` statements verbatim.
- `unit_id = sha1(repo_sha + ":" + path + ":" + line_start)` — stable across re-runs on the same
  clone; changes iff the source changes.

## 2. Cloning discipline

- Resolve each `slug` via the WordPress.org SVN/ZIP endpoint (plugins:
  `downloads.wordpress.org/plugin/{slug}.latest-stable.zip`; themes analogous) or the
  `download_url` column when present. Record the resolved version and a sha256 of the archive as
  `repo_sha` in the manifest. Pin the whole corpus to one collection date.
- Missing/closed repos (some poor-tier plugins get delisted): record `status: unavailable` in the
  manifest with the fetch error — a visible hole, never a silent skip.
- Child themes (`parent_theme` non-empty): extract the child only; record the parent slug so
  inherited-template units aren't double-counted when the parent is also in the catalog.

## 3. Mechanical filters (logged, never silent)

Drop, with a per-repo count in the manifest receipt:

1. Units < 3 lines of body (getters/setters/pass-throughs — no judgment surface).
2. Vendored/generated trees: `vendor/`, `node_modules/`, `dist/`, `build/`, minified files
   (line length p95 > 500 chars), files with a `@generated` marker.
3. Exact duplicates: same normalized-body hash (normalize = strip comments, collapse whitespace)
   — keep the first occurrence by (catalog rank, path), record the cluster members.
4. Near-duplicates: MinHash/LSH over 5-token shingles of the normalized body, Jaccard > 0.9 —
   keep one representative per cluster, record the cluster. (Plugin ecosystems copy boilerplate
   settings-page and activation code endlessly; expect large clusters.)

## 4. Overlap avoidance with the existing v4 datasets — the part that protects the science

[`v4-judge-training-dataset/`](v4-judge-training-dataset/) contains the two reference files:

| File | Rows | Role in overlap checking |
|---|---|---|
| `openai_train_relabel_v1.jsonl` | 563 (482 `<wp_judge>`) | the v4 judge's SFT training set — new **eval** items must not overlap it |
| `openai_val.jsonl` | 141 (121 `<wp_judge>`) | the held-out eval set behind every published rho — new **training** items must not overlap it, ever |

Why both directions matter: a new T1 eval item that appeared in v4 training makes the v5-vs-v4
comparison flatter the old model (it memorized the item). A new training item that appears in
`openai_val.jsonl` contaminates the 121-item eval and silently invalidates every future rho —
this is the unrecoverable direction, treat it as a hard gate.

Procedure (run at extraction time, before any tier sampling):

1. **Extract the reference code blocks**: from each reference row's user message, take the code
   inside the ` ```php ... ``` ` fence. Normalize identically to §3.3 (strip comments, collapse
   whitespace). This yields ~600 reference bodies.
2. **Exact check**: any new unit whose normalized-body hash matches a reference body →
   `overlap: {set: train|val, kind: exact}` in the manifest.
3. **Near-dup check**: MinHash the reference bodies into the same LSH index; new units with
   Jaccard > 0.8 against any reference → `overlap: {set, kind: near, jaccard}`. The threshold is
   deliberately lower than the intra-corpus 0.9 — err toward flagging (a trimmed or lightly
   edited version of a training item is still contamination).
4. **Provenance check** (cheap belt-and-braces): the v3 corpus drew from 236 repos
   (`../../config/repos.yaml`); many are also in this catalog (the top-plugins lists overlap
   heavily). Mark every unit from those repos `v3_corpus_repo: true`. Same-repo is NOT itself
   overlap — different functions from the same plugin are fine — but same-repo + same-file +
   near-dup body is a strong contamination signal worth the extra flag.
5. **Disposition rules**:
   - `overlap` vs **`openai_val.jsonl`** (exact or near): the unit is **excluded from every
     training tier (T2/T3) permanently**; usable in T1 eval only if EXACT-identical AND its v4
     human label is being deliberately reused (rare; requires sign-off recorded in the manifest).
   - `overlap` vs **train** (exact or near): excluded from T1 eval; allowed in T2/T3 only as a
     *relabel* (the old label is superseded by the new team's — record both).
   - No overlap: unrestricted.
6. **Receipt**: `overlap_receipt.json` — counts per direction/kind, list of every flagged
   unit_id, thresholds used, reference-set hashes. No receipt, no tier sampling.

## 5. Label-format reference (why the training set is also the exemplar)

The reference rows in `v4-judge-training-dataset/` show the assistant-side format the v4 judge
was trained to emit: per-dimension prose (`WPCS Compliance: score 8/10 — ...`) followed by the
structured verdict. The labeling team's output schema (LABELING_PLAN.md §6) is the
machine-readable superset of exactly this. When a labeler or adjudicator is unsure what a
dimension's prose rationale should look like at a given score, the 482 judge rows are the
canonical exemplars — read them, don't invent a house style.

## 6. The `labelled/` output folder

Finished, QA-passed batches land in [`labelled/`](labelled/) as
`labelled/batch-{NNN}.jsonl` (rows per LABELING_PLAN.md §6 schema) plus
`labelled/batch-{NNN}.qa.json` (the §8 agreement stats for that batch). `labelled/README.md`
carries the running ledger: batch → tier → item count → QA verdict → date. Nothing enters a
training or eval build except from this folder, and only batches whose QA verdict is PASS.
