# Experiment 06 — concepts above 5% activation

## Objective

Re-extract concept activations for the 24 English–French pairs used by
experiment 05 and retain every known and unknown concept whose sigmoid
activation is strictly greater than 5% for each user-content token. Rebuild the
token concept viewer from the expanded results.

## Inputs

- Model: `guidelabs/steerling-8b-instruct`
- Model revision: `6e5a87d00d45348001810c30fe9bd65110b69fc2`
  (the immutable revision used by the successful experiment 05 extraction)
- Source pairs: experiment 04's immutable `selected_pairs.jsonl`
- Pair count: 24
- Conversations: 48 (English and French for every pair)
- User-content tokens: 1,318 in the prior extraction
- Concept catalog: `.cache/concepts/concept_labels.parquet`
- Catalog rows: 134,928
- Catalog SHA-256:
  `09c3cec4ca9301f76afc848f8ad83645c4ad365937dd6c4d66fa0f5e8e749139`

## Extraction rule

- Known candidate concepts per token: 33,732
- Unknown candidate concepts per token: 101,196
- Activation: `sigmoid(raw_logit)`
- Inclusion rule: `activation > 0.05`
- Equivalent logit rule: `raw_logit > -2.944438979`
- Preserve every qualifying concept; no top-k truncation and no result cap.
- Sort qualifying concepts by activation descending within each head.
- Persist concept ID, head, raw logit, activation, provider name,
  description, group, and flags.

The model's native heads hard-sparsify effective activations to 32 known and
128 unknown concepts per token. The extraction will retain every entry in those
complete native sparse outputs above the threshold; all candidates outside the
native sparse sets have effective activation zero. This is equivalent to a
dense threshold over effective activations without materializing zero entries.

## Viewer changes

- Keep one pair visible at a time.
- Keep English above French and the concept viewer beside the text.
- Keep persistent token highlighting.
- Show every qualifying known and unknown concept for the selected token.
- Show numeric activation and a proportional fixed-length activation bar.
- Keep the concept panel scrollable.
- Load pair data on demand rather than embedding all expanded records in the
  HTML document.
- Display qualifying-concept counts for the selected token and each head.

## Compute and durability

- Modal GPU: one L40S (48 GB), BF16 model inference with FP32 score comparison.
- Detached remote-first execution.
- Immutable remote directory:
  `exploration/runs/2026-08-11_221142Z_load-model/06-token-concepts-above-5pct-v2`
- Fresh run only.
- Checkpoint after every two aligned pairs.
- Resume requires an identical configuration fingerprint.
- Atomic artifacts followed by `Volume.commit()` at every checkpoint.
- SIGTERM writes a stopped checkpoint before exit.

## Artifacts

- `config.yaml`
- `checkpoint.json`
- `progress.json`
- `dashboard.html`
- `dashboard_history.jsonl`
- `pair_data/pair_*.json.gz`
- `concept_catalog.json.gz` containing metadata for concepts actually observed
- `token_concepts.html`
- `results.json`
- `RESULTS.md`

Results will report qualifying counts by head, language, token, and pair;
minimum/median/mean/maximum concepts per token; compressed artifact sizes; and
the number of catalog-resolved versus unlabeled concepts.

## Validation

- Recompute activations from persisted logits and require every row to exceed
  0.05.
- Verify no qualifying index is omitted by comparing saved counts with the GPU
  threshold-mask counts.
- Verify IDs remain within the known/unknown head bounds.
- Verify all 24 pairs, 48 conversations, and 1,318 user-content tokens are
  present.
- Browser-check pair navigation, persistent hover, per-head counts, activation
  meters, lazy pair loading, and panel scrolling.

## Estimate

- Expected runtime: approximately 2–8 minutes after image/model cache startup.
- Expected Modal cost: under US$1 at typical L40S rates; actual billing follows
  the account's current Modal rate.
- Output size is data-dependent because 5% is a low threshold. Per-pair gzip
  files and lazy loading prevent one oversized HTML document; the run will
  report exact cardinality and storage size.

## Approval gate

Do not launch this experiment until the user sends a new explicit `proceed`.

## Revision note

The first approved submission used revision
`5612e7eb7c57858f3bf2c0983ca06e583180f51f` and stopped at 0/24 before model
loading because that snapshot is not recognized as a Steerling model. Its
checkpoint remains in `06-token-concepts-above-5pct/`. This revised card pins
the previously verified revision and uses a new immutable output directory.
