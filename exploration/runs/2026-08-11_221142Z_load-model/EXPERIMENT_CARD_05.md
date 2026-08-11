# Experiment card 05: bilingual token concept viewer

## Status

- Run ID: `2026-08-11_221142Z_load-model`
- Experiment: `05-bilingual-token-concept-viewer`
- Run mode: `fresh`
- Status: prepared, not launched
- Launch gate: explicit **proceed** for this card

## Objective

Create a self-contained HTML artifact for 24 paired English/French conversations.
Display each pair in aligned language columns as model tokens. Hovering or
keyboard-focusing a user-content token reveals its top known and unknown
concepts with provider labels and catalog metadata.

## Inputs

- Reuse the first 24 pairs from experiment 04's immutable
  `selected_pairs.jsonl`; no new dataset sample is selected.
- Model: `guidelabs/steerling-8b-instruct`
- Model revision: `6e5a87d00d45348001810c30fe9bd65110b69fc2`
- BF16 on one Modal L40S
- Concept catalog: the existing provider-authored
  `guidelabs/steerling/concept_labels.parquet`
- Catalog rows: 134,928
- Catalog SHA-256:
  `09c3cec4ca9301f76afc848f8ad83645c4ad365937dd6c4d66fa0f5e8e749139`

The exact selected-pairs file and catalog are packaged into the Modal image. The
run verifies both expected counts and the catalog hash before inference.

## Measurement

For each of 48 conversations, call the native interpretable model directly with
`minimal_output=True`. Preserve only user-content token positions. For every
token record:

- token ID and decoded token string;
- top five known concept IDs, logits, and sigmoid activations;
- top five unknown concept IDs, logits, and sigmoid activations.

Resolve every concept against the catalog by `(head, concept_id)`. Include:

- concept name and full description;
- group name;
- `is_steerable`, `is_tone`, `is_alignment`, and `is_demographic` flags;
- head, concept ID, logit, and activation.

## HTML design

- Paired English/French cards in two responsive columns
- Clearly visible token boundaries with whitespace/newline glyphs
- Teal language accent for English and salmon accent for French
- Hover and keyboard-focus tooltips for every token
- Separate known/unknown concept sections inside each tooltip
- Search over pair text and a compact pair index
- Sticky explanatory header and responsive single-column mobile layout
- All data embedded locally; no external scripts, fonts, or network requests

## Remote execution and checkpoints

- App: `sterling-bilingual-token-concepts`
- Detached Modal execution; 30-minute timeout; no automatic retries
- Checkpoint every four completed pairs
- Fresh mode rejects an existing checkpoint; resume requires an identical
  fingerprint and continues after the last completed chunk
- SIGTERM writes and commits stopped state

Remote directory:

`/mnt/run-output/exploration/runs/2026-08-11_221142Z_load-model/05-bilingual-token-concept-viewer`

## Artifacts

- `config.yaml`
- `checkpoint.json`
- `progress.json`
- `dashboard.html`
- `dashboard_history.jsonl`
- `token_concepts_chunks/*.jsonl`
- `token_concepts.jsonl`
- `token_concepts.html`
- `results.json`
- `RESULTS.md`

## Estimated duration and cost

- Warm cache: approximately 1–5 minutes
- Cold cache: approximately 8–20 minutes
- One L40S plus modest artifact storage

## Launch gate

Nothing has been submitted. Reply **proceed** to approve this exact card.
