# Experiment card 04: English–French layerwise PCA

## Status

- Run ID: `2026-08-11_221142Z_load-model`
- Experiment: `04-english-french-layerwise-pca`
- Run mode: `fresh`
- Status: prepared, not launched
- Launch gate: explicit **proceed** for this card

## Objective

Compare layerwise representations of semantically paired English and French
single-turn conversations in `guidelabs/steerling-8b-instruct`. Extract one
mean-pooled user-content vector after every transformer block for 200 aligned
translation pairs (200 English and 200 French examples), fit a separate joint
2D PCA at every layer, and render the requested 32-panel plot.

## Dataset and selection

- Dataset: `Helsinki-NLP/opus-100`
- Immutable revision: `805090dc28bf78897da9641cdf08b61287580df9`
- Configuration / split: `en-fr` / `train`
- Dataset license metadata: `unknown`
- Selection: deterministic streaming shuffle with seed 42 and buffer 10,000
- Retain the first 200 aligned pairs passing all filters:
  - both sides are non-empty and distinct;
  - each content side has 4–256 model tokens;
  - `langid` identifies the English side as `en` and French side as `fr`.

OPUS-100 supplies aligned translations rather than structured multi-turn chat.
Each side is wrapped independently in the model's native instruct template as a
single `user` turn followed by the assistant-generation header. The aligned
pair ID is retained, controlling semantic content across languages.

## Model and activation measurement

- Model: `guidelabs/steerling-8b-instruct`
- Revision: `6e5a87d00d45348001810c30fe9bd65110b69fc2`
- BF16 on one Modal L40S
- Direct model loading; no generation pipeline
- Activation location: output of each of the 32
  `transformer.blocks.{0..31}` blocks, before final RMS normalization
- Per-example representation: arithmetic mean over user-content token positions
- Inputs are processed one at a time. This avoids padding tokens influencing
  Steerling's non-causal attention within a diffusion block.
- No text generation and no concept-head computation are required.

The run preserves selected texts, pair IDs, token IDs, content positions, and
content-token counts so the activation sample can be audited.

## PCA and requested plot

For each layer independently, fit scikit-learn PCA with two components over the
joint matrix of 400 vectors (English and French together), using randomized SVD
and random state 42. PCA centers but does not variance-scale features.

Render 32 subplots in a 4×8 grid:

- English: teal `#008080`
- French: salmon `#FA8072`
- alpha: 0.7
- marker edge: black
- marker edge linewidth: 0.3
- subplot title: `Layer N` only
- no x-axis or y-axis labels
- each subplot legend contains English, French, and summed two-component
  explained variance
- save 300-DPI PNG and vector PDF

## Remote execution and checkpoints

- App: `sterling-language-layerwise-pca`
- Submit with `modal run --detach`
- Timeout: 30 minutes; no automatic retries
- Cache Volume: `sterling-model-cache`
- Output Volume: `sterling-exploration-runs`
- Remote directory:
  `/mnt/run-output/exploration/runs/2026-08-11_221142Z_load-model/04-english-french-layerwise-pca`

Safe boundaries:

1. resolved config and fingerprint committed;
2. 200 filtered pairs selected and committed;
3. tokenizer/model loaded and committed;
4. activation chunk committed every 25 completed pairs;
5. merged activations and PCA coordinates committed;
6. plots and final result artifacts validated and committed.

Fresh mode refuses an existing checkpoint. Resume requires an exact fingerprint
match and continues after the last complete 25-pair chunk. A SIGTERM records and
commits stopped state.

## Artifacts

- `config.yaml`
- `checkpoint.json`
- `progress.json`
- `dashboard.html`
- `dashboard_history.jsonl`
- `selected_pairs.jsonl`
- `activation_chunks/*.npz`
- `activations.npz`
- `pca_coordinates.csv`
- `explained_variance.json`
- `layerwise_pca.png`
- `layerwise_pca.pdf`
- `results.json`
- `RESULTS.md`

## Estimated duration and cost

- Warm model cache: roughly 2–8 minutes
- Cold cache: roughly 8–20 minutes
- One L40S for the run plus modest output storage

## Launch gate

Nothing has been submitted. Reply **proceed** to approve this exact card.
