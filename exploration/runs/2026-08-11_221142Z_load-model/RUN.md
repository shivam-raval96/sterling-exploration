# Load model run

- Run ID: `2026-08-11_221142Z_load-model`
- Started: 2026-08-11 22:11:42 UTC
- Status: initialized
- Scope: a collection of related work for loading and inspecting
  `guidelabs/steerling-8b-instruct`.

## Contents

- `scripts/`: executable and notebook-support code for this run
- `results/`: generated outputs and analysis artifacts

## Run log

- Added `scripts/inspect_model.py`. It directly loads the pinned Hugging Face
  model, prints the full module representation, enumerates every module plus its
  direct parameter/buffer shapes and types, records vocabulary/tokenizer/config
  metadata, and writes `results/model_inspection.json` atomically.
- Prepared `EXPERIMENT_CARD.md` for execution in a Modal Notebook on one L40S.
- Prepared `EXPERIMENT_CARD_02.md`, which supersedes the notebook execution
  plan with a detached, checkpointed, remote-first Modal experiment. It has not
  been approved or launched.
- Experiment 02 was subsequently approved and launched as Modal app
  `ap-gXmQC1HNfIW3Do0RHv4sil`, call `fc-01KZSFX47PYDRYYKVDGAJJP8MV`. It committed
  initialization artifacts, then stopped before model loading because the image
  omitted the required `accelerate` package. The remote artifacts were pulled to
  `results/02-durable-model-structure-inspection-stopped/`.
- Prepared `EXPERIMENT_CARD_03.md` as a narrow fresh-run correction that pins
  `accelerate==1.14.0` and uses a new immutable remote artifact directory. It has
  not been approved or launched.
- Experiment 03 was approved and completed as Modal app
  `ap-NRj1LkKdKwpGcBu8lRhMm0`, call `fc-01KZSG5VTFE15NVDRPTZVQJAQF`. The run
  completed in 16.437 seconds, persisted all required artifacts, and passed 12
  of 13 automated hypothesis checks. The only failed check came from the custom
  tokenizer's `get_vocab()` exposing special tokens rather than its full
  tiktoken ID space; this is a measurement-assumption issue, not evidence of a
  12-token model. See `RESULTS.md` for the compiled analysis.
- Prepared `EXPERIMENT_CARD_04.md`, the pinned OPUS-100 English–French
  layerwise-PCA config, and a resumable extraction/plotting script. The
  experiment has not been approved or launched.
- Experiment 04 was approved and completed as Modal app
  `ap-c1VIMzZYjp9V9HQAznSFZY`, call `fc-01KZSHWH6J74YEZAYRBAS98AZB`. It selected
  200 aligned, language-validated English–French pairs and extracted 32
  layerwise activation vectors for all 400 conversations. The complete run took
  170.753 seconds. The two PCA dimensions explain 19.0%–29.3% of variance across
  layers. The PNG/PDF, coordinates, selected texts, activation chunks, merged
  activations, and run metadata were pulled into
  `results/04-english-french-layerwise-pca/`.
