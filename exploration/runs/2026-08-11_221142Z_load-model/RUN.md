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
- No model compute has been launched and no results have been produced yet.
