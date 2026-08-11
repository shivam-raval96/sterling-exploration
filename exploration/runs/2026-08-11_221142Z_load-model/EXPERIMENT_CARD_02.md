# Experiment card 02: durable model-structure inspection

## Status and launch gate

- Run ID: `2026-08-11_221142Z_load-model`
- Experiment within run: `02-durable-model-structure-inspection`
- Run mode: `fresh`
- Status: prepared, not launched
- Approval: requires a new explicit **proceed** for this exact card

This card supersedes the execution plan in `EXPERIMENT_CARD.md`; that file is
retained as historical context. Approval of an earlier card does not authorize
this experiment. No Modal app, GPU, or model download may start until approval.

## Objective

Load the pinned `guidelabs/steerling-8b-instruct` checkpoint directly on Modal,
print its full PyTorch representation, inventory its architecture and tensor
shapes, and persist a validated machine-readable description. This experiment
does not generate responses, evaluate prompts, judge jailbreaks, steer concepts,
or alter model weights.

## Model and immutable inputs

- Hugging Face model: `guidelabs/steerling-8b-instruct`
- Model revision: `6e5a87d00d45348001810c30fe9bd65110b69fc2`
- Loader: `AutoModel.from_pretrained(..., trust_remote_code=True)`
- Tokenizer: `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`
- Dtype: BF16
- Device placement: CUDA; no CPU fallback
- GPU: one Modal L40S (48 GB)
- Inspection logic: `scripts/inspect_model.py`
- Dataset/prompts: none
- Randomness: none expected; record dependency and platform versions anyway

The pinned remote model code is executable third-party code. Changing the model
revision or dependency pins is a material change requiring another card.

## Remote-first execution design

Before launch, add a small Modal entrypoint that packages the verified inspection
script into an immutable image and executes it as a detached one-off function.
Keep imports and other global-scope work lightweight.

- Submitter: `modal run --detach`
- App name: `sterling-load-model-inspection`
- Timeout: 30 minutes
- Retry policy: at most one retry, and only after the entrypoint confirms that
  the persisted checkpoint/config fingerprint permits a safe resume
- Cache Volume: `sterling-model-cache`, mounted read/write at
  `/mnt/model-cache`
- Output Volume: `sterling-exploration-runs`, mounted read/write at
  `/mnt/run-output`
- Remote run directory:
  `/mnt/run-output/exploration/runs/2026-08-11_221142Z_load-model/02-durable-model-structure-inspection`
- Credentials: named Modal secrets only; never write tokens to artifacts

The output Volume, not the container filesystem or local submitter, is the
source of truth. Every checkpoint is written atomically where practical and
followed by `Volume.commit()`.

## Resolved configuration and fingerprint

Before submission, write `config.yaml` containing every setting in this card,
including model/revision, dependency pins, dtype, GPU, paths, timeout, retry
policy, and `run_mode: fresh`. Compute and store a SHA-256 fingerprint over the
canonical resolved configuration.

A later resume is permitted only if:

1. `checkpoint.json` exists and has status `running` or `stopped`;
2. its configuration fingerprint exactly matches;
3. it contains all state needed for the next phase; and
4. the invocation explicitly sets `run_mode: resume`.

Otherwise, stop and prepare a new card rather than reusing artifacts.

## Procedure and safe boundaries

1. **Initialize:** persist `config.yaml`, `checkpoint.json`, `progress.json`,
   `dashboard.html`, and the first `dashboard_history.jsonl` event; commit.
2. **Load tokenizer:** load the pinned tokenizer, record its class and resolved
   vocabulary metadata, update checkpoint/progress/dashboard; commit.
3. **Load model:** load the pinned model in BF16 on CUDA, record its resolved
   class, device map, and config, then checkpoint; commit.
4. **Inspect:** call the already-prepared inspection functions, capture
   `print(model)`, enumerate all modules, parameters, buffers, shapes, dtypes,
   devices, and storage sizes; checkpoint; commit.
5. **Validate:** write `model_inspection.json` atomically, parse it back, check
   required keys and invariants, then write `results.json` and `RESULTS.md`;
   mark the checkpoint completed and commit.
6. **Retrieve:** pull the complete remote experiment directory into this run's
   local `results/` area without overwriting unrelated artifacts.

Install a SIGTERM handler. A controlled stop records `status: stopped`, the last
completed phase, error/stop context, and commits all current artifacts before
exiting.

## Measurements

Primary measurements:

- resolved model and tokenizer qualified class names;
- complete model configuration and `print(model)` output;
- module count and counts by module type;
- every module path, depth, type, direct parameters, and direct buffers;
- every parameter/buffer shape, rank, element count, dtype, device, byte size,
  and trainability;
- total and trainable parameter counts and total parameter storage;
- tokenizer length, `get_vocab()` size, config vocabulary size, added tokens,
  and special token strings/IDs;
- Python, PyTorch, Transformers, Accelerate, Steerling, CUDA, and GPU versions.

No statistical aggregation or judge calls are involved.

## Falsifiable hypotheses and expected results

1. **Custom model class:** the resolved class will be a Steerling custom
   causal-diffusion architecture, not a Llama-family autoregressive class.
2. **Model scale:** parameter count will be about 8.8 billion (within roughly
   5%), with about 16–18 GiB of BF16 parameter storage before runtime overhead.
3. **Backbone geometry:** the model/config will expose approximately 32
   transformer layers, hidden size 4,096, 32 attention heads, and 4 KV heads.
4. **Concept machinery:** the config or module tree will expose known/unknown
   concept decomposition corresponding to roughly 33,732 known and 101,196
   unknown concepts.
5. **Vocabulary:** config and tokenizer vocabulary sizes will be near 100,352;
   any small difference should be attributable to added or special tokens.
6. **Placement:** all material parameters will be CUDA-resident and BF16 after
   loading. Bookkeeping buffers may legitimately differ if documented.
7. **Artifact completeness:** the final JSON will parse and contain non-empty
   model, vocabulary, totals, modules, and environment sections; parameter totals
   computed from the module inventory will agree with the top-level totals.

## Success and failure criteria

The experiment succeeds only when:

- detached submission returns an app ID;
- logs show the pinned model loading;
- at least one persisted progress/checkpoint event is visible after the local
  submitter exits;
- the completed checkpoint and all required artifacts exist on the output
  Volume;
- `model_inspection.json` passes parse and invariant checks; and
- the complete remote artifact directory is pulled locally.

Do not declare success from CLI submission alone. Fail rather than silently
switching model revision, model class, dtype, GPU type, or CPU execution.

## Required artifacts

The remote experiment directory must contain:

- `config.yaml`
- `checkpoint.json`
- `progress.json`
- `dashboard.html`
- `dashboard_history.jsonl`
- `model_inspection.json`
- `results.json`
- `RESULTS.md`
- `stdout.log`

The locally retrieved directory must preserve these files unchanged.

## Duration and cost expectation

- Warm cache: approximately 3–8 minutes
- Cold cache: approximately 8–20 minutes
- Compute: one L40S for the duration, plus negligible CPU and Volume storage

These are estimates. Queueing and first-time image/model downloads may increase
wall-clock time. Abort if runtime exceeds 30 minutes.

## If observations differ from expectations

Preserve the raw artifacts and separate implementation defects from conceptual
explanations. Candidate hypotheses include:

- the checkpoint uses weight tying or aliases that change naive parameter sums;
- concept components are encoded in config/state-dict names rather than distinct
  PyTorch modules;
- tokenizer length differs because of added or reserved special tokens;
- `device_map=auto` offloaded tensors because usable VRAM was lower than assumed;
- a dependency mismatch changed the custom remote-code class or loading path;
- cached files are incomplete or do not correspond to the pinned revision.

Suggested debugging experiments, each requiring its own card and **proceed**:

- audit state-dict keys and tied tensor identities;
- compare config fields against constructed module dimensions;
- inspect tokenizer base, added, reserved, and special tokens separately;
- load on a meta device to inspect construction without weights;
- rerun a minimal deterministic class/config-only smoke test with a clean cache;
- inspect concept-head inputs/outputs on one harmless prompt.

## Explicit approval request

Reply **proceed** only if you approve this exact detached, durable Modal
experiment. Approval authorizes implementing the small Modal wrapper/config and
launching this card once; it does not authorize any follow-up experiment.
