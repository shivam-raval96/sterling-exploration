# Load-model structure inspection

## Identity and objective

- Run ID: `2026-08-11_221142Z_load-model`
- Run mode: fresh
- Execution surface: Modal Notebook with an L40S GPU
- Objective: directly load the pinned Steerling instruction model, print its
  complete PyTorch module representation, and persist machine-readable details
  about every module, parameter, buffer, model setting, and tokenizer vocabulary.
- Script: `scripts/inspect_model.py`
- Output: `results/model_inspection.json`

This is architecture inspection only. It does not generate text, evaluate a
dataset, invoke a judge, steer concepts, or modify weights.

## Model and environment

- Model: `guidelabs/steerling-8b-instruct`
- Immutable revision: `6e5a87d00d45348001810c30fe9bd65110b69fc2`
- Loading API: `AutoModel.from_pretrained` and
  `AutoTokenizer.from_pretrained`, both with `trust_remote_code=True`
- Dtype: BF16
- Device placement: `device_map=auto`
- GPU: one Modal L40S (48 GB)
- Cache: attach `sterling-model-cache` to the Modal Notebook and pass its mount
  path with `--cache-dir`
- Code safety: the pinned remote model code is treated as executable third-party
  code; changing the revision requires a new review and card.

## Procedure

1. Open the existing Modal Notebook and select one L40S GPU.
2. Attach the `sterling-model-cache` Volume and, if needed, the existing
   Hugging Face secret.
3. Upload or synchronize this run directory into the notebook filesystem.
4. Install the pinned Steerling and Transformers dependencies without changing
   the model revision.
5. Execute:

   ```bash
   python scripts/inspect_model.py \
     --cache-dir /mnt/sterling-model-cache \
     --output results/model_inspection.json
   ```

6. Confirm that `print(model)` appears in the cell output and that the JSON file
   parses successfully.
7. Download/synchronize the JSON into this run's local `results/` directory and
   update `RUN.md` with observations and hypothesis comparisons.

## Measurements and artifacts

The JSON will contain:

- full `print(model)` text;
- model and tokenizer qualified class names;
- resolved model configuration;
- every module's path, depth, type, direct parameters, and direct buffers;
- every parameter/buffer shape, dimensionality, element count, dtype, device,
  element size, storage size, and trainability;
- total/trainable parameter counts, total storage, buffer count, module count,
  module-type counts, dtype counts, and device counts;
- tokenizer length, vocabulary sizes, added vocabulary, and special tokens/IDs;
- Python and platform information.

Primary artifact: `results/model_inspection.json`. The notebook cell output is
diagnostic and is not the source of truth.

## Hypotheses and expected results

1. **Custom architecture:** the loaded class will be the custom Steerling
   causal-diffusion model rather than a Llama-family autoregressive class.
2. **Scale:** total parameters will be approximately 8.8 billion. BF16 parameter
   storage should be approximately 16–18 GiB. A material deviation (more than
   roughly 5%) would suggest weight sharing/counting behavior, missing weights,
   or an unexpected checkpoint.
3. **Transformer structure:** the configuration and module tree will expose 32
   transformer layers, hidden size 4,096, 32 attention heads, and 4 KV heads.
4. **Interpretability structure:** the configuration/module tree will expose
   known and unknown concept decomposition machinery, with approximately 33,732
   known concepts and 101,196 unknown concepts. Missing concept modules would
   disconfirm the expected interpretable architecture.
5. **Vocabulary:** tokenizer and config vocabulary sizes should be close to
   100,352 tokens. Small differences may come from special/added tokens and must
   be explained rather than treated as an error automatically.
6. **Placement:** all loaded parameters should resolve to CUDA in BF16, except
   any deliberately non-persistent or bookkeeping buffers.

## Checkpoints, duration, and cost

- Safe boundary 1: model and tokenizer load succeeds.
- Safe boundary 2: structure collection completes in memory.
- Safe boundary 3: JSON is atomically written and parsed.
- Expected duration: about 2–8 minutes with a warm cache or 5–15 minutes with a
  cold model download.
- Expected cost: several minutes of one L40S plus negligible CPU/storage cost.
- Failure policy: do not silently fall back to CPU or a different model/revision.

## Discrepancy and debugging plan

If results differ from expectations, preserve the raw JSON and investigate:

- resolved revision, remote-code class, and `config.to_dict()`;
- missing/unexpected weight keys and weight-sharing aliases;
- tokenizer length versus `get_vocab()`, config vocabulary, added vocabulary,
  and special-token IDs;
- direct versus recursively counted parameters and duplicated/tied tensors;
- dtype/device placement after `device_map=auto`;
- whether concept heads are implemented under unexpected module names;
- dependency-version incompatibilities or partial cache contents.

Suggested follow-ups include a state-dict key audit, tied-weight identity check,
config-versus-module comparison, or CPU/meta-device construction. Each material
follow-up requires its own experiment card and explicit `proceed`.

## Launch gate

This card prepares the inspection only. Do not start the Modal Notebook GPU or
execute the loading script until the user gives an explicit **proceed** for this
exact card.
