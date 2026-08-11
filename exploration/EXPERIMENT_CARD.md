# Modal model-structure inspection

- Objective: directly load pinned `guidelabs/steerling-8b-instruct` on Modal and
  print/persist its PyTorch module structure, parameter count, module-type counts,
  tokenizer/model classes, configuration, dtype, and parameter storage size.
- Model revision: `6e5a87d00d45348001810c30fe9bd65110b69fc2`.
- Method: one fresh, detached Modal function on an L40S; BF16; no prompt,
  generation, dataset, judge, intervention, or model mutation.
- Artifacts: config, checkpoint, progress, `model_structure.txt`, `results.json`,
  and `RESULTS.md` on the persistent `sterling-outputs` Volume.
- Expected duration: 2–8 minutes with a warm cache, or 5–15 minutes on a cold
  18 GB model download. Expected cost: a few minutes of one L40S.

## Hypotheses and expected results

1. The model will load as the custom Steerling causal-diffusion class in BF16.
2. The parameter count will be approximately 8.8 billion and parameter storage
   approximately 16–18 GiB.
3. The module tree will contain 32 transformer layers plus known/unknown concept
   decomposition modules. Missing concept modules would disconfirm the expected
   architecture.

If loading or structure differs, inspect the pinned remote-code class, resolved
config, weight keys, dependency versions, and device map. Follow-up experiments
require a new card and `proceed`.

## Launch gate

Prepared only. Do not set `PROCEED=True` or submit Modal compute until the user
approves this exact card with a fresh **proceed**.
