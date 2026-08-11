# Compiled load-model results

## Outcome

Experiment 03 completed successfully on one Modal L40S. The detached run wrote
and committed every required artifact, and the full remote directory was pulled
into `results/03-model-inspection-with-accelerate/`.

- Modal app: `ap-NRj1LkKdKwpGcBu8lRhMm0`
- Function call: `fc-01KZSG5VTFE15NVDRPTZVQJAQF`
- Configuration fingerprint:
  `9dc541bb0c25f0469fbc470a227886758691a57325504fb1df507b8c05bca94c`
- Remote measured runtime: 16.437 seconds
- Hypothesis checks produced by the run: 12/13 passed

## Model structure

- Resolved class:
  `transformers_modules.guidelabs.steerling_hyphen_8b_hyphen_instruct.6e5a87d00d45348001810c30fe9bd65110b69fc2.modeling_steerling.SteerlingForCausalLM`
- Parameters: 8,392,069,120, all BF16 on `cuda:0`
- Parameter storage: 16.784 GB, or 15.631 GiB
- PyTorch modules: 400
- Transformer layers: 32
- Hidden size: 4,096
- Attention heads / KV heads: 32 / 4
- Config vocabulary: 100,352
- Known / unknown concepts: 33,732 / 101,196
- Concept dimension: 4,096
- Unknown concepts are factorized at rank 256
- Concept injection layer: 16

The module inventory includes 32 `CausalDiffusionBlock` modules, 32
`BlockCausalAttention` modules, 32 MLPs, two `ConceptHead` modules, two
`ConceptPooling` modules, and three embeddings. The token embedding and LM head
each have shape `[100352, 4096]`. Known-concept tensors are padded to 33,744 rows
even though the semantic concept count in config is 33,732.

## Hypothesis comparison

1. **Custom architecture — supported.** The class is
   `SteerlingForCausalLM`, not a Llama-family class.
2. **Scale — broadly supported.** The 8.392B count is 4.64% below the 8.8B
   estimate and within the preregistered 5% tolerance. The apparent storage
   miss is a units issue: 16.784 decimal GB equals 15.631 GiB.
3. **Backbone geometry — supported.** All four preregistered dimensions match.
4. **Concept machinery — supported.** The expected concept counts are in config
   and distinct known/unknown concept heads are present.
5. **Vocabulary — implementation-dependent result.** The model/config vocabulary
   is 100,352, but `len(tokenizer)` and `get_vocab()` both return 12.
6. **Placement — supported.** All 264 named parameter tensors are BF16 on
   `cuda:0`.
7. **Artifact completeness — supported.** The inspection JSON parses and contains
   all required non-empty sections.

## Tokenizer discrepancy

The failed tokenizer check does not imply a 12-token language model. Inspection
of the pinned provider tokenizer code shows that:

- encoding uses the full `cl100k_base` tiktoken mergeable-rank vocabulary;
- `SteerlingTokenizer.get_vocab()` intentionally returns only tiktoken's special
  token dictionary, which has 12 entries here;
- the core tokenizer's vocabulary property is based on the tiktoken vocabulary
  plus Steerling special tokens; and
- the model config/embedding table is padded to 100,352 rows.

Therefore the generic Hugging Face `len(tokenizer)` / `get_vocab()` APIs are not
valid estimators of this custom tokenizer's effective token-ID space. The
original measurement assumption was wrong; the model and tokenizer are not
shown to be corrupt.

## Suggested follow-up experiments

Do not launch these without a new experiment card and explicit **proceed**:

- a tokenizer-accounting audit comparing `_core.vocab_size`, tiktoken
  `n_vocab`, mergeable ranks, special-token IDs, config padding, and round trips;
- a tied-weight/alias audit explaining the 8.392B total and whether token
  embeddings and the LM head share storage or only have equal shapes;
- a one-prompt forward-pass probe of known/unknown concept output shapes and
  top-k IDs.
