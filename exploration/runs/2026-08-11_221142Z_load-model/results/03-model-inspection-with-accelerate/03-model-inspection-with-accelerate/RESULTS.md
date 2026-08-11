# Model inspection results

- Model class: `transformers_modules.guidelabs.steerling_hyphen_8b_hyphen_instruct.6e5a87d00d45348001810c30fe9bd65110b69fc2.modeling_steerling.SteerlingForCausalLM`
- Parameters: `8,392,069,120`
- Parameter storage: `15.631 GiB`
- Modules: `400`
- Tokenizer length: `12`
- Hypotheses passed: `12/13`

## Hypothesis checks

- PASS — `custom_model_class`
- PASS — `parameter_count_within_5_percent`
- PASS — `layers_match`
- PASS — `hidden_size_matches`
- PASS — `attention_heads_match`
- PASS — `kv_heads_match`
- PASS — `known_concepts_match`
- PASS — `unknown_concepts_match`
- PASS — `config_vocab_matches`
- FAIL — `tokenizer_vocab_near_expected`
- PASS — `all_parameters_cuda`
- PASS — `all_parameters_bfloat16`
- PASS — `nonempty_modules`

## Suggested debugging experiments

Prepare separate cards for state-dict/tied-weight auditing, tokenizer
component accounting, clean-cache loading, or config-versus-module checks.
Do not launch them without a new explicit proceed signal.
