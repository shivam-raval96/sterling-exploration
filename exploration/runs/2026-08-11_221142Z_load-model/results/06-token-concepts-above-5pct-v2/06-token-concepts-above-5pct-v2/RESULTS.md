# Bilingual token concept viewer

- Pairs: 24
- Conversations: 48
- User-content tokens: 1318
- Known concepts saved: 39710
- Unknown concepts saved: 168704
- Activation threshold: 0.05

## Distribution

- Total qualifying concept activations: 208,414
- Known activations: 39,710
- Unknown activations: 168,704
- Unique concepts observed: 3,098
- Concepts per token: minimum 135, median 160, mean 158.129, maximum 160
- Threshold violations: 0

The browser artifact normalizes concept metadata into
`concept_catalog.json` and loads compact `pair_data/pair_*.json` files on
demand. The original checkpoint chunks and consolidated raw JSONL remain local
and on the Modal Volume but are excluded from Git because they are reproducible
and 128 MB uncompressed.
