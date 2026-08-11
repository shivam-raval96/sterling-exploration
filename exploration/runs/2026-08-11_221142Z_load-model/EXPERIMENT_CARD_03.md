# Experiment card 03: model inspection with Accelerate

## Status and launch gate

- Run ID: `2026-08-11_221142Z_load-model`
- Experiment within run: `03-model-inspection-with-accelerate`
- Run mode: `fresh`
- Status: prepared, not launched
- Approval: requires a new explicit **proceed** for this exact card

This is a narrow correction to `EXPERIMENT_CARD_02.md`. All objectives, model
inputs, hypotheses, measurements, GPU, timeout, safety rules, checkpoint cadence,
required artifacts, success criteria, duration/cost estimates, and discrepancy
handling remain unchanged except where explicitly stated below.

## Verified reason for revision

Experiment 02 stopped safely after its initialization checkpoint. Tokenizer
loading triggered this verified error before any model weights loaded:

> Using a `device_map` requires `accelerate`.

The stopped remote artifacts were preserved locally under
`results/02-durable-model-structure-inspection-stopped/`. This is an
implementation/dependency defect, not evidence against any model hypothesis.

## Exact changes from experiment 02

1. Add the exact image dependency `accelerate==1.14.0`.
2. Record that pin in the resolved `config.yaml` and therefore generate a new
   SHA-256 configuration fingerprint.
3. Use a new remote artifact directory so the stopped experiment remains
   immutable:

   `/mnt/run-output/exploration/runs/2026-08-11_221142Z_load-model/03-model-inspection-with-accelerate`

4. Keep `run_mode: fresh`. Do not resume or overwrite experiment 02 because its
   fingerprint does not contain the new dependency.
5. Add a pre-load check that imports `accelerate` and records its version in the
   environment metadata before constructing the model.

## Expected result of the correction

The tokenizer and `device_map="cuda"` model-loading phases should pass the point
where experiment 02 stopped. Once model loading succeeds, the original 13
hypothesis checks from experiment 02 apply unchanged. The expected overall
runtime remains approximately 3–8 minutes with a warm cache or 8–20 minutes with
a cold cache.

## Failure and debugging policy

If the same missing-Accelerate error recurs, treat it as an image construction or
import-path defect and inspect the built image's installed distributions before
another run. If a different dependency error appears, preserve and classify it;
do not silently alter another dependency pin. Any material dependency or loading
change requires another card and explicit **proceed**.

## Explicit approval request

Reply **proceed** only if you approve this exact fresh corrected experiment.
Approval authorizes adding the pinned dependency/check and launching experiment
03 once. It does not authorize further dependency changes or follow-up runs.
