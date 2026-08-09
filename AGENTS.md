# Sterling interpretability workspace

This repository is for small, reproducible interpretability studies of
`guidelabs/steerling-8b-instruct`. The model is a custom causal-diffusion transformer,
not an autoregressive Llama-family model. Prefer its native concept outputs over
generic next-token interpretability assumptions.

## Project layout

- `configs/` contains resolved experiment inputs.
- `experiments/` contains one folder per prepared experiment, including its card,
  resolved config, prompts, and experiment-specific entrypoint.
- `src/sterling_exploration/` contains reusable analysis and artifact helpers.
- `modal_app.py` is the remote entrypoint.
- `runs/` contains pulled, immutable run artifacts; never reuse a run ID.
- `tests/` contains CPU-only unit tests. Tests must not download model weights.

## Environment and model

- Use Modal for model execution; keep the local machine to submission, testing,
  monitoring, and artifact inspection.
- Load `guidelabs/steerling-8b-instruct` from a Hugging Face snapshot pinned to
  an immutable revision. Use the official `steerling` generator for masked-
  diffusion decoding; never substitute autoregressive `transformers.generate`.
- Use BF16 and a GPU with at least 24 GB VRAM. The default is an L40S (48 GB).
- Keep Hugging Face cache and run outputs on separate Modal Volumes.
- Credentials belong in Modal secrets, never in source, configs, or artifacts.
- Treat remote model code as executable third-party code. Inspect changes before
  updating a pinned revision.

## Experiment contract

Before launching compute, present an experiment card covering the objective,
model revision, prompts/data selection, analysis settings, metrics, checkpoint
cadence, artifacts, expected duration/cost, and whether the run is fresh or a
resume. The card must also state falsifiable hypotheses and concrete expected
results, including the direction and rough magnitude or qualitative pattern of
each primary metric. Wait for an explicit user message containing **proceed**
before launch. Approval applies only to the experiment card immediately under
discussion; every materially new or changed experiment needs a new card and a
new **proceed** signal.

Every run ID is `YYYY-MM-DD_HHMMSSZ_short-description`. A run must write:

- `config.yaml` with the fully resolved settings and config fingerprint;
- `checkpoint.json` and `progress.json` at each safe boundary;
- `dashboard.html` and `dashboard_history.jsonl` throughout the run;
- `results.json` and `RESULTS.md` on completion or controlled stop.

Write artifacts atomically where practical and call `Volume.commit()` after
each checkpoint. A resume must use `run_mode: resume`, find an existing stopped
or running checkpoint, and match its configuration fingerprint. Submit detached,
then verify model loading and a persisted progress event before calling a run
active. Pull completed or stopped run folders into `runs/`.

## Analysis standards

- Begin with native known/unknown concept decomposition, reconstruction error,
  component norms, and token-level top-k concept IDs.
- Preserve token IDs and decoded token strings alongside concept results.
- Include unmodified controls before testing interventions.
- Record seeds, tokenizer behavior, masking pattern, diffusion settings, dtype,
  device, dependency versions, and model revision.
- Resolve concept IDs against Guide Labs' authoritative
  `guidelabs/steerling/concept_labels.parquet` catalog and record its immutable
  revision. Treat catalog names/descriptions as model-provider labels and show
  learned top-token alignments alongside them as an empirical cross-check.
- Audit surprising negative results end to end before drawing conclusions.
- Compare final observations to every preregistered hypothesis and expected
  result. When they differ, separate verified implementation defects from
  conceptual explanations, propose plausible alternative hypotheses, and
  suggest targeted debugging experiments (controls, small hand-inspected
  examples, masking/tokenization checks, concept-head checks, intervention
  position/strength sweeps, and deterministic reruns as relevant). Do not launch
  those follow-ups without their own experiment card and explicit **proceed**.
- For jailbreak/ASR evaluation, retain the dataset prompt, reference target,
  complete unmodified model response, judge success flag, judge rationale,
  judge response ID, token usage, retry count, and judge prompt/schema version.

## Development

Run `python -m pytest` for CPU tests and `python -m compileall modal_app.py src`
before committing. Do not commit caches, model weights, secrets, or transient
Modal state. Keep commits small and conventional.
