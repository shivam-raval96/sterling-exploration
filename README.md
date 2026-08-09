# Sterling exploration

Small, reproducible interpretability experiments on
[`guidelabs/steerling-8b`](https://huggingface.co/guidelabs/steerling-8b).

Steerling-8B is an 8.4B-parameter causal-diffusion model with a native concept
decomposition. This repository starts with a conservative probe of those native
outputs: token-level top known/unknown concept IDs, component norms, and
reconstruction residuals. It does not assume autoregressive generation or assign
human meanings to unlabeled concept IDs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Modal must already be authenticated (`modal setup`). The public model may not
need a Hugging Face token, but a token can be stored as a Modal secret named
`huggingface` using `.env.example` as the shape.

## Prepare a run

The checked-in configuration pins the model revision that was inspected while
this scaffold was created:

```bash
python -m sterling_exploration.preflight configs/basic-concepts.yaml
```

This preflight is deliberately local: unlike `modal run`, it cannot build an
image or create a remote app.

Actual GPU submission is intentionally gated. First review the experiment card
and config, then explicitly authorize it with **proceed**. The run should be
submitted detached and verified through both logs and the output Volume.
Each card includes hypotheses and expected results; closeout compares them with
observations and proposes hypotheses and debugging experiments for discrepancies.

See [EXPERIMENT_CARD.md](EXPERIMENT_CARD.md) for the card template and
[AGENTS.md](AGENTS.md) for the full experiment and artifact contract.
