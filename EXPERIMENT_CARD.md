# Experiment card template

## Identity

- Name and run type: fresh, resume, or comparison
- Objective
- Model ID and immutable revision
- Source config/artifact and config fingerprint

## Hypotheses and expected results

- Falsifiable primary hypothesis
- Expected direction, rough magnitude, or qualitative pattern
- Null/alternative hypotheses
- What result would count as disconfirming evidence

## Inputs and method

- Dataset/prompts, split, selection procedure, and sample size
- Tokenization, masking, and diffusion/generation settings
- Analysis/intervention settings, positions, layers, strengths, and seeds
- Baseline and positive/negative controls
- Material differences from the comparison run, if any

## Evaluation and operations

- Primary and secondary metrics
- Output files, raw artifacts, and plots
- Checkpoint/progress cadence and dashboard URLs
- GPU, timeout, retries, expected duration, and estimated cost
- Fresh/resume behavior and interruption-test status

## Launch gate

Do not submit locally or remotely until the user gives an explicit **proceed**
for this exact card.

## Closeout

- Compare each result with its registered expectation
- Report confirmed and disconfirmed hypotheses
- If results differ, distinguish verified bugs from conceptual explanations
- Suggest alternative hypotheses and targeted debugging experiments
- Require a new card and **proceed** before any follow-up run
