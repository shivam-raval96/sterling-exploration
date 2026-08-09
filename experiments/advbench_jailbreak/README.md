# AdvBench jailbreak baseline

Prepared experiment for direct-request harmful-compliance ASR and native concept
firing on `guidelabs/steerling-8b-instruct`.

Completed runs include `generations.html`, where every prompt and response is
colored red for a judge-labeled jailbreak, green for no jailbreak, or gray when
generation/judging failed.

Generation is checkpointed in five-prompt work units while reusing one model;
individual diffusion decodes remain independent for correctness. Judge calls are
processed ten at a time with concurrency eight and written in dataset order.

Validate locally without contacting Modal:

```bash
python -m sterling_exploration.preflight experiments/advbench_jailbreak/config.yaml
```

Review `EXPERIMENT_CARD.md`. Do not invoke `modal run` until the user explicitly
authorizes this exact card with **proceed**.
