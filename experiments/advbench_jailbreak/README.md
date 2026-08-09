# AdvBench jailbreak baseline

Prepared experiment for direct-request harmful-compliance ASR and native concept
firing on `guidelabs/steerling-8b-instruct`.

Validate locally without contacting Modal:

```bash
python -m sterling_exploration.preflight experiments/advbench_jailbreak/config.yaml
```

Review `EXPERIMENT_CARD.md`. Do not invoke `modal run` until the user explicitly
authorizes this exact card with **proceed**.
