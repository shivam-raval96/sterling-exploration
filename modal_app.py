from __future__ import annotations

import json
from pathlib import Path

import modal

from sterling_exploration.artifacts import make_run_id
from sterling_exploration.preflight import load_config

APP_NAME = "sterling-interpretability"
REMOTE_ROOT = Path("/outputs/runs")

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.48.0",
        "accelerate>=1.2,<2",
        "safetensors>=0.5,<1",
        "tiktoken>=0.8,<1",
        "pyyaml>=6,<7",
    )
    .add_local_python_source("sterling_exploration")
)
model_cache = modal.Volume.from_name("sterling-model-cache", create_if_missing=True)
outputs = modal.Volume.from_name("sterling-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
    volumes={"/cache": model_cache, "/outputs": outputs},
    secrets=[modal.Secret.from_name("huggingface", required_keys=[])],
)
def inspect_concepts(config: dict, run_id: str) -> dict:
    """Run the native concept decomposition and persist compact JSON results."""
    import os
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sterling_exploration.runner import RunArtifacts, analyze_prompt

    os.environ["HF_HOME"] = "/cache/huggingface"
    artifacts = RunArtifacts(REMOTE_ROOT / run_id, config, outputs)
    start_index = artifacts.initialize_or_resume()
    started = time.monotonic()

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"], trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    artifacts.progress("model_loaded", start_index, len(config["prompts"]), started)

    results = artifacts.load_partial_results()
    try:
        for index in range(start_index, len(config["prompts"])):
            results.append(
                analyze_prompt(
                    model,
                    tokenizer,
                    config["prompts"][index],
                    unknown_topk=int(config.get("unknown_topk", 16)),
                )
            )
            artifacts.checkpoint(index + 1, results, started)
    except BaseException:
        artifacts.stop(len(results), results, started)
        raise

    artifacts.complete(results, started)
    return {"run_id": run_id, "prompts": len(results), "status": "complete"}


@app.local_entrypoint()
def main(config_path: str = "configs/basic-concepts.yaml", dry_run: bool = True):
    config = load_config(Path(config_path))
    from sterling_exploration.artifacts import config_fingerprint

    config["fingerprint"] = config_fingerprint(config)
    run_id = make_run_id(config["description"])
    print(json.dumps({"run_id": run_id, "config": config}, indent=2))
    if dry_run:
        print("Dry run only. Re-run with --no-dry-run after explicit approval containing 'proceed'.")
        return
    result = inspect_concepts.spawn(config, run_id)
    print(json.dumps({"call_id": result.object_id, "run_id": run_id}, indent=2))
