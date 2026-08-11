from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

MODEL_ID = "guidelabs/steerling-8b-instruct"
MODEL_REVISION = "6e5a87d00d45348001810c30fe9bd65110b69fc2"
RUNS = Path("/outputs/exploration/runs")

app = modal.App("sterling-model-inspection")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install(
        "accelerate>=1.2,<2",
        "torch~=2.8.0",
        "transformers>=4.48,<5",
        "steerling @ git+https://github.com/guidelabs/steerling.git@f34ffa89e46969445f3cf6e7c885e9623a2047c1",
    )
)
model_cache = modal.Volume.from_name("sterling-model-cache", create_if_missing=True)
outputs = modal.Volume.from_name("sterling-outputs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


@app.function(
    image=image,
    gpu="L40S",
    timeout=1_800,
    retries=modal.Retries(max_retries=1),
    volumes={"/cache": model_cache, "/outputs": outputs},
    secrets=[hf_secret],
)
def inspect_model(run_id: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "run_id": run_id,
        "run_mode": "fresh",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "gpu": "L40S",
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "checkpoint.json", {"status": "running", "phase": "load"})
    write_json(run_dir / "progress.json", {"phase": "load", "completed": 0, "total": 1})
    outputs.commit()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        cache_dir="/cache/huggingface",
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        cache_dir="/cache/huggingface",
    )
    model.eval()

    structure = str(model)
    print(structure, flush=True)
    parameters = list(model.named_parameters())
    total_parameters = sum(parameter.numel() for _, parameter in parameters)
    result = {
        "status": "complete",
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
        "total_parameters": total_parameters,
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for _, parameter in parameters
        ),
        "module_type_counts": dict(
            sorted(Counter(type(module).__name__ for module in model.modules()).items())
        ),
        "config": model.config.to_dict(),
    }
    (run_dir / "model_structure.txt").write_text(structure + "\n")
    write_json(run_dir / "results.json", result)
    (run_dir / "RESULTS.md").write_text(
        "# Steerling model inspection\n\n"
        f"- Model class: `{result['model_class']}`\n"
        f"- Parameters: {total_parameters:,}\n"
        f"- Parameter storage: {result['parameter_bytes'] / 2**30:.2f} GiB\n"
        f"- Module types: {len(result['module_type_counts'])}\n"
    )
    write_json(run_dir / "checkpoint.json", {"status": "complete", "phase": "complete"})
    write_json(run_dir / "progress.json", {"phase": "complete", "completed": 1, "total": 1})
    model_cache.commit()
    outputs.commit()
    return result


def make_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%SZ")
    return f"{timestamp}_model-structure"


def inspection_spec() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "gpu": "L40S",
        "dtype": "bfloat16",
        "artifacts_root": str(RUNS),
    }


@app.local_entrypoint()
def main(dry_run: bool = True) -> None:
    if dry_run:
        print(json.dumps({"dry_run": True, **inspection_spec()}, indent=2))
        return
    run_id = make_run_id()
    call = inspect_model.spawn(run_id)
    print(json.dumps({"run_id": run_id, "call_id": call.object_id}, indent=2))


if __name__ == "__main__":
    print(json.dumps({"local_validation": True, **inspection_spec()}, indent=2))
