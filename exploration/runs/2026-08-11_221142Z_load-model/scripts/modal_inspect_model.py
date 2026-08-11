from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import modal
import yaml

APP_NAME = "sterling-load-model-inspection"
CONFIG_LOCAL = Path(
    "exploration/runs/2026-08-11_221142Z_load-model/config.yaml"
)
INSPECTOR_LOCAL = Path(
    "exploration/runs/2026-08-11_221142Z_load-model/scripts/inspect_model.py"
)
CONFIG_REMOTE = Path("/root/config.yaml")
INSPECTOR_REMOTE = Path("/root/inspect_model.py")
REMOTE_RUN_DIR = Path(
    "/mnt/run-output/exploration/runs/2026-08-11_221142Z_load-model/"
    "03-model-inspection-with-accelerate"
)

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("steerling==0.2.0", "accelerate==1.14.0", "pyyaml==6.0.2")
    .add_local_file(str(CONFIG_LOCAL), str(CONFIG_REMOTE), copy=True)
    .add_local_file(str(INSPECTOR_LOCAL), str(INSPECTOR_REMOTE), copy=True)
)
model_cache = modal.Volume.from_name("sterling-model-cache", create_if_missing=True)
outputs = modal.Volume.from_name("sterling-exploration-runs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


def _canonical_fingerprint(config: dict[str, Any]) -> str:
    value = {key: item for key, item in config.items() if key != "fingerprint"}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        stream.write(value)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_history(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "dashboard_history.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _render_dashboard(history: list[dict[str, Any]]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['timestamp']))}</td>"
        f"<td>{html.escape(str(row['phase']))}</td>"
        f"<td>{html.escape(str(row['status']))}</td>"
        f"<td>{html.escape(str(row.get('detail', '')))}</td>"
        "</tr>"
        for row in history
    )
    return (
        "<!doctype html><meta charset='utf-8'><title>Model inspection</title>"
        "<style>body{font:15px system-ui;margin:2rem;max-width:1000px}"
        "table{border-collapse:collapse;width:100%}th,td{padding:.6rem;"
        "border:1px solid #ccc;text-align:left}th{background:#eee}</style>"
        "<h1>Steerling model inspection</h1><table><thead><tr>"
        "<th>Timestamp</th><th>Phase</th><th>Status</th><th>Detail</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _persist_phase(
    run_dir: Path,
    config: dict[str, Any],
    *,
    phase: str,
    status: str,
    next_phase: str,
    started: float,
    detail: str = "",
) -> None:
    event = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": config["run_id"],
        "experiment": config["experiment"],
        "phase": phase,
        "status": status,
        "next_phase": next_phase,
        "detail": detail,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "fingerprint": config["fingerprint"],
    }
    _atomic_json(run_dir / "checkpoint.json", event)
    _atomic_json(run_dir / "progress.json", event)
    history_path = run_dir / "dashboard_history.jsonl"
    with history_path.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    _atomic_text(run_dir / "dashboard.html", _render_dashboard(_read_history(run_dir)))
    print("PROGRESS " + json.dumps(event, sort_keys=True), flush=True)
    outputs.commit()


def _load_inspector() -> Any:
    spec = importlib.util.spec_from_file_location("inspect_model", INSPECTOR_REMOTE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load inspection module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hypothesis_checks(result: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    config = result["model"]["config"]
    totals = result["totals"]
    vocabulary = result["vocabulary"]
    parameter_count = totals["parameters"]
    target = expected["parameter_count"]
    tolerance = expected["parameter_tolerance_fraction"]
    model_class = result["model"]["class"]
    checks = {
        "custom_model_class": "steerling" in model_class.lower() and "llama" not in model_class.lower(),
        "parameter_count_within_5_percent": abs(parameter_count - target) <= target * tolerance,
        "layers_match": config.get("n_layers") == expected["layers"],
        "hidden_size_matches": config.get("n_embd") == expected["hidden_size"],
        "attention_heads_match": config.get("n_head") == expected["attention_heads"],
        "kv_heads_match": config.get("n_kv_heads") == expected["kv_heads"],
        "known_concepts_match": config.get("n_concepts") == expected["known_concepts"],
        "unknown_concepts_match": config.get("n_unknown_concepts") == expected["unknown_concepts"],
        "config_vocab_matches": config.get("vocab_size") == expected["vocab_size"],
        "tokenizer_vocab_near_expected": abs(vocabulary["tokenizer_length"] - expected["vocab_size"]) <= 32,
        "all_parameters_cuda": all(device.startswith("cuda") for device in totals["parameter_device_counts"]),
        "all_parameters_bfloat16": set(totals["parameter_dtype_counts"]) == {"torch.bfloat16"},
        "nonempty_modules": bool(result["modules"]),
    }
    return {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_passed": all(checks.values()),
    }


def _results_markdown(result: dict[str, Any], validation: dict[str, Any]) -> str:
    totals = result["totals"]
    vocab = result["vocabulary"]
    lines = [
        "# Model inspection results",
        "",
        f"- Model class: `{result['model']['class']}`",
        f"- Parameters: `{totals['parameters']:,}`",
        f"- Parameter storage: `{totals['parameter_storage_bytes'] / 2**30:.3f} GiB`",
        f"- Modules: `{totals['module_count']:,}`",
        f"- Tokenizer length: `{vocab['tokenizer_length']:,}`",
        f"- Hypotheses passed: `{validation['passed']}/{validation['total']}`",
        "",
        "## Hypothesis checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`"
        for name, passed in validation["checks"].items()
    )
    lines.append("")
    if not validation["all_passed"]:
        lines.extend(
            [
                "## Suggested debugging experiments",
                "",
                "Prepare separate cards for state-dict/tied-weight auditing, tokenizer",
                "component accounting, clean-cache loading, or config-versus-module checks.",
                "Do not launch them without a new explicit proceed signal.",
                "",
            ]
        )
    return "\n".join(lines)


@app.function(
    image=image,
    gpu="L40S",
    timeout=1800,
    retries=modal.Retries(max_retries=0),
    volumes={"/mnt/model-cache": model_cache, "/mnt/run-output": outputs},
    secrets=[hf_secret],
)
def inspect_remote(run_mode: str = "fresh") -> dict[str, Any]:
    import accelerate
    import torch
    from transformers import AutoModel, AutoTokenizer

    started = time.monotonic()
    config = yaml.safe_load(CONFIG_REMOTE.read_text())
    config["run_mode"] = run_mode
    config["fingerprint"] = _canonical_fingerprint(config)
    run_dir = REMOTE_RUN_DIR
    checkpoint_path = run_dir / "checkpoint.json"

    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if run_mode != "resume":
            raise ValueError("existing checkpoint requires explicit run_mode: resume")
        if checkpoint.get("fingerprint") != config["fingerprint"]:
            raise ValueError("resume config fingerprint mismatch")
        if checkpoint.get("status") not in {"running", "stopped"}:
            raise ValueError(f"cannot resume checkpoint status {checkpoint.get('status')}")
    elif run_mode != "fresh":
        raise FileNotFoundError("resume requested without checkpoint.json")

    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(run_dir / "config.yaml", yaml.safe_dump(config, sort_keys=True))
    _persist_phase(
        run_dir,
        config,
        phase="initialize",
        status="running",
        next_phase="load_tokenizer",
        started=started,
    )

    def handle_term(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt("SIGTERM received")

    signal.signal(signal.SIGTERM, handle_term)
    os.environ["HF_HOME"] = "/mnt/model-cache/huggingface"
    inspector = _load_inspector()
    common = {
        "revision": config["model"]["revision"],
        "trust_remote_code": True,
        "cache_dir": "/mnt/model-cache/huggingface/hub",
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"], **common)
        _persist_phase(
            run_dir,
            config,
            phase="tokenizer_loaded",
            status="running",
            next_phase="load_model",
            started=started,
            detail=type(tokenizer).__name__,
        )
        model = AutoModel.from_pretrained(
            config["model"]["id"],
            dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
            **common,
        ).eval()
        _persist_phase(
            run_dir,
            config,
            phase="model_loaded",
            status="running",
            next_phase="inspect",
            started=started,
            detail=type(model).__name__,
        )

        printed = str(model)
        print(printed, flush=True)
        _atomic_text(run_dir / "stdout.log", printed + "\n")
        result = inspector.inspect(model, tokenizer)
        result["environment"].update(
            {
                "accelerate": accelerate.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            }
        )
        _persist_phase(
            run_dir,
            config,
            phase="inspected",
            status="running",
            next_phase="validate",
            started=started,
            detail=f"{result['totals']['parameters']} parameters",
        )

        inspection_path = run_dir / "model_inspection.json"
        _atomic_json(inspection_path, result)
        reparsed = json.loads(inspection_path.read_text())
        required = {"model", "vocabulary", "totals", "modules", "environment"}
        if not required.issubset(reparsed) or not reparsed["modules"]:
            raise ValueError("inspection JSON failed required-key validation")
        validation = _hypothesis_checks(reparsed, config["expected"])
        summary = {
            "run_id": config["run_id"],
            "experiment": config["experiment"],
            "status": "complete",
            "fingerprint": config["fingerprint"],
            "validation": validation,
            "totals": reparsed["totals"],
            "vocabulary": reparsed["vocabulary"],
            "model_class": reparsed["model"]["class"],
        }
        _atomic_json(run_dir / "results.json", summary)
        _atomic_text(run_dir / "RESULTS.md", _results_markdown(reparsed, validation))
        _persist_phase(
            run_dir,
            config,
            phase="complete",
            status="complete",
            next_phase="none",
            started=started,
            detail=f"{validation['passed']}/{validation['total']} checks passed",
        )
        return summary
    except KeyboardInterrupt as exc:
        _persist_phase(
            run_dir,
            config,
            phase="stopped",
            status="stopped",
            next_phase="load_tokenizer",
            started=started,
            detail=str(exc),
        )
        raise
    except BaseException as exc:
        _persist_phase(
            run_dir,
            config,
            phase="failed",
            status="stopped",
            next_phase="load_tokenizer",
            started=started,
            detail=f"{type(exc).__name__}: {exc}",
        )
        raise


@app.local_entrypoint()
def main(run_mode: str = "fresh") -> None:
    if run_mode not in {"fresh", "resume"}:
        raise ValueError("run_mode must be fresh or resume")
    call = inspect_remote.spawn(run_mode)
    print(json.dumps({"app": APP_NAME, "call_id": call.object_id, "run_mode": run_mode}))
