from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, config_fingerprint


def _norm(tensor: Any) -> float:
    return float(tensor.float().norm(dim=-1).mean().item())


def analyze_prompt(model: Any, tokenizer: Any, prompt: str, unknown_topk: int) -> dict[str, Any]:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    with torch.inference_mode():
        _, decomposition = model(
            input_ids=input_ids, minimal_output=False, unknown_topk=unknown_topk
        )

    known_values, known_ids = decomposition.known_logits.float().topk(
        k=min(16, decomposition.known_logits.shape[-1]), dim=-1
    )
    residual = decomposition.hidden.float() - decomposition.composed.float()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    rows = []
    for position, token in enumerate(tokens):
        row = {
            "position": position,
            "token_id": int(input_ids[0, position]),
            "token": token,
            "known_concept_ids": known_ids[0, position].tolist(),
            "known_concept_logits": known_values[0, position].tolist(),
        }
        if decomposition.unknown_topk_indices is not None:
            row["unknown_concept_ids"] = decomposition.unknown_topk_indices[
                0, position, :unknown_topk
            ].tolist()
            row["unknown_concept_logits"] = decomposition.unknown_topk_logits[
                0, position, :unknown_topk
            ].float().tolist()
        rows.append(row)

    return {
        "prompt": prompt,
        "tokens": rows,
        "mean_norms": {
            "hidden": _norm(decomposition.hidden),
            "known": _norm(decomposition.known_features),
            "unknown": _norm(decomposition.unk_hat),
            "epsilon": _norm(decomposition.epsilon),
            "reconstruction_residual": _norm(residual),
        },
    }


class RunArtifacts:
    def __init__(self, root: Path, config: dict[str, Any], volume: Any):
        self.root = root
        self.config = config
        self.volume = volume
        self.fingerprint = config_fingerprint(config)

    def initialize_or_resume(self) -> int:
        self.root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.root / "checkpoint.json"
        if self.config["run_mode"] == "fresh":
            if checkpoint_path.exists():
                raise FileExistsError(f"fresh run already exists: {self.root.name}")
            resolved = dict(self.config, fingerprint=self.fingerprint, run_id=self.root.name)
            (self.root / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True))
            self._write_dashboard({"phase": "initializing", "completed": 0, "total": len(self.config["prompts"])})
            self._commit()
            return 0
        if not checkpoint_path.exists():
            raise FileNotFoundError("resume requested without checkpoint.json")
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint["fingerprint"] != self.fingerprint:
            raise ValueError("resume config fingerprint mismatch")
        if checkpoint["status"] not in {"running", "stopped"}:
            raise ValueError(f"cannot resume checkpoint with status {checkpoint['status']}")
        return int(checkpoint["next_step"])

    def load_partial_results(self) -> list[dict[str, Any]]:
        path = self.root / "partial_results.json"
        return json.loads(path.read_text()) if path.exists() else []

    def progress(self, phase: str, completed: int, total: int, started: float) -> None:
        elapsed = max(time.monotonic() - started, 0.0)
        payload = {
            "phase": phase,
            "completed": completed,
            "total": total,
            "elapsed_seconds": elapsed,
            "throughput_per_second": completed / elapsed if elapsed else 0.0,
            "eta_seconds": ((total - completed) * elapsed / completed) if completed else None,
            "fingerprint": self.fingerprint,
            "run_id": self.root.name,
            "errors": 0,
        }
        atomic_json(self.root / "progress.json", payload)
        with (self.root / "dashboard_history.jsonl").open("a") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
        self._write_dashboard(payload)
        print("PROGRESS " + json.dumps(payload, sort_keys=True), flush=True)
        self._commit()

    def checkpoint(self, next_step: int, results: list[dict[str, Any]], started: float) -> None:
        atomic_json(self.root / "partial_results.json", results)
        atomic_json(
            self.root / "checkpoint.json",
            {"status": "running", "next_step": next_step, "fingerprint": self.fingerprint},
        )
        self.progress("analyzing", next_step, len(self.config["prompts"]), started)

    def stop(self, next_step: int, results: list[dict[str, Any]], started: float) -> None:
        atomic_json(self.root / "partial_results.json", results)
        atomic_json(
            self.root / "checkpoint.json",
            {"status": "stopped", "next_step": next_step, "fingerprint": self.fingerprint},
        )
        self.progress("stopped", next_step, len(self.config["prompts"]), started)

    def complete(self, results: list[dict[str, Any]], started: float) -> None:
        atomic_json(self.root / "results.json", results)
        atomic_json(
            self.root / "checkpoint.json",
            {"status": "complete", "next_step": len(results), "fingerprint": self.fingerprint},
        )
        (self.root / "RESULTS.md").write_text(
            f"# Results\n\nAnalyzed {len(results)} prompts with native concept decomposition.\n"
        )
        self.progress("complete", len(results), len(self.config["prompts"]), started)

    def _write_dashboard(self, payload: dict[str, Any]) -> None:
        pretty = html.escape(json.dumps(payload, indent=2, sort_keys=True))
        document = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="10">
<title>{html.escape(self.root.name)}</title>
<style>body{{font:16px system-ui;max-width:900px;margin:3rem auto;padding:0 1rem}}pre{{background:#111;color:#eee;padding:1rem;overflow:auto}}</style>
</head><body><h1>{html.escape(self.root.name)}</h1><pre>{pretty}</pre></body></html>"""
        (self.root / "dashboard.html").write_text(document)

    def _commit(self) -> None:
        self.volume.commit()
