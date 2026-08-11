from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Any

import modal
import yaml

APP_NAME = "sterling-language-layerwise-pca"
CONFIG_LOCAL = Path(
    "exploration/runs/2026-08-11_221142Z_load-model/config_language_layerwise_pca.yaml"
)
CONFIG_REMOTE = Path("/root/config_language_layerwise_pca.yaml")
REMOTE_RUN_DIR = Path(
    "/mnt/run-output/exploration/runs/2026-08-11_221142Z_load-model/"
    "04-english-french-layerwise-pca"
)

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "steerling==0.2.0",
        "accelerate==1.14.0",
        "datasets==4.0.0",
        "langid==1.1.6",
        "matplotlib==3.10.5",
        "scikit-learn==1.7.1",
        "pyyaml==6.0.2",
    )
    .add_local_file(str(CONFIG_LOCAL), str(CONFIG_REMOTE), copy=True)
)
model_cache = modal.Volume.from_name("sterling-model-cache", create_if_missing=True)
outputs = modal.Volume.from_name("sterling-exploration-runs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        stream.write(value)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def fingerprint(config: dict[str, Any]) -> str:
    clean = {key: value for key, value in config.items() if key != "fingerprint"}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def chat_prompt_ids(tokenizer: Any, text: str) -> tuple[list[int], list[int]]:
    prefix = (
        [tokenizer.start_header_id]
        + tokenizer.encode("user", add_special_tokens=False)
        + [tokenizer.end_header_id]
        + tokenizer.encode("\n\n", add_special_tokens=False)
    )
    content = tokenizer.encode(text, add_special_tokens=False)
    suffix = (
        [tokenizer.endofchunk_token_id, tokenizer.eot_id, tokenizer.start_header_id]
        + tokenizer.encode("assistant", add_special_tokens=False)
        + [tokenizer.end_header_id]
        + tokenizer.encode("\n\n", add_special_tokens=False)
    )
    return prefix + content + suffix, list(range(len(prefix), len(prefix) + len(content)))


def render_plot(
    coordinates: Any,
    labels: list[str],
    pair_ids: list[int],
    explained: list[float],
    output_png: Path,
    plot_config: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows, columns = plot_config["rows"], plot_config["columns"]
    figure, axes = plt.subplots(rows, columns, figsize=(24, 12), constrained_layout=True)
    english = [index for index, label in enumerate(labels) if label == "English"]
    french = [index for index, label in enumerate(labels) if label == "French"]
    paired_indices: dict[int, dict[str, int]] = {}
    for index, (pair_id, language) in enumerate(zip(pair_ids, labels, strict=True)):
        paired_indices.setdefault(int(pair_id), {})[language] = index
    for layer, axis in enumerate(axes.flat):
        layer_xy = coordinates[layer]
        for pair in paired_indices.values():
            if "English" not in pair or "French" not in pair:
                continue
            english_index, french_index = pair["English"], pair["French"]
            axis.plot(
                [layer_xy[english_index, 0], layer_xy[french_index, 0]],
                [layer_xy[english_index, 1], layer_xy[french_index, 1]],
                color=plot_config["pair_line_color"],
                alpha=plot_config["pair_line_alpha"],
                linewidth=plot_config["pair_line_width"],
                zorder=1,
            )
        axis.scatter(
            layer_xy[english, 0],
            layer_xy[english, 1],
            s=plot_config["marker_size"],
            c=plot_config["english_color"],
            alpha=plot_config["alpha"],
            edgecolors=plot_config["edge_color"],
            linewidths=plot_config["edge_linewidth"],
            zorder=2,
        )
        axis.scatter(
            layer_xy[french, 0],
            layer_xy[french, 1],
            s=plot_config["marker_size"],
            c=plot_config["french_color"],
            alpha=plot_config["alpha"],
            edgecolors=plot_config["edge_color"],
            linewidths=plot_config["edge_linewidth"],
            zorder=2,
        )
        axis.set_title(f"Layer {layer + 1}")
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_xticks([])
        axis.set_yticks([])
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=plot_config["english_color"], markeredgecolor="black", markeredgewidth=0.3, alpha=0.7, label="English"),
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=plot_config["french_color"], markeredgecolor="black", markeredgewidth=0.3, alpha=0.7, label="French"),
            Line2D([], [], linestyle="", label=f"2D EV: {explained[layer] * 100:.1f}%"),
        ]
        axis.legend(handles=handles, loc="best", fontsize=5, frameon=True, handletextpad=0.3)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=plot_config["dpi"], bbox_inches="tight")
    plt.close(figure)


def rerender_existing(results_dir: Path) -> None:
    import numpy as np

    explained_payload = json.loads((results_dir / "explained_variance.json").read_text())
    explained = [row["two_dimensional"] for row in explained_payload["layers"]]
    rows = list(csv.DictReader((results_dir / "pca_coordinates.csv").open()))
    layer_count = max(int(row["layer"]) for row in rows)
    examples_per_layer = len(rows) // layer_count
    coordinates = np.empty((layer_count, examples_per_layer, 2), dtype=np.float32)
    labels, pair_ids = [], []
    for row_index, row in enumerate(rows):
        layer = int(row["layer"]) - 1
        example = row_index % examples_per_layer
        coordinates[layer, example] = [float(row["pc1"]), float(row["pc2"])]
        if layer == 0:
            labels.append(row["language"])
            pair_ids.append(int(row["pair_id"]))
    config = yaml.safe_load(CONFIG_LOCAL.read_text())
    render_plot(
        coordinates,
        labels,
        pair_ids,
        explained,
        results_dir / "layerwise_pca.png",
        config["plot"],
    )
    print(f"Re-rendered: {results_dir / 'layerwise_pca.png'}")


def render_dashboard(history: list[dict[str, Any]]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['timestamp']))}</td>"
        f"<td>{html.escape(str(row['phase']))}</td>"
        f"<td>{row['completed']}/{row['total']}</td>"
        f"<td>{html.escape(str(row['status']))}</td>"
        f"<td>{html.escape(str(row.get('detail', '')))}</td>"
        "</tr>"
        for row in history
    )
    return (
        "<!doctype html><meta charset='utf-8'><title>Language PCA</title>"
        "<style>body{font:15px system-ui;margin:2rem;max-width:1000px}"
        "table{border-collapse:collapse;width:100%}th,td{padding:.6rem;"
        "border:1px solid #ccc;text-align:left}th{background:#eee}</style>"
        "<h1>English–French layerwise PCA</h1><table><thead><tr>"
        "<th>Timestamp</th><th>Phase</th><th>Progress</th><th>Status</th><th>Detail</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def persist(
    run_dir: Path,
    config: dict[str, Any],
    *,
    phase: str,
    completed: int,
    total: int,
    status: str,
    started: float,
    detail: str = "",
) -> None:
    event = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": phase,
        "completed": completed,
        "total": total,
        "status": status,
        "detail": detail,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "fingerprint": config["fingerprint"],
    }
    atomic_json(run_dir / "checkpoint.json", event)
    atomic_json(run_dir / "progress.json", event)
    history_path = run_dir / "dashboard_history.jsonl"
    with history_path.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    history = [json.loads(line) for line in history_path.read_text().splitlines()]
    atomic_text(run_dir / "dashboard.html", render_dashboard(history))
    print("PROGRESS " + json.dumps(event, sort_keys=True), flush=True)
    outputs.commit()


def select_pairs(config: dict[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    import langid
    from datasets import load_dataset

    dataset_config = config["dataset"]
    stream = load_dataset(
        dataset_config["id"],
        dataset_config["config"],
        split=dataset_config["split"],
        revision=dataset_config["revision"],
        streaming=True,
    ).shuffle(seed=dataset_config["shuffle_seed"], buffer_size=dataset_config["shuffle_buffer"])
    selected = []
    for stream_index, row in enumerate(stream):
        translation = row["translation"]
        english = translation["en"].strip()
        french = translation["fr"].strip()
        if not english or not french or english == french:
            continue
        english_ids, english_positions = chat_prompt_ids(tokenizer, english)
        french_ids, french_positions = chat_prompt_ids(tokenizer, french)
        lengths = (len(english_positions), len(french_positions))
        if min(lengths) < dataset_config["min_content_tokens"] or max(lengths) > dataset_config["max_content_tokens"]:
            continue
        if dataset_config["require_langid_match"]:
            if langid.classify(english)[0] != "en" or langid.classify(french)[0] != "fr":
                continue
        selected.append(
            {
                "pair_id": len(selected),
                "stream_index": stream_index,
                "english": english,
                "french": french,
                "english_token_ids": english_ids,
                "french_token_ids": french_ids,
                "english_content_positions": english_positions,
                "french_content_positions": french_positions,
            }
        )
        if len(selected) == dataset_config["pairs"]:
            return selected
    raise RuntimeError(f"stream exhausted after selecting only {len(selected)} pairs")


def extract_one(model: Any, token_ids: list[int], content_positions: list[int]) -> Any:
    import numpy as np
    import torch

    ids = torch.tensor([token_ids], device="cuda", dtype=torch.long)
    positions = torch.tensor(content_positions, device="cuda", dtype=torch.long)
    vectors = []
    with torch.inference_mode():
        hidden = model.transformer.tok_emb(ids)
        for block in model.transformer.blocks:
            hidden = block(hidden)
            vectors.append(hidden[0].index_select(0, positions).float().mean(dim=0).cpu().numpy())
    return np.asarray(vectors, dtype=np.float16)


def self_test(output: Path) -> None:
    import numpy as np
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(42)
    labels = ["English"] * 20 + ["French"] * 20
    pair_ids = list(range(20)) + list(range(20))
    activations = rng.normal(size=(4, 40, 12)).astype(np.float32)
    coordinates, explained = [], []
    for layer in activations:
        pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
        coordinates.append(pca.fit_transform(layer))
        explained.append(float(pca.explained_variance_ratio_.sum()))
    config = {
        "rows": 2,
        "columns": 2,
        "english_color": "#008080",
        "french_color": "#FA8072",
        "alpha": 0.7,
        "edge_color": "black",
        "edge_linewidth": 0.3,
        "pair_line_color": "#B0B0B0",
        "pair_line_alpha": 0.45,
        "pair_line_width": 0.35,
        "marker_size": 10,
        "dpi": 100,
    }
    render_plot(np.asarray(coordinates), labels, pair_ids, explained, output, config)
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("self-test plot was not created")
    print(f"Self-test plot: {output}")


@app.function(
    image=image,
    gpu="L40S",
    timeout=1800,
    retries=modal.Retries(max_retries=0),
    volumes={"/mnt/model-cache": model_cache, "/mnt/run-output": outputs},
    secrets=[hf_secret],
)
def run_remote(run_mode: str = "fresh") -> dict[str, Any]:
    import numpy as np
    import torch
    from sklearn.decomposition import PCA
    from transformers import AutoModel, AutoTokenizer

    started = time.monotonic()
    config = yaml.safe_load(CONFIG_REMOTE.read_text())
    config["run_mode"] = run_mode
    config["fingerprint"] = fingerprint(config)
    run_dir = REMOTE_RUN_DIR
    checkpoint_path = run_dir / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        if run_mode != "resume":
            raise ValueError("existing checkpoint requires explicit run_mode: resume")
        if checkpoint.get("fingerprint") != config["fingerprint"]:
            raise ValueError("resume config fingerprint mismatch")
        if checkpoint.get("status") not in {"running", "stopped"}:
            raise ValueError(f"cannot resume status {checkpoint.get('status')}")
    elif run_mode != "fresh":
        raise FileNotFoundError("resume requested without checkpoint")

    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(run_dir / "config.yaml", yaml.safe_dump(config, sort_keys=True))
    persist(run_dir, config, phase="initialize", completed=0, total=200, status="running", started=started)

    def handle_term(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt("SIGTERM received")

    signal.signal(signal.SIGTERM, handle_term)
    os.environ["HF_HOME"] = "/mnt/model-cache/huggingface"
    common = {
        "revision": config["model"]["revision"],
        "trust_remote_code": True,
        "cache_dir": "/mnt/model-cache/huggingface/hub",
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"], **common)
        pairs_path = run_dir / "selected_pairs.jsonl"
        if pairs_path.exists():
            pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line]
        else:
            pairs = select_pairs(config, tokenizer)
            atomic_text(pairs_path, "".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs))
            outputs.commit()
        persist(run_dir, config, phase="pairs_selected", completed=0, total=len(pairs), status="running", started=started)

        model = AutoModel.from_pretrained(
            config["model"]["id"],
            dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
            **common,
        ).eval()
        persist(run_dir, config, phase="model_loaded", completed=0, total=len(pairs), status="running", started=started)

        chunks_dir = run_dir / "activation_chunks"
        chunks_dir.mkdir(exist_ok=True)
        checkpoint_pairs = config["representation"]["checkpoint_pairs"]
        completed = 0
        for chunk_path in sorted(chunks_dir.glob("chunk_*.npz")):
            with np.load(chunk_path) as chunk:
                completed += int(chunk["activations"].shape[0] // 2)
        for start in range(completed, len(pairs), checkpoint_pairs):
            chunk_pairs = pairs[start : start + checkpoint_pairs]
            vectors, pair_ids, languages = [], [], []
            for pair in chunk_pairs:
                for language, key in (("English", "english"), ("French", "french")):
                    vectors.append(extract_one(model, pair[f"{key}_token_ids"], pair[f"{key}_content_positions"]))
                    pair_ids.append(pair["pair_id"])
                    languages.append(language)
            target = chunks_dir / f"chunk_{start:04d}_{start + len(chunk_pairs):04d}.npz"
            np.savez_compressed(target, activations=np.stack(vectors), pair_ids=np.asarray(pair_ids), languages=np.asarray(languages))
            completed = start + len(chunk_pairs)
            persist(run_dir, config, phase="extract", completed=completed, total=len(pairs), status="running", started=started)

        arrays, pair_ids, labels = [], [], []
        for chunk_path in sorted(chunks_dir.glob("chunk_*.npz")):
            with np.load(chunk_path) as chunk:
                arrays.append(chunk["activations"])
                pair_ids.extend(chunk["pair_ids"].tolist())
                labels.extend(chunk["languages"].tolist())
        example_major = np.concatenate(arrays, axis=0)
        layer_major = np.transpose(example_major, (1, 0, 2)).astype(np.float32)
        np.savez_compressed(run_dir / "activations.npz", activations=layer_major.astype(np.float16), pair_ids=np.asarray(pair_ids), languages=np.asarray(labels))

        coordinates, explained_rows = [], []
        for layer_index, layer in enumerate(layer_major):
            pca = PCA(n_components=2, svd_solver=config["pca"]["svd_solver"], random_state=config["pca"]["random_state"])
            xy = pca.fit_transform(layer)
            coordinates.append(xy)
            explained_rows.append(
                {
                    "layer": layer_index + 1,
                    "pc1": float(pca.explained_variance_ratio_[0]),
                    "pc2": float(pca.explained_variance_ratio_[1]),
                    "two_dimensional": float(pca.explained_variance_ratio_.sum()),
                }
            )
        coordinates_array = np.asarray(coordinates)
        atomic_json(run_dir / "explained_variance.json", {"layers": explained_rows})
        with (run_dir / "pca_coordinates.csv").open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["layer", "pair_id", "language", "pc1", "pc2"])
            for layer_index, xy in enumerate(coordinates_array):
                for example_index, point in enumerate(xy):
                    writer.writerow([layer_index + 1, pair_ids[example_index], labels[example_index], float(point[0]), float(point[1])])

        explained = [row["two_dimensional"] for row in explained_rows]
        render_plot(coordinates_array, labels, pair_ids, explained, run_dir / "layerwise_pca.png", config["plot"])
        results = {
            "status": "complete",
            "pairs": len(pairs),
            "examples": len(labels),
            "layers": int(layer_major.shape[0]),
            "hidden_size": int(layer_major.shape[2]),
            "fingerprint": config["fingerprint"],
            "explained_variance_2d": explained_rows,
        }
        atomic_json(run_dir / "results.json", results)
        atomic_text(
            run_dir / "RESULTS.md",
            "# English–French layerwise PCA\n\n"
            f"- Pairs: {len(pairs)}\n- Examples: {len(labels)}\n"
            f"- Layers: {layer_major.shape[0]}\n- Hidden size: {layer_major.shape[2]}\n",
        )
        required = ["layerwise_pca.png", "pca_coordinates.csv", "explained_variance.json", "activations.npz"]
        if any(not (run_dir / name).exists() or (run_dir / name).stat().st_size == 0 for name in required):
            raise RuntimeError("one or more final artifacts are missing or empty")
        persist(run_dir, config, phase="complete", completed=len(pairs), total=len(pairs), status="complete", started=started)
        return results
    except BaseException as exc:
        status = "stopped" if isinstance(exc, KeyboardInterrupt) else "failed"
        persist(run_dir, config, phase=status, completed=locals().get("completed", 0), total=200, status="stopped", started=started, detail=f"{type(exc).__name__}: {exc}")
        raise


@app.local_entrypoint()
def main(run_mode: str = "fresh") -> None:
    if run_mode not in {"fresh", "resume"}:
        raise ValueError("run_mode must be fresh or resume")
    call = run_remote.spawn(run_mode)
    print(json.dumps({"app": APP_NAME, "call_id": call.object_id, "run_mode": run_mode}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--rerender-results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/tmp/language_layerwise_pca_self_test.png"))
    args = parser.parse_args()
    if args.self_test:
        self_test(args.output)
    elif args.rerender_results:
        rerender_existing(args.rerender_results)
