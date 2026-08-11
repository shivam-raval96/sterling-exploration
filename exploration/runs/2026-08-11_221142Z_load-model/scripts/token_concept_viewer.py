from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Any

import modal
import yaml

APP_NAME = "sterling-bilingual-token-concepts"
RUN_ROOT = Path("exploration/runs/2026-08-11_221142Z_load-model")
CONFIG_LOCAL = RUN_ROOT / "config_token_concept_viewer.yaml"
PAIRS_LOCAL = (
    RUN_ROOT
    / "results/04-english-french-layerwise-pca/04-english-french-layerwise-pca/selected_pairs.jsonl"
)
CATALOG_LOCAL = Path(".cache/concepts/concept_labels.parquet")
CONFIG_REMOTE = Path("/root/config_token_concept_viewer.yaml")
PAIRS_REMOTE = Path("/root/selected_pairs.jsonl")
CATALOG_REMOTE = Path("/root/concept_labels.parquet")
REMOTE_RUN_DIR = Path(
    "/mnt/run-output/exploration/runs/2026-08-11_221142Z_load-model/"
    "05-bilingual-token-concept-viewer"
)

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "steerling==0.2.0",
        "accelerate==1.14.0",
        "pandas==2.2.3",
        "pyarrow==25.0.1",
        "pyyaml==6.0.2",
    )
    .add_local_file(str(CONFIG_LOCAL), str(CONFIG_REMOTE), copy=True)
    .add_local_file(str(PAIRS_LOCAL), str(PAIRS_REMOTE), copy=True)
    .add_local_file(str(CATALOG_LOCAL), str(CATALOG_REMOTE), copy=True)
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
    # Execution mode changes during recovery, but the experiment inputs do not.
    clean["run_mode"] = "fresh"
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        "<!doctype html><meta charset='utf-8'><title>Token concepts</title>"
        "<style>body{font:15px system-ui;margin:2rem;max-width:1000px}"
        "table{border-collapse:collapse;width:100%}th,td{padding:.6rem;"
        "border:1px solid #ccc;text-align:left}th{background:#eee}</style>"
        "<h1>Bilingual token concepts</h1><table><thead><tr>"
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


def catalog_records(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    records = {}
    fields = [
        "concept_name",
        "concept_description",
        "group_name",
        "is_steerable",
        "is_tone",
        "is_alignment",
        "is_demographic",
    ]
    for row in frame.itertuples(index=False):
        value = row._asdict()
        record = {}
        for field in fields:
            item = value[field]
            if field == "group_name":
                record[field] = None if pd.isna(item) else str(item)
            elif field.startswith("is_"):
                record[field] = False if pd.isna(item) else bool(item)
            else:
                record[field] = None if pd.isna(item) else str(item)
        records[(str(value["head"]), int(value["concept_id"]))] = record
    return records


def visible_token(token: str) -> str:
    if token == "":
        return "∅"
    return token.replace(" ", "␠").replace("\n", "↵").replace("\t", "⇥")


def concept_html(concept: dict[str, Any]) -> str:
    flags = [
        name.removeprefix("is_")
        for name in ("is_steerable", "is_tone", "is_alignment", "is_demographic")
        if concept.get(name)
    ]
    badges = "".join(f"<span class='flag'>{html.escape(flag)}</span>" for flag in flags)
    group = concept.get("group_name")
    return (
        "<li>"
        f"<div><strong>{html.escape(str(concept['concept_name']))}</strong> "
        f"<code>{html.escape(str(concept['head']))}:{concept['concept_id']}</code></div>"
        f"<div class='score'>activation {concept['activation']:.3f} · logit {concept['logit']:.3f}</div>"
        + (f"<div class='group'>{html.escape(str(group))}</div>" if group else "")
        + f"<p>{html.escape(str(concept['concept_description']))}</p>{badges}</li>"
    )


def token_html(token: dict[str, Any], language: str) -> str:
    known = "".join(concept_html(item) for item in token["known_concepts"])
    unknown = "".join(concept_html(item) for item in token["unknown_concepts"])
    token_text = visible_token(token["token"])
    return (
        f"<span class='token {language.lower()}' tabindex='0' aria-label='Token {token['position']}: "
        f"{html.escape(token_text)}'>{html.escape(token_text)}"
        "<template class='concept-template'>"
        "<div class='panelhead'>Concept viewer</div>"
        f"<div class='conceptcol'><b>Known concepts</b><ol>{known}</ol></div>"
        f"<div class='conceptcol'><b>Unknown concepts</b><ol>{unknown}</ol></div>"
        "</template></span>"
    )


def viewer_html(records: list[dict[str, Any]]) -> str:
    cards = []
    for record in records:
        languages = {}
        for language in ("English", "French"):
            item = record[language.lower()]
            tokens = "".join(token_html(token, language) for token in item["tokens"])
            languages[language] = (
                f"<section class='language {language.lower()}'><h3>{language}</h3>"
                f"<p class='raw'>{html.escape(item['text'])}</p><div class='tokens'>{tokens}</div></section>"
            )
        search = (record["english"]["text"] + " " + record["french"]["text"]).lower()
        cards.append(
            f"<article class='pair' id='pair-{record['pair_id']}' data-search='{html.escape(search, quote=True)}'>"
            f"<h2>Pair {record['pair_id'] + 1}</h2><div class='contentgrid'><div class='pairgrid'>"
            f"{languages['English']}{languages['French']}</div>"
            "<section class='concept-panel' aria-live='polite'>"
            "<p class='panel-empty'>Hover over or focus a token to inspect its concepts.</p>"
            "</section></div></article>"
        )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>English–French token concepts</title><style>
:root{{--bg:#f6f7f9;--panel:#fff;--text:#172033;--muted:#667085;--line:#d7dce5;--en:#008080;--fr:#fa8072;--known:#3157d5;--unknown:#a55b13}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:100;background:#fffffff2;backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:14px 20px}}
header>div,main{{max-width:1440px;margin:auto}}h1{{margin:0;font-size:24px}}.sub{{margin:4px 0 10px;color:var(--muted)}}
.nav{{display:flex;align-items:center;gap:8px;flex-wrap:wrap}}button,select{{padding:9px 12px;border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--text);font:inherit}}button{{cursor:pointer}}button:disabled{{opacity:.4;cursor:default}}select{{min-width:130px}}.counter{{color:var(--muted);margin-left:auto}}main{{padding:20px}}
.pair{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;margin:0;overflow:hidden}}.pair[hidden]{{display:none}}.pair h2{{font-size:14px;color:var(--muted);margin:0 0 10px}}
.contentgrid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(380px,.9fr);gap:18px;align-items:start}}.pairgrid{{display:grid;grid-template-columns:1fr;gap:22px}}.language{{border-top:4px solid;padding-top:8px;min-width:0}}.language.english{{border-color:var(--en)}}.language.french{{border-color:var(--fr)}}
.language h3{{margin:0}}.english h3{{color:var(--en)}}.french h3{{color:#c45146}}.raw{{color:var(--muted);margin:.35rem 0 .8rem}}
.tokens{{display:flex;flex-wrap:wrap;gap:4px;align-items:flex-start}}.token{{display:inline-block;border:1px solid var(--line);border-radius:5px;padding:3px 5px;background:#fff;font:13px ui-monospace,monospace;cursor:pointer;outline:none;transition:background .12s,border-color .12s,box-shadow .12s}}
.token.english{{box-shadow:inset 0 -2px var(--en)}}.token.french{{box-shadow:inset 0 -2px var(--fr)}}.token:hover,.token:focus{{border-color:#111}}.token.selected{{color:#fff;border-color:#111;box-shadow:none}}.token.english.selected{{background:var(--en)}}.token.french.selected{{background:#c45146}}
.concept-panel{{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px;min-height:340px;max-height:600px;overflow-y:auto;overscroll-behavior:contain;background:#fcfaff;color:#282039;border:1px solid #eadff7;border-radius:10px;font:12px/1.4 system-ui,sans-serif}}
.panelhead{{position:sticky;top:-16px;z-index:2;grid-column:1/-1;margin:-16px -16px 0;padding:12px 16px;background:#f7f2ff;border-bottom:1px solid #eadff7;font:600 13px ui-monospace,monospace}}.panel-empty{{grid-column:1/-1;align-self:center;text-align:center;color:#746789}}.conceptcol>b{{color:#57466f}}ol{{padding-left:20px;margin:6px 0}}li{{margin:0 0 12px}}li p{{margin:3px 0;color:#514760}}code{{color:#6546a3}}.score,.group{{color:#746789}}.flag{{display:inline-block;background:#eee6fa;color:#503979;border-radius:10px;padding:1px 6px;margin-right:3px}}
@media(max-width:1000px){{.contentgrid{{grid-template-columns:1fr}}.concept-panel{{max-height:440px}}}}@media(max-width:650px){{.concept-panel{{grid-template-columns:1fr}}.panelhead{{grid-column:1}}.counter{{width:100%;margin-left:0}}}}
</style></head><body><header><div><h1>English–French token concepts</h1>
<p class="sub">24 aligned pairs · inspect one pair at a time</p>
<div class="nav"><button id="prev" type="button">← Previous</button><select id="pair-select" aria-label="Choose pair">{''.join(f'<option value="{i}">Pair {i + 1}</option>' for i in range(len(records)))}</select><button id="next" type="button">Next →</button><span class="counter" id="counter"></span></div></div></header>
<main>{''.join(cards)}</main><script>
const pairs=[...document.querySelectorAll('.pair')],select=document.getElementById('pair-select'),prev=document.getElementById('prev'),next=document.getElementById('next'),counter=document.getElementById('counter');let current=0;
function showPair(index){{current=Math.max(0,Math.min(pairs.length-1,index));pairs.forEach((pair,i)=>pair.hidden=i!==current);select.value=String(current);counter.textContent=`${{current+1}} of ${{pairs.length}}`;prev.disabled=current===0;next.disabled=current===pairs.length-1;window.scrollTo({{top:0,behavior:'smooth'}})}}
function selectToken(token){{const pair=token.closest('.pair'),panel=pair.querySelector('.concept-panel'),template=token.querySelector('.concept-template');pair.querySelectorAll('.token.selected').forEach(item=>item.classList.remove('selected'));token.classList.add('selected');panel.replaceChildren(template.content.cloneNode(true));panel.scrollTop=0}}
pairs.forEach(pair=>pair.querySelectorAll('.token').forEach(token=>{{token.addEventListener('mouseenter',()=>selectToken(token));token.addEventListener('focus',()=>selectToken(token));token.addEventListener('click',()=>selectToken(token))}}));
select.addEventListener('change',()=>showPair(Number(select.value)));prev.addEventListener('click',()=>showPair(current-1));next.addEventListener('click',()=>showPair(current+1));document.addEventListener('keydown',event=>{{if(event.key==='ArrowLeft')showPair(current-1);if(event.key==='ArrowRight')showPair(current+1)}});showPair(0);
</script></body></html>"""


def enrich_concepts(
    head: str,
    ids: Any,
    logits: Any,
    catalog: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    import torch

    enriched = []
    for concept_id, logit in zip(ids, logits, strict=True):
        cid, score = int(concept_id), float(logit)
        metadata = catalog.get((head, cid))
        if metadata is None:
            metadata = {
                "concept_name": "Unlabeled concept",
                "concept_description": "No provider catalog row was found.",
                "group_name": None,
                "is_steerable": False,
                "is_tone": False,
                "is_alignment": False,
                "is_demographic": False,
            }
        enriched.append(
            {
                "head": head,
                "concept_id": cid,
                "logit": score,
                "activation": float(torch.sigmoid(torch.tensor(score))),
                **metadata,
            }
        )
    return enriched


def analyze_language(
    model: Any,
    tokenizer: Any,
    pair: dict[str, Any],
    key: str,
    catalog: dict[tuple[str, int], dict[str, Any]],
    known_top_k: int,
    unknown_top_k: int,
) -> dict[str, Any]:
    import torch

    ids = torch.tensor([pair[f"{key}_token_ids"]], device="cuda", dtype=torch.long)
    with torch.inference_mode():
        _, decomposition = model(ids, minimal_output=True, unknown_topk=unknown_top_k)
    positions = pair[f"{key}_content_positions"]
    tokens = []
    for position in positions:
        tokens.append(
            {
                "position": position,
                "token_id": int(ids[0, position]),
                "token": tokenizer.decode([int(ids[0, position])], skip_special_tokens=False),
                "known_concepts": enrich_concepts(
                    "known",
                    decomposition.known_topk_indices[0, position, :known_top_k],
                    decomposition.known_topk_logits[0, position, :known_top_k],
                    catalog,
                ),
                "unknown_concepts": enrich_concepts(
                    "unknown",
                    decomposition.unknown_topk_indices[0, position, :unknown_top_k],
                    decomposition.unknown_topk_logits[0, position, :unknown_top_k],
                    catalog,
                ),
            }
        )
    return {"text": pair[key], "tokens": tokens}


def self_test(output: Path) -> None:
    concept = {
        "head": "known",
        "concept_id": 2,
        "concept_name": "Data Visualization Plots",
        "concept_description": "Tokens related to graphical representations of data.",
        "group_name": "Plotting and Visualization",
        "activation": 0.91,
        "logit": 2.31,
        "is_steerable": True,
        "is_tone": False,
        "is_alignment": False,
        "is_demographic": False,
    }
    token = {"position": 4, "token_id": 123, "token": " hello", "known_concepts": [concept] * 5, "unknown_concepts": [{**concept, "head": "unknown"}] * 5}
    records = [{"pair_id": 0, "english": {"text": "Hello world", "tokens": [token]}, "french": {"text": "Bonjour le monde", "tokens": [{**token, "token": " Bonjour"}]}}]
    atomic_text(output, viewer_html(records))
    if output.stat().st_size < 1000:
        raise RuntimeError("self-test HTML is unexpectedly small")
    print(f"Self-test viewer: {output}")


@app.function(
    image=image,
    gpu="L40S",
    timeout=1800,
    retries=modal.Retries(max_retries=0),
    volumes={"/mnt/model-cache": model_cache, "/mnt/run-output": outputs},
    secrets=[hf_secret],
)
def run_remote(run_mode: str = "fresh") -> dict[str, Any]:
    import torch
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
            raise ValueError("resume fingerprint mismatch")
        if checkpoint.get("status") not in {"running", "stopped"}:
            raise ValueError(f"cannot resume status {checkpoint.get('status')}")
    elif run_mode != "fresh":
        raise FileNotFoundError("resume requested without checkpoint")

    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(run_dir / "config.yaml", yaml.safe_dump(config, sort_keys=True))
    total = config["source"]["pair_count"]
    persist(run_dir, config, phase="initialize", completed=0, total=total, status="running", started=started)

    def handle_term(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt("SIGTERM received")

    signal.signal(signal.SIGTERM, handle_term)
    os.environ["HF_HOME"] = "/mnt/model-cache/huggingface"
    try:
        if file_sha256(CATALOG_REMOTE) != config["catalog"]["sha256"]:
            raise ValueError("concept catalog SHA-256 mismatch")
        catalog = catalog_records(CATALOG_REMOTE)
        if len(catalog) != config["catalog"]["rows"]:
            raise ValueError("concept catalog row-count mismatch")
        pairs = [json.loads(line) for line in PAIRS_REMOTE.read_text().splitlines() if line][:total]
        if len(pairs) != total:
            raise ValueError("selected-pairs file has too few rows")
        common = {
            "revision": config["model"]["revision"],
            "trust_remote_code": True,
            "cache_dir": "/mnt/model-cache/huggingface/hub",
        }
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"], **common)
        model = AutoModel.from_pretrained(
            config["model"]["id"],
            dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
            **common,
        ).eval()
        persist(run_dir, config, phase="model_loaded", completed=0, total=total, status="running", started=started)

        chunks_dir = run_dir / "token_concepts_chunks"
        chunks_dir.mkdir(exist_ok=True)
        completed = sum(1 for path in chunks_dir.glob("chunk_*.jsonl") for line in path.read_text().splitlines() if line)
        step = config["processing"]["checkpoint_pairs"]
        for start in range(completed, total, step):
            records = []
            for pair in pairs[start : start + step]:
                records.append(
                    {
                        "pair_id": pair["pair_id"],
                        "english": analyze_language(model, tokenizer, pair, "english", catalog, config["concepts"]["known_top_k"], config["concepts"]["unknown_top_k"]),
                        "french": analyze_language(model, tokenizer, pair, "french", catalog, config["concepts"]["known_top_k"], config["concepts"]["unknown_top_k"]),
                    }
                )
            atomic_text(chunks_dir / f"chunk_{start:04d}_{start + len(records):04d}.jsonl", "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
            completed = start + len(records)
            persist(run_dir, config, phase="extract", completed=completed, total=total, status="running", started=started)

        all_records = [json.loads(line) for path in sorted(chunks_dir.glob("chunk_*.jsonl")) for line in path.read_text().splitlines() if line]
        atomic_text(run_dir / "token_concepts.jsonl", "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in all_records))
        atomic_text(run_dir / "token_concepts.html", viewer_html(all_records))
        token_count = sum(len(record[language]["tokens"]) for record in all_records for language in ("english", "french"))
        results = {"status": "complete", "pairs": len(all_records), "conversations": len(all_records) * 2, "user_tokens": token_count, "known_top_k": config["concepts"]["known_top_k"], "unknown_top_k": config["concepts"]["unknown_top_k"], "catalog_sha256": config["catalog"]["sha256"], "fingerprint": config["fingerprint"]}
        atomic_json(run_dir / "results.json", results)
        atomic_text(run_dir / "RESULTS.md", f"# Bilingual token concept viewer\n\n- Pairs: {len(all_records)}\n- Conversations: {len(all_records) * 2}\n- User-content tokens: {token_count}\n")
        if (run_dir / "token_concepts.html").stat().st_size < 1000:
            raise RuntimeError("viewer HTML is missing or unexpectedly small")
        persist(run_dir, config, phase="complete", completed=total, total=total, status="complete", started=started)
        return results
    except BaseException as exc:
        status = "stopped" if isinstance(exc, KeyboardInterrupt) else "failed"
        persist(run_dir, config, phase=status, completed=locals().get("completed", 0), total=total, status="stopped", started=started, detail=f"{type(exc).__name__}: {exc}")
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
    parser.add_argument("--output", type=Path, default=Path("/tmp/token_concept_viewer_self_test.html"))
    parser.add_argument("--render-results", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test(args.output)
    elif args.render_results:
        records = [
            json.loads(line)
            for line in args.render_results.read_text().splitlines()
            if line
        ]
        atomic_text(args.output, viewer_html(records))
        print(f"Rendered viewer: {args.output}")
