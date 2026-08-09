from __future__ import annotations

import html
import json
from typing import Any


def concept_distribution_html(distribution: dict[str, Any], top_n: int) -> str:
    sections = []
    for scope, groups in distribution.items():
        for concept_type, rows in groups.items():
            items = []
            for row in rows[:top_n]:
                aligned = row.get("top_tokens", [])
                token_text = ", ".join(html.escape(item["token"]) for item in aligned)
                items.append(
                    "<li>"
                    f"<code>{row['concept_id']}</code> — "
                    f"{row['input_firings']} inputs; {row['token_firings']} tokens; "
                    f"mean={row['mean_activation']:.4f}; max={row['max_activation']:.4f}"
                    + (f"; aligned tokens: {token_text}" if token_text else "")
                    + "</li>"
                )
            sections.append(
                f"<h2>{html.escape(scope)} / {html.escape(concept_type)}</h2><ol>{''.join(items)}</ol>"
            )
    payload = html.escape(json.dumps(distribution, sort_keys=True))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Concept firing distribution</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:0 1rem}}li{{margin:.45rem 0}}code{{background:#eee;padding:.1rem .25rem}}</style>
</head><body><h1>Concept firing distribution</h1>{''.join(sections)}
<script type="application/json" id="distribution">{payload}</script></body></html>"""


def generations_html(rows: list[dict[str, Any]]) -> str:
    cards = []
    counts = {"jailbreak": 0, "not_jailbroken": 0, "error": 0}
    for row in rows:
        judgment = row.get("judgment") or {}
        label = judgment.get("harmful_compliance")
        if label is True:
            state, title = "jailbreak", "JAILBREAK"
        elif label is False:
            state, title = "not-jailbroken", "NOT JAILBROKEN"
        else:
            state, title = "error", "UNEVALUATED / ERROR"
        counts[state.replace("-", "_")] += 1
        confidence = judgment.get("confidence")
        confidence_text = f" · confidence {float(confidence):.2f}" if confidence is not None else ""
        rationale = judgment.get("rationale") or judgment.get("judgment_error") or "No judgment."
        generation_error = row.get("generation_error")
        if generation_error:
            rationale = f"Generation error: {generation_error}"
        cards.append(
            f"<article class='card {state}'>"
            f"<header><span class='badge'>{title}</span>"
            f"<strong>Prompt {int(row['input_index'])}</strong>{confidence_text}</header>"
            f"<h3>Prompt</h3><pre>{html.escape(str(row.get('prompt') or ''))}</pre>"
            f"<h3>Generation</h3><pre>{html.escape(str(row.get('response') or ''))}</pre>"
            f"<h3>Judge rationale</h3><p>{html.escape(str(rationale))}</p>"
            "<details><summary>AdvBench reference target</summary>"
            f"<pre>{html.escape(str(row.get('target') or ''))}</pre></details>"
            "</article>"
        )
    summary = (
        f"<strong>{counts['jailbreak']}</strong> jailbreak · "
        f"<strong>{counts['not_jailbroken']}</strong> not jailbroken · "
        f"<strong>{counts['error']}</strong> unevaluated/error"
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>AdvBench generations</title>
<style>
:root{{--red:#b42318;--red-bg:#fff1f0;--green:#067647;--green-bg:#ecfdf3;--gray:#475467;--gray-bg:#f2f4f7}}
body{{font:16px system-ui;max-width:1200px;margin:3rem auto;padding:0 1rem;color:#101828}}
.legend{{position:sticky;top:0;background:white;padding:1rem 0;border-bottom:1px solid #d0d5dd;z-index:1}}
.card{{border:2px solid;margin:1.25rem 0;padding:1rem 1.25rem;border-radius:10px}}
.card.jailbreak{{border-color:var(--red);background:var(--red-bg)}}
.card.not-jailbroken{{border-color:var(--green);background:var(--green-bg)}}
.card.error{{border-color:var(--gray);background:var(--gray-bg)}}
header{{display:flex;gap:.75rem;align-items:center}}.badge{{font-weight:800;padding:.25rem .5rem;border-radius:4px}}
.jailbreak .badge{{background:var(--red);color:white}}.not-jailbroken .badge{{background:var(--green);color:white}}
.error .badge{{background:var(--gray);color:white}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#ffffffaa;padding:.75rem;border-radius:6px}}
</style></head><body><h1>AdvBench generations</h1><div class="legend">{summary}<br>
<small>Red = judge found harmful compliance (jailbreak). Green = judge found no jailbreak. Gray is excluded from ASR.</small></div>
{''.join(cards)}</body></html>"""
