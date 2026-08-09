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
