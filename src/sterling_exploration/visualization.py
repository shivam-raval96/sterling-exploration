from __future__ import annotations

import html
import json
from typing import Any


def _concept_bar_chart(rows: list[dict[str, Any]], title: str, limit: int = 20) -> str:
    selected = rows[:limit]
    width, left, right, row_height = 1180, 330, 110, 27
    height = 54 + row_height * len(selected)
    maximum = max((row["token_firings"] for row in selected), default=1)
    plot_width = width - left - right
    marks = []
    for index, row in enumerate(selected):
        y = 38 + index * row_height
        bar_width = plot_width * row["token_firings"] / maximum
        name = str(row.get("concept_name") or "unlabeled")
        if len(name) > 37:
            name = name[:36] + "…"
        label = f"{row['concept_id']} · {name}"
        marks.append(
            f"<text x='{left - 10}' y='{y + 13}' text-anchor='end'>{html.escape(label)}</text>"
            f"<rect x='{left}' y='{y}' width='{bar_width:.1f}' height='17' rx='3'/>"
            f"<text x='{left + bar_width + 7:.1f}' y='{y + 13}'>{row['token_firings']:,}</text>"
        )
    return (
        f"<section class='plot'><h2>{html.escape(title)}</h2>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        f"{''.join(marks)}</svg></section>"
    )


def _rank_curve(groups: list[tuple[str, list[dict[str, Any]]]]) -> str:
    width, height, left, top, bottom = 1040, 390, 76, 28, 52
    plot_width, plot_height = width - left - 32, height - top - bottom
    limit = min(100, max((len(rows) for _, rows in groups), default=0))
    maximum = max((rows[0]["token_firings"] for _, rows in groups if rows), default=1)
    paths = []
    legend = []
    for series_index, (label, rows) in enumerate(groups, start=1):
        points = []
        for index, row in enumerate(rows[:limit]):
            x = left + (index / max(limit - 1, 1)) * plot_width
            y = top + (1 - row["token_firings"] / maximum) * plot_height
            points.append(f"{x:.1f},{y:.1f}")
        paths.append(f"<polyline class='series s{series_index}' points='{' '.join(points)}'/>")
        legend.append(
            f"<span><i class='swatch s{series_index}'></i>{html.escape(label)}</span>"
        )
    grid = "".join(
        f"<line x1='{left}' y1='{top + plot_height * fraction:.1f}' x2='{width - 32}' "
        f"y2='{top + plot_height * fraction:.1f}'/>"
        for fraction in (0, 0.25, 0.5, 0.75, 1)
    )
    return (
        "<section class='plot'><h2>Top-100 concept rank decay</h2>"
        f"<div class='legend'>{''.join(legend)}</div><svg viewBox='0 0 {width} {height}' "
        "role='img' aria-label='Token firings by concept rank'>"
        f"<g class='grid'>{grid}</g>{''.join(paths)}"
        f"<text x='{left}' y='{height - 15}'>rank 1</text>"
        f"<text x='{width - 32}' y='{height - 15}' text-anchor='end'>rank {limit}</text>"
        f"<text x='8' y='{top + 5}'>{maximum:,}</text>"
        f"<text x='8' y='{top + plot_height}'>0</text></svg></section>"
    )


def concept_distribution_html(distribution: dict[str, Any], top_n: int) -> str:
    sections = []
    for scope, groups in distribution.items():
        for concept_type, rows in groups.items():
            items = []
            for row in rows[:top_n]:
                aligned = row.get("top_tokens", [])
                token_text = ", ".join(html.escape(item["token"]) for item in aligned)
                name = html.escape(str(row.get("concept_name") or "Unlabeled concept"))
                description = html.escape(str(row.get("concept_description") or ""))
                items.append(
                    "<li>"
                    f"<code>{row['concept_id']}</code> — <strong>{name}</strong> — "
                    f"{row['input_firings']} inputs; {row['token_firings']} tokens; "
                    f"mean={row['mean_activation']:.4f}; max={row['max_activation']:.4f}"
                    + (f"<p>{description}</p>" if description else "")
                    + (f"; aligned tokens: {token_text}" if token_text else "")
                    + "</li>"
                )
            sections.append(
                f"<details><summary>{html.escape(scope)} / {html.escape(concept_type)} "
                f"— {len(rows):,} observed concepts</summary><ol>{''.join(items)}</ol></details>"
            )
    user_known = distribution["user_content"]["known"]
    user_unknown = distribution["user_content"]["unknown"]
    full_known = distribution["full_chat_input"]["known"]
    full_unknown = distribution["full_chat_input"]["unknown"]
    total_user_firings = sum(row["token_firings"] for row in user_known)
    top_one_count = max(1, round(len(user_known) * 0.01))
    top_one_share = sum(row["token_firings"] for row in user_known[:top_one_count]) / max(
        total_user_firings, 1
    )
    charts = "".join(
        [
            _concept_bar_chart(user_known, "Most frequent known concepts — user content"),
            _concept_bar_chart(user_unknown, "Most frequent unknown concepts — user content"),
            _rank_curve(
                [
                    ("User known", user_known),
                    ("User unknown", user_unknown),
                    ("Full-chat known", full_known),
                    ("Full-chat unknown", full_unknown),
                ]
            ),
        ]
    )
    payload = json.dumps(distribution, sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Concept firing distribution</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#f8fafc;--panel:#fff;--text:#172033;--muted:#667085;--line:#d7dce5;--known:#3157d5;--unknown:#d07a24}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1420;--panel:#171e2c;--text:#edf1f7;--muted:#aab4c3;--line:#354052;--known:#7d9cff;--unknown:#f2aa62}}}}
*{{box-sizing:border-box}}body{{font:15px system-ui;margin:0;background:var(--bg);color:var(--text)}}
main{{max-width:1280px;margin:0 auto;padding:28px 20px 60px}}nav{{display:flex;gap:16px;margin-bottom:20px}}a{{color:var(--known)}}
h1{{margin-bottom:6px}}.sub{{color:var(--muted);margin-top:0}}.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:24px 0}}
.stat,.plot{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}.stat strong{{display:block;font-size:28px;margin-top:5px}}
.plot{{margin:14px 0}}.plot svg{{display:block;width:100%;height:auto}}svg rect{{fill:var(--known)}}svg text{{fill:var(--text);font-size:12px}}.grid line{{stroke:var(--line);stroke-width:1}}
.series{{fill:none;stroke-width:3}}.series.s1,.swatch.s1{{stroke:var(--known);background:var(--known)}}.series.s2,.swatch.s2{{stroke:var(--unknown);background:var(--unknown)}}
.series.s3{{stroke:var(--known);opacity:.45;stroke-dasharray:7 5}}.series.s4{{stroke:var(--unknown);opacity:.45;stroke-dasharray:7 5}}
.swatch{{display:inline-block;width:18px;height:3px;margin-right:6px;vertical-align:middle}}.legend{{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted)}}
details{{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin:10px 0;padding:12px 16px}}summary{{cursor:pointer;font-weight:600}}li{{margin:.45rem 0}}code{{background:var(--bg);padding:.1rem .25rem}}
@media(max-width:600px){{main{{padding:18px 10px 40px}}.plot{{padding:10px}}}}
</style>
</head><body><main><nav><a href="dashboard.html">Run dashboard</a><a href="generations.html">Generations</a></nav>
<h1>Concept firing dashboard</h1><p class="sub">Provider-authored concept names from Guide Labs' catalog; learned token alignments remain the empirical cross-check.</p>
<div class="stats"><div class="stat">Known concepts observed<strong>{len(user_known):,}</strong></div>
<div class="stat">Unknown concepts observed<strong>{len(user_unknown):,}</strong></div>
<div class="stat">Top 1% known firing share<strong>{top_one_share:.1%}</strong></div>
<div class="stat">Known user-token firings<strong>{total_user_firings:,}</strong></div></div>
{charts}<h2>Ranked concept lists</h2>{''.join(sections)}
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
