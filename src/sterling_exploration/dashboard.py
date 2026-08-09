from __future__ import annotations

import html
import json
from collections.abc import Callable
from typing import Any


def _svg_plot(
    history: list[dict[str, Any]],
    *,
    title: str,
    series: list[tuple[str, Callable[[dict[str, Any]], float | None], str]],
) -> str:
    width, height = 760, 230
    left, right, top, bottom = 58, 18, 28, 38
    plot_width, plot_height = width - left - right, height - top - bottom
    elapsed = [float(row.get("elapsed_seconds") or 0.0) for row in history]
    x_max = max(elapsed, default=1.0) or 1.0
    values = [value for _, getter, _ in series for row in history if (value := getter(row)) is not None]
    y_max = max(values, default=1.0) or 1.0

    paths = []
    legend = []
    for series_index, (label, getter, color) in enumerate(series):
        points = []
        for row, x_value in zip(history, elapsed, strict=True):
            value = getter(row)
            if value is None:
                continue
            x = left + (x_value / x_max) * plot_width
            y = top + plot_height - (float(value) / y_max) * plot_height
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            paths.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
                'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        legend.append(
            f'<text x="{left + series_index * 190}" y="{height - 8}" fill="{color}" '
            f'font-size="12">● {html.escape(label)}</text>'
        )
    grid = "".join(
        f'<line x1="{left}" x2="{width-right}" y1="{top + plot_height * step / 4:.1f}" '
        f'y2="{top + plot_height * step / 4:.1f}" stroke="#e4e7ec"/>'
        for step in range(5)
    )
    return f"""<section class="plot"><h2>{html.escape(title)}</h2>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
{grid}<line x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_height}" stroke="#98a2b3"/>
<line x1="{left}" x2="{width-right}" y1="{top+plot_height}" y2="{top+plot_height}" stroke="#98a2b3"/>
<text x="4" y="{top+10}" font-size="11" fill="#667085">{y_max:.2f}</text>
<text x="{left}" y="{height-20}" font-size="11" fill="#667085">0m</text>
<text x="{width-right-38}" y="{height-20}" font-size="11" fill="#667085">{x_max/60:.1f}m</text>
{''.join(paths)}{''.join(legend)}</svg></section>"""


def render_dashboard(history: list[dict[str, Any]]) -> str:
    if not history:
        return "<!doctype html><html><body><h1>Waiting for progress data…</h1></body></html>"
    latest = history[-1]
    completed = int(latest.get("completed") or 0)
    total = int(latest.get("total") or 0)
    percent = completed / total * 100 if total else 0.0
    eta = latest.get("eta_seconds")
    eta_text = f"{float(eta)/60:.1f} min" if eta is not None else "—"
    cards = [
        ("Phase", str(latest.get("phase") or "unknown")),
        ("Progress", f"{completed} / {total} ({percent:.1f}%)"),
        ("Throughput", f"{float(latest.get('throughput_per_second') or 0):.3f} rows/s"),
        ("ETA", eta_text),
        ("Errors", str(latest.get("errors") or 0)),
        ("ASR", "—" if latest.get("latest_metric") is None else f"{float(latest['latest_metric']):.1%}"),
    ]
    plots = [
        _svg_plot(
            history,
            title="Rows completed",
            series=[("completed", lambda row: float(row.get("completed") or 0), "#155eef")],
        ),
        _svg_plot(
            history,
            title="Throughput",
            series=[
                ("rows / second", lambda row: float(row.get("throughput_per_second") or 0), "#7a5af8")
            ],
        ),
        _svg_plot(
            history,
            title="Unique concepts discovered",
            series=[
                (
                    "known",
                    lambda row: _method_value(row, "unique_known_concepts_fired"),
                    "#067647",
                ),
                (
                    "unknown",
                    lambda row: _method_value(row, "unique_unknown_concepts_fired"),
                    "#dc6803",
                ),
            ],
        ),
        _svg_plot(
            history,
            title="Attack success rate (judge phase)",
            series=[("ASR", lambda row: _optional_float(row.get("latest_metric")), "#b42318")],
        ),
    ]
    card_html = "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in cards
    )
    payload = html.escape(json.dumps(latest, indent=2, sort_keys=True))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="10"><title>Sterling AdvBench dashboard</title>
<style>body{{font:15px system-ui;background:#f8fafc;color:#101828;max-width:1500px;margin:2rem auto;padding:0 1.25rem}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:1rem}}.card,.plot{{background:white;border:1px solid #e4e7ec;border-radius:12px;padding:1rem;box-shadow:0 1px 2px #1018280d}}
.card span{{display:block;color:#667085;margin-bottom:.35rem}}.card strong{{font-size:1.35rem}}.bar{{height:14px;background:#e4e7ec;border-radius:99px;overflow:hidden;margin:1.25rem 0 2rem}}.bar div{{height:100%;background:#155eef;width:{percent:.2f}%}}
.plots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:1rem}}.plot h2{{font-size:1rem;margin:0 0 .5rem}}svg{{width:100%;height:auto}}pre{{background:#101828;color:#f2f4f7;padding:1rem;border-radius:10px;overflow:auto}}</style></head>
<body><h1>Sterling AdvBench experiment</h1><p><code>{html.escape(str(latest.get('run_id') or ''))}</code></p>
<div class="cards">{card_html}</div><div class="bar"><div></div></div><div class="plots">{''.join(plots)}</div>
<details><summary>Latest structured progress</summary><pre>{payload}</pre></details></body></html>"""


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _method_value(row: dict[str, Any], key: str) -> float | None:
    value = (row.get("method_metrics") or {}).get(key)
    return _optional_float(value)
