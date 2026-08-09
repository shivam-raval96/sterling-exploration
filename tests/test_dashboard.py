from sterling_exploration.dashboard import render_dashboard


def test_dashboard_renders_progress_and_plots() -> None:
    rendered = render_dashboard(
        [
            {
                "run_id": "run-one",
                "phase": "generation",
                "completed": 5,
                "total": 10,
                "elapsed_seconds": 20,
                "throughput_per_second": 0.25,
                "eta_seconds": 20,
                "errors": 0,
                "latest_metric": None,
                "method_metrics": {
                    "unique_known_concepts_fired": 100,
                    "unique_unknown_concepts_fired": 20,
                },
            }
        ]
    )
    assert "5 / 10 (50.0%)" in rendered
    assert rendered.count("<svg") == 4
    assert "Unique concepts discovered" in rendered
