from datetime import UTC, datetime

from sterling_exploration.artifacts import config_fingerprint, make_run_id


def test_fingerprint_ignores_resume_metadata() -> None:
    fresh = {
        "model": "x",
        "run_mode": "fresh",
        "run_id": "one",
        "fingerprint": "old",
        "seed": 42,
    }
    resumed = {
        "model": "x",
        "run_mode": "resume",
        "run_id": "two",
        "fingerprint": "new",
        "seed": 42,
    }
    assert config_fingerprint(fresh) == config_fingerprint(resumed)


def test_run_id_is_utc_and_slugged() -> None:
    now = datetime(2026, 8, 8, 19, 2, 3, tzinfo=UTC)
    assert make_run_id("Basic concept probe!", now) == (
        "2026-08-08_190203Z_basic-concept-probe"
    )
