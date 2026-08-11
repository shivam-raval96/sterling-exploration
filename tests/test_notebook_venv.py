from pathlib import Path

from exploration.check_notebook_venv import configure_environment, environment_report


def test_environment_report_matches_current_prefix() -> None:
    report = environment_report(Path(__import__("sys").prefix))
    assert report["in_virtualenv"] is True
    assert report["matches_expected_prefix"] is True
    assert report["python_supported"] is True


def test_configure_environment_matches_kernel(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    prefix = configure_environment()
    assert __import__("os").environ["PATH"].split(__import__("os").pathsep)[0] == str(
        Path(__import__("sys").executable).resolve().parent
    )
    assert Path(__import__("os").environ["VIRTUAL_ENV"]) == prefix
