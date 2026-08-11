from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def environment_report(expected_prefix: Path | None = None) -> dict[str, str | bool]:
    prefix = Path(sys.prefix).resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    expected = expected_prefix.resolve() if expected_prefix else None
    return {
        "python": sys.executable,
        "prefix": str(prefix),
        "base_prefix": str(base_prefix),
        "in_virtualenv": prefix != base_prefix,
        "matches_expected_prefix": expected is None or prefix == expected,
        "python_supported": sys.version_info >= (3, 11),
    }


def configure_environment() -> Path:
    report = environment_report()
    if not report["in_virtualenv"]:
        raise RuntimeError("Select a virtual-environment-backed Jupyter kernel.")
    prefix = Path(sys.prefix).resolve()
    binary_dir = prefix / ("Scripts" if os.name == "nt" else "bin")
    os.environ["VIRTUAL_ENV"] = str(prefix)
    os.environ["PATH"] = os.pathsep.join(
        [str(binary_dir), *os.environ.get("PATH", "").split(os.pathsep)]
    )
    os.environ["PIP_REQUIRE_VIRTUALENV"] = "true"
    return prefix


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the active Steerling notebook kernel.")
    parser.add_argument("--expected-prefix", type=Path)
    args = parser.parse_args()
    report = environment_report(args.expected_prefix)
    for key, value in report.items():
        print(f"{key}: {value}")
    if not report["in_virtualenv"]:
        raise SystemExit("Use a virtual-environment-backed Jupyter kernel.")
    if not report["matches_expected_prefix"]:
        raise SystemExit("The active interpreter does not match --expected-prefix.")
    if not report["python_supported"]:
        raise SystemExit("Python 3.11 or newer is required.")
    configure_environment()


if __name__ == "__main__":
    main()
