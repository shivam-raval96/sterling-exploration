from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .artifacts import config_fingerprint, make_run_id


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    required = {"description", "run_mode"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"missing config fields: {sorted(missing)}")
    if config["run_mode"] not in {"fresh", "resume"}:
        raise ValueError("run_mode must be fresh or resume")
    if "model" in config:
        if not {"id", "revision"} <= config["model"].keys():
            raise ValueError("nested model config requires id and revision")
    elif not {"model_id", "model_revision"} <= config.keys():
        raise ValueError("config requires model/model revision fields")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a run config without contacting Modal")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    config["fingerprint"] = config_fingerprint(config)
    print(json.dumps({"run_id": make_run_id(config["description"]), "config": config}, indent=2))


if __name__ == "__main__":
    main()
