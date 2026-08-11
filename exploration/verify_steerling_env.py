from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the dedicated Steerling environment.")
    parser.add_argument("--env", type=Path, default=Path(".venv-steerling"))
    args = parser.parse_args()
    env_dir = args.env.resolve()
    python = env_dir / "bin/python"
    program = """
import json
import sys
import torch
import transformers
import steerling
from steerling import GenerationConfig, SteerlingGenerator
print(json.dumps({
    'prefix': sys.prefix,
    'executable': sys.executable,
    'python': sys.version.split()[0],
    'torch': torch.__version__,
    'transformers': transformers.__version__,
    'steerling': steerling.__file__,
    'generator': SteerlingGenerator.__name__,
    'config': GenerationConfig.__name__,
}))
"""
    completed = subprocess.run(
        [str(python), "-c", program], check=True, capture_output=True, text=True
    )
    report = json.loads(completed.stdout)
    if Path(report["prefix"]).resolve() != env_dir:
        raise SystemExit(f"wrong prefix: {report['prefix']} != {env_dir}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
