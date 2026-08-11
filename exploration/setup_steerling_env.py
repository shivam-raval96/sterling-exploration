from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

KERNEL_NAME = "sterling-exploration"
STEERLING_SOURCE = (
    "steerling @ git+https://github.com/guidelabs/steerling.git"
    "@f34ffa89e46969445f3cf6e7c885e9623a2047c1"
)
DEPENDENCIES = (
    "accelerate>=1.2,<2",
    "huggingface-hub>=0.20",
    "ipykernel>=6,<8",
    "numpy>=2.3,<3",
    "pandas>=2.2,<3",
    "pyarrow>=15,<22",
    "pydantic>=2.10,<3",
    "safetensors>=0.4",
    "tiktoken>=0.8,<0.9",
    "torch>=2.8,<3",
    "transformers>=4.48,<5",
)


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def find_python() -> str:
    for candidate in ("python3.13", "python3.12", "python3.11"):
        executable = shutil.which(candidate)
        if executable:
            return executable
    raise RuntimeError("Python 3.11 or newer was not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the Steerling notebook environment.")
    parser.add_argument("--env", type=Path, default=Path(".venv-steerling"))
    args = parser.parse_args()
    env_dir = args.env.resolve()
    if not env_dir.exists():
        run(find_python(), "-m", "venv", str(env_dir))
    python = env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    run(str(python), "-m", "pip", "install", "--upgrade", "pip")
    run(str(python), "-m", "pip", "install", *DEPENDENCIES)
    run(
        str(python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--ignore-requires-python",
        STEERLING_SOURCE,
    )
    run(
        str(python),
        "-m",
        "ipykernel",
        "install",
        "--user",
        "--name",
        KERNEL_NAME,
        "--display-name",
        "Python (sterling-exploration)",
    )
    print(f"Environment ready: {env_dir}")
    print(f"Kernel registered: {KERNEL_NAME}")


if __name__ == "__main__":
    main()
