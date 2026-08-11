from __future__ import annotations

import subprocess
import sys

STEERLING_SOURCE = (
    "steerling @ git+https://github.com/guidelabs/steerling.git"
    "@f34ffa89e46969445f3cf6e7c885e9623a2047c1"
)
PORTABLE_DEPENDENCIES = (
    "accelerate>=1.2,<2",
    "huggingface-hub>=0.20",
    "numpy>=2.3,<3",
    "pandas>=2.2,<3",
    "pyarrow>=15,<22",
    "pydantic>=2.10,<3",
    "safetensors>=0.4",
    "tiktoken>=0.8,<0.9",
    "transformers>=4.48,<5",
)


def run_pip(*arguments: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", *arguments], check=True)


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Install the CUDA-enabled PyTorch build supplied by your notebook platform first."
        ) from exc
    print(f"Using existing PyTorch {torch.__version__} from {torch.__file__}")
    run_pip("install", *PORTABLE_DEPENDENCIES)
    run_pip("install", "--no-deps", "--ignore-requires-python", STEERLING_SOURCE)

    import steerling
    from steerling import GenerationConfig, SteerlingGenerator

    print(f"Steerling import succeeded: {steerling.__file__}")
    print(f"Generator: {SteerlingGenerator.__name__}; config: {GenerationConfig.__name__}")


if __name__ == "__main__":
    main()
