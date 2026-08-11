from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

MODEL_ID = "guidelabs/steerling-8b-instruct"
MODEL_REVISION = "6e5a87d00d45348001810c30fe9bd65110b69fc2"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "model_inspection.json"


def tensor_record(name: str, tensor: Any) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dimensions": tensor.dim(),
        "numel": tensor.numel(),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "element_size_bytes": tensor.element_size(),
        "storage_bytes": tensor.numel() * tensor.element_size(),
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
    }


def module_records(model: Any) -> list[dict[str, Any]]:
    records = []
    for name, module in model.named_modules():
        direct_parameters = [
            tensor_record(parameter_name, parameter)
            for parameter_name, parameter in module.named_parameters(recurse=False)
        ]
        direct_buffers = [
            tensor_record(buffer_name, buffer)
            for buffer_name, buffer in module.named_buffers(recurse=False)
        ]
        records.append(
            {
                "name": name or "<root>",
                "depth": 0 if not name else name.count(".") + 1,
                "type": type(module).__name__,
                "qualified_type": f"{type(module).__module__}.{type(module).__name__}",
                "direct_parameter_count": sum(row["numel"] for row in direct_parameters),
                "direct_buffer_count": sum(row["numel"] for row in direct_buffers),
                "parameters": direct_parameters,
                "buffers": direct_buffers,
            }
        )
    return records


def vocabulary_record(tokenizer: Any, config: Any) -> dict[str, Any]:
    vocabulary = tokenizer.get_vocab()
    special_tokens = {}
    for name, value in tokenizer.special_tokens_map.items():
        values = value if isinstance(value, list) else [value]
        tokens = [str(token) for token in values]
        special_tokens[name] = {
            "tokens": tokens,
            "token_ids": tokenizer.convert_tokens_to_ids(tokens),
        }
    return {
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
        "tokenizer_length": len(tokenizer),
        "get_vocab_size": len(vocabulary),
        "config_vocab_size": getattr(config, "vocab_size", None),
        "added_vocab_size": len(tokenizer.get_added_vocab()),
        "special_tokens": special_tokens,
        "special_token_ids": list(tokenizer.all_special_ids),
        "special_token_strings": list(tokenizer.all_special_tokens),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def inspect(model: Any, tokenizer: Any) -> dict[str, Any]:
    parameters = list(model.named_parameters())
    buffers = list(model.named_buffers())
    modules = module_records(model)
    return {
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "printed_structure": str(model),
            "config": model.config.to_dict(),
        },
        "vocabulary": vocabulary_record(tokenizer, model.config),
        "totals": {
            "parameters": sum(parameter.numel() for _, parameter in parameters),
            "trainable_parameters": sum(
                parameter.numel() for _, parameter in parameters if parameter.requires_grad
            ),
            "parameter_storage_bytes": sum(
                parameter.numel() * parameter.element_size() for _, parameter in parameters
            ),
            "buffers": sum(buffer.numel() for _, buffer in buffers),
            "module_count": len(modules),
            "module_type_counts": dict(
                sorted(Counter(row["type"] for row in modules).items())
            ),
            "parameter_dtype_counts": dict(
                sorted(Counter(str(parameter.dtype) for _, parameter in parameters).items())
            ),
            "parameter_device_counts": dict(
                sorted(Counter(str(parameter.device) for _, parameter in parameters).items())
            ),
        },
        "modules": modules,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the pinned Steerling model.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    common = {
        "revision": MODEL_REVISION,
        "trust_remote_code": True,
    }
    if args.cache_dir is not None:
        common["cache_dir"] = str(args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **common)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        **common,
    )
    model.eval()

    print(model)
    result = inspect(model, tokenizer)
    atomic_json(args.output.resolve(), result)
    print(f"Saved JSON: {args.output.resolve()}")


if __name__ == "__main__":
    main()
