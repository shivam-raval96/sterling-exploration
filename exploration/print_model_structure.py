from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

MODEL_ID = "guidelabs/steerling-8b-instruct"
MODEL_REVISION = "6e5a87d00d45348001810c30fe9bd65110b69fc2"


def human_size(byte_count: float) -> str:
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def shape_text(shape: Any) -> str:
    return "×".join(str(dimension) for dimension in shape) or "scalar"


def print_config(config: Any) -> None:
    print(f"Model: {MODEL_ID}")
    print(f"Revision: {MODEL_REVISION}")
    print(f"Config class: {type(config).__module__}.{type(config).__name__}")
    print(f"Architectures: {getattr(config, 'architectures', None)}")
    print("\nCore architecture")
    fields = (
        "model_type",
        "vocab_size",
        "n_layers",
        "n_head",
        "n_kv_heads",
        "n_embd",
        "intermediate_size",
        "block_size",
        "diff_block_size",
        "inject_layer",
        "n_concepts",
        "n_unknown_concepts",
        "concept_dim",
        "topk_known",
        "unknown_topk",
        "factorize_unknown",
        "factorize_rank",
        "use_epsilon_correction",
        "torch_dtype",
    )
    for field in fields:
        if hasattr(config, field):
            print(f"  {field}: {getattr(config, field)}")


def parameter_summary(model: Any) -> None:
    import torch

    parameters = list(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in parameters)
    trainable = sum(parameter.numel() for _, parameter in parameters if parameter.requires_grad)
    buffers = sum(buffer.numel() for _, buffer in model.named_buffers())
    dtype_counts = Counter(str(parameter.dtype).removeprefix("torch.") for _, parameter in parameters)
    actual_bytes = sum(parameter.numel() * parameter.element_size() for _, parameter in parameters)

    print("\nParameter summary")
    print(f"  parameters: {total:,}")
    print(f"  trainable: {trainable:,}")
    print(f"  buffers: {buffers:,}")
    print(f"  current tensor storage: {human_size(actual_bytes)}")
    print(f"  estimated BF16/FP16: {human_size(total * 2)}")
    print(f"  estimated FP32: {human_size(total * 4)}")
    print(f"  estimated INT8: {human_size(total)}")
    print(f"  dtypes: {json.dumps(dtype_counts, sort_keys=True)}")
    devices = Counter(str(parameter.device) for _, parameter in parameters)
    print(f"  devices: {json.dumps(devices, sort_keys=True)}")
    if any(parameter.device.type == "meta" for _, parameter in parameters):
        print("  note: meta tensors report logical size; no weight storage is allocated")
    if not torch.cuda.is_available():
        print("  CUDA: unavailable")


def print_module_tree(model: Any, max_depth: int) -> None:
    module_types = Counter(type(module).__name__ for module in model.modules())
    print("\nModule type counts")
    for module_type, count in module_types.most_common():
        print(f"  {module_type}: {count:,}")

    print(f"\nModule tree (maximum depth {max_depth})")
    for name, module in model.named_modules():
        depth = 0 if not name else name.count(".") + 1
        if depth > max_depth:
            continue
        direct_parameters = list(module.named_parameters(recurse=False))
        direct_count = sum(parameter.numel() for _, parameter in direct_parameters)
        label = name or "<root>"
        suffix = f" — direct params: {direct_count:,}" if direct_count else ""
        print(f"{'  ' * depth}{label}: {type(module).__name__}{suffix}")
        for parameter_name, parameter in direct_parameters:
            print(
                f"{'  ' * (depth + 1)}@{parameter_name}: "
                f"shape={shape_text(parameter.shape)}, dtype={parameter.dtype}, "
                f"device={parameter.device}, params={parameter.numel():,}"
            )


def load_config() -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )


def build_model(config: Any, mode: str, device_map: str) -> Any:
    from transformers import AutoModel

    if mode == "meta":
        from accelerate import init_empty_weights

        with init_empty_weights():
            return AutoModel.from_config(config, trust_remote_code=True)
    return AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=device_map,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the pinned Steerling model structure.")
    parser.add_argument("--mode", choices=("config", "meta", "loaded"), default="config")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()
    if args.max_depth < 0:
        parser.error("--max-depth must be non-negative")

    config = load_config()
    print_config(config)
    if args.mode == "config":
        print("\nUse --mode meta for the full module tree without loading weights.")
        return
    model = build_model(config, args.mode, args.device_map)
    print(f"\nModel class: {type(model).__module__}.{type(model).__name__}")
    parameter_summary(model)
    print_module_tree(model, args.max_depth)


if __name__ == "__main__":
    main()
