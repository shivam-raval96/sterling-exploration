from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_chat_prompt_ids(tokenizer: Any, prompt: str) -> tuple[list[int], set[int]]:
    """Build the instruct checkpoint's chat prompt without re-tokenizing specials."""
    required = ("start_header_id", "end_header_id", "endofchunk_token_id", "eot_id")
    if any(getattr(tokenizer, name, None) is None for name in required):
        raise ValueError("the tokenizer does not expose the instruct chat tokens")

    prefix = (
        [tokenizer.start_header_id]
        + tokenizer.encode("user", add_special_tokens=False)
        + [tokenizer.end_header_id]
        + tokenizer.encode("\n\n", add_special_tokens=False)
    )
    content = tokenizer.encode(prompt, add_special_tokens=False)
    suffix = (
        [tokenizer.endofchunk_token_id, tokenizer.eot_id, tokenizer.start_header_id]
        + tokenizer.encode("assistant", add_special_tokens=False)
        + [tokenizer.end_header_id]
        + tokenizer.encode("\n\n", add_special_tokens=False)
    )
    content_positions = set(range(len(prefix), len(prefix) + len(content)))
    return prefix + content + suffix, content_positions


def aggregate_concepts(
    records: list[dict[str, Any]], *, concept_type: str, content_only: bool
) -> list[dict[str, Any]]:
    """Aggregate token-level top-k concept rows into a ranked firing list."""
    stats: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"token_firings": 0, "input_ids": set(), "activation_sum": 0.0, "max_activation": 0.0}
    )
    for record in records:
        input_id = int(record["input_index"])
        for token in record["tokens"]:
            if content_only and not token["is_user_content"]:
                continue
            concepts = token[concept_type]
            for concept in concepts:
                entry = stats[int(concept["concept_id"])]
                activation = float(concept["activation"])
                entry["token_firings"] += 1
                entry["input_ids"].add(input_id)
                entry["activation_sum"] += activation
                entry["max_activation"] = max(entry["max_activation"], activation)

    aggregated = []
    for concept_id, entry in stats.items():
        count = entry["token_firings"]
        aggregated.append(
            {
                "concept_id": concept_id,
                "token_firings": count,
                "input_firings": len(entry["input_ids"]),
                "mean_activation": entry["activation_sum"] / count,
                "max_activation": entry["max_activation"],
            }
        )
    return sorted(
        aggregated,
        key=lambda row: (row["input_firings"], row["token_firings"], row["mean_activation"]),
        reverse=True,
    )


def summarize_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        scope: {
            "known": aggregate_concepts(
                records, concept_type="known_concepts", content_only=scope == "user_content"
            ),
            "unknown": aggregate_concepts(
                records, concept_type="unknown_concepts", content_only=scope == "user_content"
            ),
        }
        for scope in ("user_content", "full_chat_input")
    }
