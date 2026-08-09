from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class HarmfulComplianceJudgment(BaseModel):
    harmful_compliance: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=500)


def judge_response(
    client: Any,
    *,
    model: str,
    reasoning_effort: str,
    instructions: str,
    prompt: str,
    target: str,
    response: str,
) -> tuple[HarmfulComplianceJudgment, dict[str, Any]]:
    result = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        store=False,
        instructions=instructions,
        input=(
            f"USER REQUEST:\n{prompt}\n\n"
            f"REFERENCE TARGET (context only):\n{target}\n\n"
            f"ASSISTANT RESPONSE:\n{response}"
        ),
        text_format=HarmfulComplianceJudgment,
    )
    if result.output_parsed is None:
        raise ValueError("judge returned no parsed structured output")
    usage = result.usage.model_dump() if result.usage is not None else None
    return result.output_parsed, {"response_id": result.id, "usage": usage}


def load_judge_instructions(path: str | Path) -> str:
    return Path(path).read_text().strip()
