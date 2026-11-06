from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from bench.config import LLM_CONFIGS

from .prompts import build_judge_prompt
from .schema import JudgeResult, TraceRecord


class JudgeOutputValidationError(ValueError):
    pass


@dataclass(frozen=True)
class JudgeBackendConfig:
    """
    Runtime configuration for the judge backend.

    Selects the judge model, completion-token limit, and retry count for
    invalid outputs from the configured LLM backend.
    """

    judge_model: str = "gpt-5.4"
    max_retries_on_invalid_output: int = 2
    max_completion_tokens: int = 1024


def _build_llm(model_name: str, max_completion_tokens: int) -> Any:
    if model_name not in LLM_CONFIGS:
        raise ValueError(f"Unknown judge model '{model_name}'.")
    config = LLM_CONFIGS[model_name]
    llm_cls = config["class"]
    params = dict(config["params"])
    params["max_completion_tokens"] = max_completion_tokens
    return llm_cls(**params)


def _coerce_score(value: Any) -> int:
    if value not in (0, 1, 2):
        raise JudgeOutputValidationError(f"Invalid score {value!r}; expected one of 0, 1, 2.")
    return int(value)


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise JudgeOutputValidationError(
            f"Invalid confidence {value!r}; expected a float in [0, 1]."
        ) from exc
    if confidence < 0.0 or confidence > 1.0:
        raise JudgeOutputValidationError(
            f"Invalid confidence {confidence}; expected a float in [0, 1]."
        )
    return confidence


def _coerce_evidence(value: Any) -> List[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise JudgeOutputValidationError(
            "Invalid evidence field; expected a list of strings."
        )
    if len(value) == 0:
        raise JudgeOutputValidationError(
            "Invalid evidence field; expected at least one evidence item."
        )
    return value


def _coerce_rationale(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JudgeOutputValidationError(
            "Invalid rationale field; expected a non-empty string."
        )
    return value.strip()


def parse_judge_output(raw_output: Any) -> Dict[str, Any]:
    """
    Parse raw judge output into a JSON object.

    Supported inputs:
    - already-decoded dict
    - JSON string
    """

    if isinstance(raw_output, dict):
        return raw_output

    if isinstance(raw_output, str):
        text = raw_output.strip()

        candidates = [text]

        fenced_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        candidates.extend(fenced_blocks)

        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            candidates.append(text[first_brace:last_brace + 1])

        last_exc: Optional[json.JSONDecodeError] = None
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_exc = exc
                continue
            if not isinstance(parsed, dict):
                raise JudgeOutputValidationError(
                    "Judge output must decode to a JSON object."
                )
            return parsed

        raise JudgeOutputValidationError(
            "Judge output is not valid JSON."
        ) from last_exc

    raise JudgeOutputValidationError(
        f"Unsupported judge output type {type(raw_output).__name__}; expected dict or JSON string."
    )


def validate_and_build_result(
    trace_record: TraceRecord,
    rubric_key: str,
    judge_model: str,
    raw_output: Any,
) -> JudgeResult:
    """
    Validate one judge response and convert it into a normalized JudgeResult.
    """

    parsed = parse_judge_output(raw_output)

    missing = {"score", "confidence", "evidence", "rationale"} - set(parsed.keys())
    if missing:
        raise JudgeOutputValidationError(
            f"Judge output is missing required fields: {sorted(missing)}"
        )

    result = JudgeResult(
        tag=trace_record.tag,
        sample_id=trace_record.sample_id,
        iteration=trace_record.iteration,
        rubric_key=rubric_key,
        score=_coerce_score(parsed["score"]),
        confidence=_coerce_confidence(parsed["confidence"]),
        evidence=_coerce_evidence(parsed["evidence"]),
        rationale=_coerce_rationale(parsed["rationale"]),
        model_label=trace_record.model_label,
        judge_model=judge_model,
        raw_output=parsed,
    )
    return result


def run_judge_with_retries(
    trace_record: TraceRecord,
    rubric_key: str,
    backend_config: JudgeBackendConfig,
) -> JudgeResult:
    """
    Call the judge model, validate the response, and retry on invalid outputs.

    The judge is asked to return strict JSON. We parse and validate that JSON.
    If the output is missing fields or uses invalid values, we retry a small
    fixed number of times.
    """

    prompt_parts = build_judge_prompt(
        rubric_key=rubric_key,
        trace_payload=trace_record.to_prompt_payload(),
    )
    llm = _build_llm(
        backend_config.judge_model, backend_config.max_completion_tokens
    )

    last_error: Optional[Exception] = None
    for attempt in range(backend_config.max_retries_on_invalid_output + 1):
        response = llm.invoke(
            [
                {"role": "system", "content": prompt_parts.system_prompt},
                {"role": "user", "content": prompt_parts.user_prompt},
            ]
        )
        raw_content = getattr(response, "content", response)
        try:
            return validate_and_build_result(
                trace_record=trace_record,
                rubric_key=rubric_key,
                judge_model=backend_config.judge_model,
                raw_output=raw_content,
            )
        except JudgeOutputValidationError as exc:
            last_error = exc
            if attempt >= backend_config.max_retries_on_invalid_output:
                break

    raise JudgeOutputValidationError(
        f"Judge output remained invalid after {backend_config.max_retries_on_invalid_output + 1} attempts: {last_error}"
    )
