from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _to_plain_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_python(v) for v in value]
    if isinstance(value, tuple):
        return [_to_plain_python(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        if hasattr(value, "items"):
            return {str(k): _to_plain_python(v) for k, v in value.items()}
    except Exception:
        pass
    return str(value)


@dataclass
class TraceRecord:
    """
    Normalized trace unit consumed by the judge pipeline.

    One TraceRecord corresponds to one `(tag, sample_id, iteration)` item.
    """

    tag: str
    sample_id: str
    iteration: int
    model_label: Optional[str] = None
    previous_user_message: Optional[str] = None
    mechanism_definitions: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = None
    source_call_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "previous_user_message": self.previous_user_message,
        }
        if self.mechanism_definitions is not None:
            payload["mechanism_definitions"] = self.mechanism_definitions
        if self.raw_response is not None:
            payload["raw_response"] = self.raw_response
        return payload

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain_python(
            {
                "tag": self.tag,
                "sample_id": self.sample_id,
                "iteration": self.iteration,
                "model_label": self.model_label,
                "previous_user_message": self.previous_user_message,
                "mechanism_definitions": self.mechanism_definitions,
                "raw_response": self.raw_response,
                "source_call_id": self.source_call_id,
                "extra": self.extra,
            }
        )


@dataclass
class JudgeResult:
    """
    Parsed result for one `(trace, rubric)` judge call.
    """

    tag: str
    sample_id: str
    iteration: int
    rubric_key: str
    score: int
    confidence: Optional[float]
    evidence: List[str]
    rationale: str
    model_label: Optional[str] = None
    judge_model: Optional[str] = None
    raw_output: Optional[Dict[str, Any]] = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "tag": self.tag,
            "sample_id": self.sample_id,
            "iteration": self.iteration,
            "model_label": self.model_label,
            "rubric_key": self.rubric_key,
            "score": self.score,
            "confidence": self.confidence,
            "evidence": " | ".join(self.evidence),
            "rationale": self.rationale,
            "judge_model": self.judge_model,
        }
