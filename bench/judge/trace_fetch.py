from __future__ import annotations

import os
from typing import Any, Dict, Optional

import dotenv
from bench.proposers.mechanisms import CANDIDATE_MECHANISMS

from .schema import TraceRecord


PROPOSER_CALL_SUBSTR = "AgentProposer.propose"
_WEAVE_CLIENTS: Dict[str, Any] = {}


def _normalize_batch_tag(tag: str) -> str:
    return tag[:-4] if tag.endswith(".tar") else tag


def _load_weave() -> Any:
    """
    Lazy import weave so the judge package can be imported without weave installed.
    """

    import weave  # type: ignore

    return weave


def _init_weave_client(project_name: str = "KineticMechanismBench") -> Any:
    if project_name in _WEAVE_CLIENTS:
        return _WEAVE_CLIENTS[project_name]
    dotenv.load_dotenv(".env")
    weave = _load_weave()
    client = weave.init(project_name=project_name)
    _WEAVE_CLIENTS[project_name] = client
    return client


def _extract_previous_user_message(call_inputs: Dict[str, Any]) -> Optional[str]:
    history = call_inputs.get("history")
    if not history:
        return None
    last_item = history[-1]
    if isinstance(last_item, dict):
        role = last_item.get("role")
        content = last_item.get("content")
        if role == "user" and isinstance(content, str):
            return content
    try:
        role = getattr(last_item, "get", lambda *_: None)("role")
        content = getattr(last_item, "get", lambda *_: None)("content")
        if role == "user" and isinstance(content, str):
            return content
    except Exception:
        pass
    text = str(last_item)
    return text if text else None


def _build_query_expr(tag: str, sample_id: str) -> Dict[str, Any]:
    batch_tag = _normalize_batch_tag(tag)
    return {
        "$and": [
            {"$eq": [{"$getField": "attributes.batch_tag"}, {"$literal": batch_tag}]},
            {"$eq": [{"$getField": "attributes.ground_truth_id"}, {"$literal": sample_id}]},
            {
                "$contains": {
                    "input": {"$getField": "op_name"},
                    "substr": {"$literal": PROPOSER_CALL_SUBSTR},
                    "case_insensitive": False,
                }
            },
        ]
    }


def fetch_trace_record(
    tag: str,
    sample_id: str,
    iteration: int,
    model_label: Optional[str] = None,
    include_raw_response: bool = True,
    project_name: str = "KineticMechanismBench",
) -> TraceRecord:
    """
    Fetch one judgeable trace from Weave.

    Current assumptions:
    - iteration is 1-indexed
    - the relevant reasoning is stored in `output.raw_response`
    - propose calls can be filtered by batch_tag and ground_truth_id
    """

    client = _init_weave_client(project_name=project_name)

    query = {"$expr": _build_query_expr(tag=tag, sample_id=sample_id)}
    calls = list(
        client.get_calls(
            query=query,
            columns=["id", "inputs", "output", "attributes", "summary"],
        )
    )

    if not calls:
        raise ValueError(f"No Weave calls found for tag={tag}, sample_id={sample_id}")

    sorted_calls = sorted(
        calls,
        key=lambda call: int(getattr(call, "inputs", {}).get("step", 0)),
    )

    target_call = None
    for call in sorted_calls:
        step = int(getattr(call, "inputs", {}).get("step", 0))
        if step == iteration:
            target_call = call
            break

    if target_call is None:
        raise ValueError(
            f"No Weave call found for tag={tag}, sample_id={sample_id}, iteration={iteration}"
        )

    output = getattr(target_call, "output", {}) or {}
    inputs = getattr(target_call, "inputs", {}) or {}
    raw_response = str(output.get("raw_response", "") or "")

    record = TraceRecord(
        tag=tag,
        sample_id=sample_id,
        iteration=iteration,
        model_label=model_label,
        previous_user_message=_extract_previous_user_message(inputs),
        mechanism_definitions=CANDIDATE_MECHANISMS,
        raw_response=raw_response if include_raw_response else None,
        source_call_id=str(getattr(target_call, "id", "")),
        extra={
            "inputs_step": inputs.get("step"),
            "attributes": getattr(target_call, "attributes", {}),
        },
    )
    return record
