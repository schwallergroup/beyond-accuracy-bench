from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .rubrics import RubricDefinition, get_rubric


@dataclass(frozen=True)
class JudgePromptParts:
    """
    Prompt container for one rubric-level judge call.

    The intended use is:
    - `system_prompt` for the invariant evaluator instructions
    - `user_prompt` for the rubric-specific task plus trace payload
    """

    system_prompt: str
    user_prompt: str


DEFAULT_SYSTEM_PROMPT = """You are evaluating one rubric dimension of scientific reasoning quality.

You must judge only the requested rubric axis.
Use only the information provided in the trace payload and rubric instructions.
Do not use outside knowledge of the ground truth beyond what is explicitly given.
Return strict JSON only.
"""


def _format_trace_payload(trace_payload: Dict[str, Any]) -> str:
    """
    Pretty-print the trace payload that will be shown to the judge.

    For now we keep this generic. Later, this can be replaced by a stricter
    formatter that orders fields and trims long NLL summaries or history.
    """

    return json.dumps(trace_payload, indent=2, ensure_ascii=False, sort_keys=True)


def build_rubric_user_prompt(
    rubric: RubricDefinition,
    trace_payload: Dict[str, Any],
) -> str:
    """
    Build a single-axis judge prompt.

    The current design intentionally follows the repo decision:
    - one rubric axis per prompt
    - explicit 0/1/2 scale
    - explicit 'do not reward' guidance
    - structured JSON output with score/confidence/evidence/rationale
    """

    look_at_lines = "\n".join(f"- {field}" for field in rubric.look_at)
    do_not_reward_lines = "\n".join(f"- {item}" for item in rubric.do_not_reward)
    score_lines = "\n".join(
        f"- {score}: {description}"
        for score, description in rubric.score_descriptions.items()
    )
    note_lines = "\n".join(f"- {note}" for note in rubric.notes)

    return f"""Task:
Score the trace on the rubric dimension "{rubric.display_name}".

Definition:
{rubric.definition}

Scoring rubric:
{score_lines}

Focus your judgment on these parts of the trace:
{look_at_lines}

Do not reward:
{do_not_reward_lines}

Additional notes:
{note_lines}
Return JSON with exactly these fields:
{{
  "score": 0 or 1 or 2,
  "confidence": 0.0 to 1.0,
  "evidence": ["short quote or tight paraphrase 1", "short quote or tight paraphrase 2"],
  "rationale": "1-3 sentences explaining why the evidence maps to the rubric score"
}}

Trace payload:
{_format_trace_payload(trace_payload)}
"""


def build_judge_prompt(
    rubric_key: str,
    trace_payload: Dict[str, Any],
) -> JudgePromptParts:
    rubric = get_rubric(rubric_key)
    return JudgePromptParts(
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_prompt=build_rubric_user_prompt(
            rubric=rubric,
            trace_payload=trace_payload,
        ),
    )
