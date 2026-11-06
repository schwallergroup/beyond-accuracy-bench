from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal


RubricScore = Literal[0, 1, 2]


@dataclass(frozen=True)
class RubricDefinition:
    """
    Canonical definition for one rubric axis used in judge-based trace analysis.

    The intention is:
    - `definition` explains what the axis measures
    - `score_descriptions` defines the 0/1/2 ordinal scale
    - `look_at` constrains which parts of the trace the judge should focus on
    - `do_not_reward` prevents leakage from neighboring axes
    - `notes` stores extra implementation guidance without forcing prompt usage
    """

    key: str
    display_name: str
    definition: str
    score_descriptions: Dict[RubricScore, str]
    look_at: List[str]
    do_not_reward: List[str]
    notes: List[str]


RUBRIC_ORDER: List[str] = [
    "hypothesis_space_alignment",
    "mechanistic_feature_extraction",
    "predictive_sharpness",
    "feature_to_action_faithfulness",
]


RUBRICS: Dict[str, RubricDefinition] = {
    "hypothesis_space_alignment": RubricDefinition(
        key="hypothesis_space_alignment",
        display_name="Hypothesis-Space Alignment",
        definition=(
            "Whether the trace's stated candidate mechanisms are aligned with the "
            "immediately preceding NLL feedback. This measures whether the model is "
            "tracking the current hypothesis space correctly, not whether it "
            "eventually identifies the correct mechanism."
        ),
        score_descriptions={
            0: (
                "The trace fails to mention the relevant top mechanisms, or discusses "
                "mechanisms that are clearly inconsistent with the NLL feedback."
            ),
            1: (
                "The trace mentions some relevant candidates, but important top "
                "mechanisms are missing, misranked, or replaced by implausible "
                "alternatives."
            ),
            2: (
                "The trace accurately reflects the top candidate set or top candidate "
                "cluster implied by the NLL feedback."
            ),
        },
        look_at=[
            "reflection",
            "hypothesis",
            "candidate_mechanisms",
            "nll_summary",
        ],
        do_not_reward=[
            "Do not reward general mechanistic fluency if the candidate set is not aligned with the NLL feedback.",
            "Do not use eventual correctness or ground-truth identity to score this axis.",
        ],
        notes=[
            "This axis is about belief-state alignment, not final success.",
            "If the implementation passes only a top-k summary instead of full NLLs, score against that summary.",
        ],
    ),
    "mechanistic_feature_extraction": RubricDefinition(
        key="mechanistic_feature_extraction",
        display_name="Mechanistic Feature Extraction",
        definition=(
            "Whether the trace identifies the mechanistically relevant differences "
            "among the current top candidate mechanisms."
        ),
        score_descriptions={
            0: (
                "The trace does not identify meaningful mechanistic differences among "
                "the top candidates."
            ),
            1: (
                "The trace identifies only coarse family-level distinctions, such as "
                "activation vs. deactivation or dimerization vs. product effect."
            ),
            2: (
                "The trace identifies discriminative mechanistic features at the level "
                "of sub-reaction steps, catalyst states, reversibility, active species, "
                "or other specific mechanistic motifs."
            ),
        },
        look_at=[
            "reflection",
            "hypothesis",
            "candidate_mechanisms",
            "mechanism_definitions",
        ],
        do_not_reward=[
            "Do not reward merely listing mechanism names without identifying their differences.",
            "Do not reward family labels alone as a high score.",
        ],
        notes=[
            "This axis asks whether the model extracted discriminative mechanism-level features.",
            "Mechanism definitions may be omitted in early implementations, but scoring is stronger if they are available.",
        ],
    ),
    "predictive_sharpness": RubricDefinition(
        key="predictive_sharpness",
        display_name="Predictive Reasoning (PR)",
        definition=(
            "Whether the trace predicts how the proposed experiment would behave under "
            "different candidate mechanisms."
        ),
        score_descriptions={
            0: (
                "The trace proposes an experiment without giving a concrete prediction "
                "of how different candidate mechanisms would respond."
            ),
            1: (
                "The trace gives a prediction, but only at a coarse family level or in "
                "vague terms such as 'should increase rate' or 'should show inhibition.'"
            ),
            2: (
                "The trace gives contrastive, mechanism-specific predictions tied to "
                "sub-reaction steps, species, equilibria, rate limitation, or clearly "
                "stated observable consequences such as curvature, early slowdown, or "
                "final conversion."
            ),
        },
        look_at=[
            "reflection",
            "hypothesis",
            "suggestion",
            "candidate_mechanisms",
            "mechanism_definitions",
        ],
        do_not_reward=[
            "Do not reward generic statements like 'this will be informative' as a prediction.",
            "Do not reward family-level predictions as a top score unless they are also contrastive and mechanism-specific.",
        ],
        notes=[
            "A score of 2 should usually involve candidate-specific or cluster-specific contrastive predictions.",
            "Observable linkage matters: rates, curvature, plateaus, early slowdown, or final conversion are stronger than abstract claims.",
        ],
    ),
    "feature_to_action_faithfulness": RubricDefinition(
        key="feature_to_action_faithfulness",
        display_name="Discriminative Experiment Design (DED)",
        definition=(
            "Whether the proposed experiment directly tests the mechanistic difference "
            "identified in the reflection or hypothesis."
        ),
        score_descriptions={
            0: (
                "The proposal does not meaningfully probe the mechanistic feature that "
                "the trace claims is important."
            ),
            1: (
                "The proposal is loosely related to the claimed feature, but remains "
                "generic, indirect, or only partially diagnostic."
            ),
            2: (
                "The proposal directly and intentionally probes the specific "
                "mechanistic feature identified in the reasoning."
            ),
        },
        look_at=[
            "reflection",
            "hypothesis",
            "suggestion",
        ],
        do_not_reward=[
            "Do not reward a proposal merely for being extreme or broad if it does not test the claimed feature.",
            "Do not reward reasonable experimental intuition unless it matches the stated mechanistic rationale.",
        ],
        notes=[
            "This axis captures the gap between good verbal reasoning and actually diagnostic experiment design.",
            "It is often useful for distinguishing 'mechanistically articulate but non-diagnostic' traces.",
        ],
    ),
    "top_hypothesis_targeting": RubricDefinition(
        key="top_hypothesis_targeting",
        display_name="Top-Hypothesis Targeting",
        definition=(
            "Whether the proposed experiment is primarily aimed at separating the "
            "top-ranked competing mechanisms at that step."
        ),
        score_descriptions={
            0: (
                "The proposal mainly targets lower-ranked or marginal mechanisms, or "
                "ignores the main top-ranked uncertainty."
            ),
            1: (
                "The proposal is somewhat relevant to the top-ranked competition, but "
                "significant attention is still spent on lower-priority alternatives."
            ),
            2: (
                "The proposal is clearly designed to discriminate the most competitive "
                "top-ranked mechanisms or top-ranked mechanism cluster."
            ),
        },
        look_at=[
            "reflection",
            "hypothesis",
            "candidate_mechanisms",
            "nll_summary",
            "suggestion",
        ],
        do_not_reward=[
            "Do not reward targeting the third- or fourth-ranked mechanism when the top-ranked competition is left unresolved.",
            "Do not reward broad exploration alone as top-hypothesis targeting.",
        ],
        notes=[
            "This axis is distinct from hypothesis alignment: a trace may mention the right candidates but still design an experiment that targets the wrong comparison.",
            "If the implementation uses a top-k cluster rather than strict top-1/top-2, the prompt should state that explicitly.",
        ],
    ),
}


def get_rubric(key: str) -> RubricDefinition:
    try:
        return RUBRICS[key]
    except KeyError as exc:
        valid_keys = ", ".join(RUBRICS)
        raise KeyError(f"Unknown rubric key '{key}'. Valid keys: {valid_keys}") from exc
