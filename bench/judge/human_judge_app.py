from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

# Streamlit executes this file directly, so include the repository root for
# package imports and the public annotation entry point's shared helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from bench.judge.rubrics import get_rubric
from bench.judge.trace_io import (
    load_trace_record,
    manifest_path_for_trace_dir,
)
from human_judge_reasoning_traces import (
    append_result_csv,
    append_result_jsonl,
    build_row_result_key,
    ensure_output_layout,
    HUMAN_RUBRICS,
    iter_work_items,
    load_completed_keys,
    output_dir_for_annotator,
    read_manifest,
    resolve_annotator_id,
    validate_and_build_human_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--annotator-id", default=None)
    parser.add_argument("--output-root", default="bench/judge/human_runs")
    parser.add_argument("--rubrics", nargs="+", default=HUMAN_RUBRICS)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mechanism_reference(definitions: Dict[str, object]) -> str:
    def sort_key(key: str) -> Tuple[int, str]:
        if key.startswith("M") and key[1:].isdigit():
            return int(key[1:]), key
        return 10**9, key

    return "\n".join(
        f"{key}: {definitions[key]}" for key in sorted(definitions, key=sort_key)
    )


def render_scoring_card(rubric_key: str) -> int | None:
    rubric = get_rubric(rubric_key)
    st.subheader(f"Scoring card — {rubric.display_name}")
    st.caption(rubric.definition)
    for score, description in sorted(rubric.score_descriptions.items(), reverse=True):
        st.markdown(f"**{score}** — {description}")
    if rubric.do_not_reward:
        st.caption(f"Important: {rubric.do_not_reward[0]}")
    return st.selectbox(
        "Your score",
        options=[None, 2, 1, 0],
        format_func=lambda value: "Choose a score" if value is None else str(value),
        key=f"score_{rubric_key}_{st.session_state['item_key']}",
    )


def main() -> None:
    st.set_page_config(page_title="Reasoning Trace Annotation", layout="wide")
    st.markdown(
        """
        <style>
        .stTextArea textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
        div[data-testid="stSidebar"] { min-width: 330px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    try:
        args = parse_args()
        manifest_path = manifest_path_for_trace_dir(args.trace_dir)
        rows = read_manifest(str(manifest_path))
        rubrics = args.rubrics
        annotator_id = resolve_annotator_id(args.output_root, args.annotator_id)
        output_dir = output_dir_for_annotator(args.output_root, annotator_id)
    except (FileNotFoundError, ValueError, SystemExit) as exc:
        st.error(f"Could not start annotation: {exc}")
        st.stop()

    layout = ensure_output_layout(str(output_dir))
    judge_model = f"human:{annotator_id}"
    if not (layout["base"] / "manifest.csv").exists():
        shutil.copyfile(manifest_path, layout["base"] / "manifest.csv")
    if not (layout["base"] / "run_config.json").exists():
        with open(layout["base"] / "run_config.json", "w") as f:
            json.dump(
                {
                    "interface": "streamlit",
                    "judge_model": judge_model,
                    "annotator_id": annotator_id,
                    "trace_dir": args.trace_dir,
                    "manifest": str(manifest_path),
                    "rubrics": rubrics,
                },
                f,
                indent=2,
            )
    completed = load_completed_keys(layout["jsonl"])
    work_items = list(iter_work_items(rows, rubrics))
    pending = [
        (row, rubric_key)
        for row, rubric_key in work_items
        if build_row_result_key(row, rubric_key, judge_model) not in completed
    ]

    if not pending:
        st.success("This assignment is complete. All ratings have been saved.")
        st.stop()

    row, rubric_key = pending[0]
    item_number = len(work_items) - len(pending) + 1
    item_key = "|".join((row["tag"], row["sample_id"], row["iteration"], rubric_key))
    st.session_state["item_key"] = item_key
    if st.session_state.get("started_item_key") != item_key:
        st.session_state["started_item_key"] = item_key
        st.session_state["started_at"] = utc_now_iso()

    record = load_trace_record(args.trace_dir, row)
    rubric = get_rubric(rubric_key)

    st.header("Reasoning Trace Annotation")
    st.caption(f"Rating {item_number} of {len(work_items)}")
    st.progress(len(completed) / len(work_items) if work_items else 1.0)
    st.caption("The source model, sample ID, and ground truth are intentionally hidden.")
    st.caption(f"Your ratings are saved locally in: {layout['csv']}")
    content_column, rating_column = st.columns([3, 2], gap="large")

    with content_column:
        has_feedback = "nll_summary" in rubric.look_at
        with st.container(height=520, border=True):
            if has_feedback:
                st.markdown("**Experimental Feedback Available to the Model**")
                st.text_area(
                    "Feedback",
                    value=record.previous_user_message or "",
                    height=120,
                    disabled=True,
                    label_visibility="collapsed",
                )
            st.markdown("**Model Reasoning to Evaluate**")
            st.text_area(
                "Reasoning",
                value=record.raw_response or "",
                height=350 if has_feedback else 470,
                disabled=True,
                label_visibility="collapsed",
            )

        with st.expander("Reference: Candidate Mechanism Definitions", expanded=False):
            st.code(mechanism_reference(record.mechanism_definitions or {}), language=None)

    with rating_column:
        with st.form("rating_form", clear_on_submit=True):
            score = render_scoring_card(rubric_key)
            submitted = st.form_submit_button("Save and show next rating", type="primary")
        if submitted:
            if score is None:
                st.error("Choose 0, 1, or 2 before saving.")
            else:
                raw_output = {
                    "score": score,
                    "evidence": ["Human score-only judgment; evidence not collected."],
                    "rationale": "Human score-only judgment; rationale not collected.",
                    "judge_id": annotator_id,
                    "started_at": st.session_state["started_at"],
                    "completed_at": utc_now_iso(),
                }
                result = validate_and_build_human_result(
                    trace_record=record,
                    rubric_key=rubric_key,
                    judge_model=judge_model,
                    raw_output=raw_output,
                )
                append_result_jsonl(result, layout["jsonl"])
                append_result_csv(result, layout["csv"])
                st.rerun()


if __name__ == "__main__":
    main()
