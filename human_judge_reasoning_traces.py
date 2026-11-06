from __future__ import annotations

import argparse
import csv
import json
import os
import re
import secrets
import shutil
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from bench.judge.rubrics import RubricDefinition, get_rubric
from bench.judge.schema import JudgeResult, TraceRecord
from bench.judge.trace_io import (
    load_trace_record,
    manifest_path_for_trace_dir,
)


ResultKey = Tuple[str, str, int, str, str]
HUMAN_RUBRICS = ["feature_to_action_faithfulness", "predictive_sharpness"]
ANNOTATOR_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
ANSI_CYAN_BOLD = "\033[1;36m"
ANSI_RESET = "\033[0m"


class HumanJudgeOutputValidationError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run terminal-based human judging over the same trace content used by the LLM judge."
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        required=True,
        help=(
            "Directory containing downloaded traces/*.json. The bundled ../manifest.csv "
            "is used automatically."
        ),
    )
    parser.add_argument(
        "--annotator-id",
        type=str,
        default=None,
        help=(
            "Optional anonymous annotator identifier. When omitted, the program "
            "generates and locally remembers one for this assignment."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="bench/judge/human_runs",
        help=(
            "Parent directory for annotation runs. Default: bench/judge/human_runs. "
            "The annotator ID determines the final directory name."
        ),
    )
    parser.add_argument(
        "--rubrics",
        nargs="+",
        default=HUMAN_RUBRICS,
        help=f"Rubric keys to score. Default: {' '.join(HUMAN_RUBRICS)}",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip trace/rubric items already present in results/human_judge_results.jsonl.",
    )
    parser.add_argument(
        "--show-progress-metadata",
        action="store_true",
        help=(
            "Show tag/sample/model metadata outside the judge content. Off by default so the "
            "human sees only the same substantive content as the LLM judge, reformatted."
        ),
    )
    return parser.parse_args()


def output_dir_for_annotator(output_root: str, annotator_id: str) -> Path:
    """Return the run directory while preventing an ID from escaping output_root."""

    if not ANNOTATOR_ID_RE.fullmatch(annotator_id):
        raise ValueError(
            "Invalid annotator ID. Use only letters, numbers, underscores, and hyphens, "
            "starting with a letter or number."
        )
    return Path(output_root) / annotator_id


def resolve_annotator_id(output_root: str, supplied_id: Optional[str]) -> str:
    """Return a stable local anonymous ID, generating one on the first launch."""

    if supplied_id is not None:
        output_dir_for_annotator(output_root, supplied_id)
        return supplied_id

    root = Path(output_root)
    marker = root / ".active_annotator_id"
    if marker.exists():
        annotator_id = marker.read_text().strip()
        output_dir_for_annotator(output_root, annotator_id)
        return annotator_id

    root.mkdir(parents=True, exist_ok=True)
    annotator_id = f"annotator_{secrets.token_hex(6)}"
    marker.write_text(f"{annotator_id}\n")
    return annotator_id


def read_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"tag", "sample_id", "iteration"}
    missing = required.difference(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    return rows


def ensure_output_layout(output_dir: str) -> Dict[str, Path]:
    base = Path(output_dir)
    results_dir = base / "results"
    base.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "results": results_dir,
        "jsonl": results_dir / "human_judge_results.jsonl",
        "csv": results_dir / "human_judge_results.csv",
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _section(title: str) -> str:
    rule = "=" * 88
    colored_title = (
        f"{ANSI_CYAN_BOLD}{title}{ANSI_RESET}" if sys.stdout.isatty() else title
    )
    return f"\n{rule}\n{colored_title}\n{rule}\n"


def clear_terminal() -> None:
    """Clear the visible terminal before each item without affecting redirected output."""

    if sys.stdout.isatty():
        print("\033[2J\033[H", end="", flush=True)


def _indent_block(text: str, prefix: str = "  ") -> str:
    if not text:
        return ""
    return textwrap.indent(text.rstrip(), prefix)


def _format_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_structured_value(value: Any, indent: int = 0) -> str:
    """
    Render nested dictionaries/lists in a stable, YAML-like form without changing
    the underlying content shown to the judge.
    """

    pad = " " * indent
    child_indent = indent + 2

    if isinstance(value, dict):
        lines: List[str] = []
        for key in sorted(value, key=lambda item: str(item)):
            child = value[key]
            if isinstance(child, (dict, list, tuple)):
                lines.append(f"{pad}{key}:")
                lines.append(_format_structured_value(child, child_indent))
            else:
                lines.append(f"{pad}{key}: {_format_scalar(child)}")
        return "\n".join(line for line in lines if line is not None)

    if isinstance(value, (list, tuple)):
        lines = []
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                lines.append(f"{pad}-")
                lines.append(_format_structured_value(item, child_indent))
            else:
                lines.append(f"{pad}- {_format_scalar(item)}")
        return "\n".join(lines)

    return f"{pad}{_format_scalar(value)}"


def format_mechanism_definitions(definitions: Dict[str, Any]) -> str:
    """Render M1, M2, ..., M20 in numeric order for quick reference."""

    def sort_key(item: str) -> Tuple[int, str]:
        match = re.fullmatch(r"M(\d+)", str(item))
        return (int(match.group(1)), str(item)) if match else (float("inf"), str(item))

    return "\n".join(
        f"{key}: {_format_scalar(definitions[key])}"
        for key in sorted(definitions, key=sort_key)
    )


def format_human_scoring_card(rubric: RubricDefinition) -> str:
    score_lines = "\n".join(
        f"{score}: {description}"
        for score, description in rubric.score_descriptions.items()
    )
    important_note = rubric.do_not_reward[0] if rubric.do_not_reward else ""

    return "\n".join(
        [
            f"Scoring dimension: {rubric.display_name}",
            "",
            "What to assess:",
            rubric.definition,
            "",
            "Score:",
            score_lines,
            "",
            f"Important: {important_note}",
        ]
    )


def format_human_judge_view(
    rubric: RubricDefinition,
    trace_payload: Dict[str, Any],
) -> str:
    parts = [
        _section("Scoring Card"),
        format_human_scoring_card(rubric),
    ]

    # The preceding user message contains the NLL feedback. It is evidence for
    # belief-state rubrics, but is not needed for the other default axes.
    if (
        "nll_summary" in rubric.look_at
        and "previous_user_message" in trace_payload
    ):
        parts.extend(
            [
                _section("Experimental Feedback Available to the Model"),
                _format_scalar(trace_payload.get("previous_user_message")).rstrip(),
            ]
        )

    if "raw_response" in trace_payload:
        parts.extend(
            [
                _section("Model Reasoning to Evaluate"),
                _format_scalar(trace_payload.get("raw_response")).rstrip(),
            ]
        )

    if "mechanism_definitions" in trace_payload:
        parts.extend(
            [
                _section("Reference: Candidate Mechanism Definitions"),
                format_mechanism_definitions(trace_payload.get("mechanism_definitions")).rstrip(),
            ]
        )

    return "\n".join(parts).rstrip() + "\n"


def _result_key_from_dict(row: Dict[str, Any]) -> Optional[ResultKey]:
    try:
        return (
            str(row["tag"]),
            str(row["sample_id"]),
            int(row["iteration"]),
            str(row["rubric_key"]),
            str(row["judge_model"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_completed_keys(results_path: Path) -> Set[ResultKey]:
    completed: Set[ResultKey] = set()
    if not results_path.exists():
        return completed
    with open(results_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            key = _result_key_from_dict(json.loads(line))
            if key is not None:
                completed.add(key)
    return completed


def append_result_jsonl(result: JudgeResult, out_path: Path) -> None:
    with open(out_path, "a") as f:
        f.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")


def append_result_csv(result: JudgeResult, out_path: Path) -> None:
    row = result.to_row()
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _coerce_score(value: Any) -> int:
    if value not in (0, 1, 2):
        raise HumanJudgeOutputValidationError(
            f"Invalid score {value!r}; expected one of 0, 1, 2."
        )
    return int(value)


def _coerce_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise HumanJudgeOutputValidationError(
            f"Invalid confidence {value!r}; expected a float in [0, 1]."
        ) from exc
    if confidence < 0.0 or confidence > 1.0:
        raise HumanJudgeOutputValidationError(
            f"Invalid confidence {confidence}; expected a float in [0, 1]."
        )
    return confidence


def _coerce_evidence(value: Any) -> List[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise HumanJudgeOutputValidationError(
            "Invalid evidence field; expected a list of strings."
        )
    if len(value) == 0:
        raise HumanJudgeOutputValidationError(
            "Invalid evidence field; expected at least one evidence item."
        )
    return value


def _coerce_rationale(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HumanJudgeOutputValidationError(
            "Invalid rationale field; expected a non-empty string."
        )
    return value.strip()


def validate_and_build_human_result(
    trace_record: TraceRecord,
    rubric_key: str,
    judge_model: str,
    raw_output: Dict[str, Any],
) -> JudgeResult:
    missing = {"score", "evidence", "rationale"} - set(raw_output.keys())
    if missing:
        raise HumanJudgeOutputValidationError(
            f"Human judge output is missing required fields: {sorted(missing)}"
        )

    return JudgeResult(
        tag=trace_record.tag,
        sample_id=trace_record.sample_id,
        iteration=trace_record.iteration,
        rubric_key=rubric_key,
        score=_coerce_score(raw_output["score"]),
        confidence=_coerce_confidence(raw_output.get("confidence")),
        evidence=_coerce_evidence(raw_output["evidence"]),
        rationale=_coerce_rationale(raw_output["rationale"]),
        model_label=trace_record.model_label,
        judge_model=judge_model,
        raw_output=raw_output,
    )


def prompt_line(label: str, allow_blank: bool = False) -> str:
    while True:
        value = input(label).strip()
        if value or allow_blank:
            return value
        print("Value is required.")


def build_score_only_human_output(score_text: str) -> Dict[str, Any]:
    return {
        "score": int(score_text),
        "evidence": ["Human score-only judgment; evidence not collected."],
        "rationale": "Human score-only judgment; rationale not collected.",
    }


def collect_human_output() -> Dict[str, Any]:
    score_text = prompt_line("score [0/1/2]: ")
    return build_score_only_human_output(score_text)


def confirm_or_edit(raw_output: Dict[str, Any]) -> Dict[str, Any]:
    while True:
        print("\nParsed judgment:")
        print(_indent_block(_format_structured_value(raw_output), "  "))
        command = prompt_line("Save? [y/edit]: ").lower()
        if command in ("y", "yes"):
            return raw_output
        if command in ("e", "edit"):
            preserved = {
                key: raw_output[key]
                for key in ("judge_id", "started_at")
                if key in raw_output
            }
            edited = collect_human_output()
            edited.update(preserved)
            edited["completed_at"] = _utc_now_iso()
            raw_output = edited
        else:
            print("Expected 'y' or 'edit'.")


def build_result_key(
    record: TraceRecord,
    rubric_key: str,
    judge_model: str,
) -> ResultKey:
    return (
        record.tag,
        record.sample_id,
        record.iteration,
        rubric_key,
        judge_model,
    )


def build_row_result_key(
    row: Dict[str, str],
    rubric_key: str,
    judge_model: str,
) -> ResultKey:
    return (
        row["tag"],
        row["sample_id"],
        int(row["iteration"]),
        rubric_key,
        judge_model,
    )


def iter_work_items(
    rows: Iterable[Dict[str, str]],
    rubrics: List[str],
) -> Iterable[Tuple[Dict[str, str], str]]:
    for row in rows:
        for rubric_key in rubrics:
            yield row, rubric_key


def main() -> None:
    args = parse_args()

    manifest_path = manifest_path_for_trace_dir(args.trace_dir)
    rows = read_manifest(str(manifest_path))
    rubrics = args.rubrics
    annotator_id = resolve_annotator_id(args.output_root, args.annotator_id)
    output_dir = output_dir_for_annotator(args.output_root, annotator_id)
    layout = ensure_output_layout(str(output_dir))
    judge_model = f"human:{annotator_id}"
    completed = load_completed_keys(layout["jsonl"]) if args.resume else set()

    shutil.copyfile(manifest_path, layout["base"] / "manifest.csv")
    with open(layout["base"] / "run_config.json", "w") as f:
        json.dump(
            {
                "judge_model": judge_model,
                "annotator_id": annotator_id,
                "output_dir": str(output_dir),
                "trace_dir": args.trace_dir,
                "manifest": str(manifest_path),
                "rubrics": rubrics,
                "resume": args.resume,
                "show_progress_metadata": args.show_progress_metadata,
            },
            f,
            indent=2,
        )

    work_items = list(iter_work_items(rows, rubrics))
    total_items = len(work_items)

    try:
        for item_index, (row, rubric_key) in enumerate(work_items, start=1):
            row_key = build_row_result_key(row, rubric_key, judge_model)
            if row_key in completed:
                continue

            record = load_trace_record(args.trace_dir, row)
            key = build_result_key(record, rubric_key, judge_model)

            rubric = get_rubric(rubric_key)
            view = format_human_judge_view(
                rubric=rubric,
                trace_payload=record.to_prompt_payload(),
            )

            if args.show_progress_metadata:
                progress = _section(f"Progress {item_index} / {total_items}")
            else:
                progress = ""

            clear_terminal()
            if progress:
                print(progress, sep="")
            print(view)
            started_at = _utc_now_iso()

            while True:
                try:
                    raw_output = collect_human_output()
                    raw_output["judge_id"] = annotator_id
                    raw_output["started_at"] = started_at
                    raw_output["completed_at"] = _utc_now_iso()
                    raw_output = confirm_or_edit(raw_output)
                    result = validate_and_build_human_result(
                        trace_record=record,
                        rubric_key=rubric_key,
                        judge_model=judge_model,
                        raw_output=raw_output,
                    )
                    append_result_jsonl(result, layout["jsonl"])
                    append_result_csv(result, layout["csv"])
                    completed.add(key)
                    break
                except (ValueError, HumanJudgeOutputValidationError) as exc:
                    print(f"Invalid judgment: {exc}")
                    print("Please enter this judgment again.")

    except KeyboardInterrupt:
        print("\nStopped. Saved judgments remain in the results directory.")


def launch_browser_app(argv: List[str]) -> None:
    """Replace this process with the Streamlit annotation interface."""

    app_path = Path(__file__).resolve().parent / "bench" / "judge" / "human_judge_app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--",
        *argv,
    ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    if "--terminal" in sys.argv:
        sys.argv.remove("--terminal")
        main()
    else:
        launch_browser_app(sys.argv[1:])
