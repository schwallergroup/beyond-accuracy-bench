from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .schema import TraceRecord


def trace_slug(row: Dict[str, str]) -> str:
    """Return the stable filename stem for one manifest row."""

    tag = row["tag"].replace("/", "_")
    return f"{tag}__{row['sample_id']}__iter{row['iteration']}"


def save_trace_record(record: TraceRecord, out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)


def manifest_path_for_trace_dir(trace_dir: str | Path) -> Path:
    """Locate the manifest bundled beside a downloaded traces directory."""

    path = Path(trace_dir).parent / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found beside trace directory: {path}. "
            "Use a traces/ directory created by download_reasoning_traces.py."
        )
    return path


def load_trace_record(trace_dir: str | Path, row: Dict[str, str]) -> TraceRecord:
    """Load and validate the archived trace selected by a manifest row."""

    path = Path(trace_dir) / f"{trace_slug(row)}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Archived trace not found: {path}. Download the traces for this manifest first."
        )

    with open(path, "r") as f:
        payload: Dict[str, Any] = json.load(f)

    expected = (row["tag"], row["sample_id"], int(row["iteration"]))
    observed = (
        str(payload.get("tag")),
        str(payload.get("sample_id")),
        payload.get("iteration"),
    )
    if observed != expected:
        raise ValueError(
            f"Archived trace identity does not match manifest row in {path}: "
            f"expected {expected}, found {observed}."
        )

    return TraceRecord(
        tag=expected[0],
        sample_id=expected[1],
        iteration=expected[2],
        model_label=payload.get("model_label"),
        previous_user_message=payload.get("previous_user_message"),
        mechanism_definitions=payload.get("mechanism_definitions"),
        raw_response=payload.get("raw_response"),
        source_call_id=payload.get("source_call_id"),
        extra=payload.get("extra") or {},
    )
