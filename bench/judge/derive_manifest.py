from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

SAMPLE_ID_RE = re.compile(r"^M\d+_(\d+)$")
SOURCE_FIELDS = ["tag", "model_label", "sample_id", "iteration"]
MANIFEST_FIELDS = SOURCE_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a judge manifest from an existing manifest while preserving its "
            "tag/model-label mapping and recording provenance."
        )
    )
    parser.add_argument(
        "--source-manifest",
        required=True,
        help="Existing CSV manifest that supplies tag, model_label, and sample_id.",
    )
    parser.add_argument(
        "--sample-indices",
        nargs="+",
        type=int,
        help="Sample indices to retain, e.g. 0 1 for M*_0 and M*_1.",
    )
    parser.add_argument(
        "--iterations",
        nargs="+",
        type=int,
        help="Iterations to create for every retained tag/model/sample combination.",
    )
    parser.add_argument(
        "--sample-ids-by-iteration",
        nargs="+",
        metavar="ITERATION:SAMPLE_ID[,SAMPLE_ID...]",
        help=(
            "Optional per-iteration sample selection, e.g. "
            "3:M2_1,M7_0,M12_0 4:M2_1,M7_0,M12_0,M3_0,M9_0,M15_0. "
            "Cannot be combined with --sample-indices/--iterations."
        ),
    )
    parser.add_argument("--output", required=True, help="Output CSV manifest path.")
    return parser.parse_args()


def read_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    missing = set(SOURCE_FIELDS).difference(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Source manifest is missing required columns: {sorted(missing)}")
    return rows


def sample_index(sample_id: str) -> int:
    match = SAMPLE_ID_RE.fullmatch(sample_id)
    if match is None:
        raise ValueError(f"Invalid sample_id {sample_id!r}; expected M<number>_<index>.")
    return int(match.group(1))


def unique_trace_sources(rows: Iterable[Dict[str, str]]) -> List[Tuple[str, str, str]]:
    seen: set[Tuple[str, str, str]] = set()
    sources: List[Tuple[str, str, str]] = []
    for row in rows:
        key = (row["tag"], row["model_label"], row["sample_id"])
        if key not in seen:
            seen.add(key)
            sources.append(key)
    return sources


def derive_rows(
    source_rows: Iterable[Dict[str, str]],
    sample_indices: List[int],
    iterations: List[int],
) -> List[Dict[str, str]]:
    selected_indices = set(sample_indices)
    sources = unique_trace_sources(
        row for row in source_rows if sample_index(row["sample_id"]) in selected_indices
    )
    if not sources:
        raise ValueError("No rows remain after filtering source manifest by sample indices.")
    return [
        {
            "tag": tag,
            "model_label": model_label,
            "sample_id": sample_id,
            "iteration": str(iteration),
        }
        for tag, model_label, sample_id in sources
        for iteration in iterations
    ]


def parse_sample_ids_by_iteration(values: List[str]) -> Dict[int, List[str]]:
    selections: Dict[int, List[str]] = {}
    for value in values:
        try:
            iteration_text, sample_ids_text = value.split(":", 1)
            iteration = int(iteration_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid selection {value!r}; expected ITERATION:SAMPLE_ID[,SAMPLE_ID...]."
            ) from exc
        sample_ids = [sample_id for sample_id in sample_ids_text.split(",") if sample_id]
        if not sample_ids:
            raise ValueError(f"Selection {value!r} does not include a sample ID.")
        if iteration in selections:
            raise ValueError(f"Iteration {iteration} appears more than once.")
        selections[iteration] = sample_ids
    return selections


def derive_rows_by_iteration(
    source_rows: Iterable[Dict[str, str]], selections: Dict[int, List[str]]
) -> List[Dict[str, str]]:
    sources = unique_trace_sources(source_rows)
    available = {sample_id for _, _, sample_id in sources}
    requested = {sample_id for sample_ids in selections.values() for sample_id in sample_ids}
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"Requested sample IDs are absent from source manifest: {missing}")
    return [
        {
            "tag": tag,
            "model_label": model_label,
            "sample_id": sample_id,
            "iteration": str(iteration),
        }
        for iteration, sample_ids in selections.items()
        for tag, model_label, sample_id in sources
        if sample_id in sample_ids
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_manifest)
    source_rows = read_manifest(str(source_path))
    if args.sample_ids_by_iteration:
        if args.sample_indices or args.iterations:
            raise ValueError(
                "Use either --sample-ids-by-iteration or --sample-indices with --iterations."
            )
        derived_rows = derive_rows_by_iteration(
            source_rows, parse_sample_ids_by_iteration(args.sample_ids_by_iteration)
        )
    else:
        if not args.sample_indices or not args.iterations:
            raise ValueError(
                "--sample-indices and --iterations are required without per-iteration selections."
            )
        derived_rows = derive_rows(source_rows, args.sample_indices, args.iterations)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(derived_rows)

    provenance_path = output_path.with_suffix(".provenance.json")
    with open(provenance_path, "w") as f:
        json.dump(
            {
                "derived_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_manifest": str(source_path),
                "source_manifest_sha256": sha256(source_path),
                "source_row_count": len(source_rows),
                "sample_indices": args.sample_indices,
                "iterations": args.iterations,
                "sample_ids_by_iteration": args.sample_ids_by_iteration,
                "derived_row_count": len(derived_rows),
                "command": [sys.executable, *sys.argv],
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
