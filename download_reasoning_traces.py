from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List

from bench.judge.trace_io import load_trace_record, save_trace_record, trace_slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download manifest-selected reasoning traces from Weave for local judging."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV manifest with tag,sample_id,iteration and optional model_label.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory that will contain manifest.csv, run_config.json, and traces/.",
    )
    return parser.parse_args()


def read_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    required = {"tag", "sample_id", "iteration"}
    missing = required.difference(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    return rows


def main() -> None:
    args = parse_args()
    from bench.judge.trace_fetch import fetch_trace_record

    rows = read_manifest(args.manifest)
    base = Path(args.output_dir)
    traces_dir = base / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.manifest, base / "manifest.csv")
    with open(base / "run_config.json", "w") as f:
        json.dump({"source": "weave", "trace_count": len(rows)}, f, indent=2)

    for index, row in enumerate(rows, start=1):
        destination = traces_dir / f"{trace_slug(row)}.json"
        if destination.is_file():
            # A validated local record is immutable input for both judge paths;
            # retain it so a stopped download can be safely resumed.
            load_trace_record(traces_dir, row)
            continue
        record = fetch_trace_record(
            tag=row["tag"],
            sample_id=row["sample_id"],
            iteration=int(row["iteration"]),
            model_label=row.get("model_label"),
        )
        save_trace_record(record, destination)
        if index % 25 == 0 or index == len(rows):
            print(f"Downloaded {index}/{len(rows)} traces.", flush=True)


if __name__ == "__main__":
    main()
