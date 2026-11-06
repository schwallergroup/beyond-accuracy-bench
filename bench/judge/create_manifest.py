from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple


GROUND_TRUTH_RE = re.compile(r"^(M\d+)_ground_truth_(\d+)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a balanced judge manifest from ground-truth files."
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default="ground_truths_200_samples_v2",
        help="Ground-truth directory to sample from.",
    )
    parser.add_argument(
        "--rows-per-mechanism",
        type=int,
        default=3,
        help="Number of sample ids to keep for each exact mechanism ID.",
    )
    parser.add_argument(
        "--sample-indices",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional explicit sample indices to keep for each mechanism, e.g. "
            "`--sample-indices 0` to keep only M*_0. If set, this overrides "
            "`--rows-per-mechanism`."
        ),
    )
    parser.add_argument(
        "--iterations",
        nargs="+",
        type=int,
        default=[3, 5],
        help="Iterations to include for each selected sample.",
    )
    parser.add_argument(
        "--tag-label",
        nargs=2,
        action="append",
        metavar=("TAG", "MODEL_LABEL"),
        required=True,
        help="Pair of tag and model label to include in the manifest. Repeat this flag.",
    )
    parser.add_argument(
        "--tag-label-mechanisms",
        nargs=3,
        action="append",
        metavar=("TAG", "MODEL_LABEL", "MECHANISMS"),
        default=None,
        help=(
            "Pair of tag/model label restricted to mechanisms, e.g. "
            "`--tag-label-mechanisms 20260429_091641.tar 'Claude Opus 4.7 with Mechanism Info' M1-M4,M18-M20`. "
            "Repeat this flag for experiments split across tags."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV path.",
    )
    return parser.parse_args()


def parse_mechanism_spec(spec: str) -> set[str]:
    mechanisms: set[str] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            if not (start.startswith("M") and end.startswith("M")):
                raise ValueError(f"Invalid mechanism range: {part}")
            start_num = int(start[1:])
            end_num = int(end[1:])
            if start_num > end_num:
                raise ValueError(f"Invalid descending mechanism range: {part}")
            mechanisms.update(f"M{idx}" for idx in range(start_num, end_num + 1))
        else:
            if not part.startswith("M"):
                raise ValueError(f"Invalid mechanism: {part}")
            mechanisms.add(part)
    return mechanisms


def collect_sample_ids(gt_dir: str) -> Dict[str, List[str]]:
    by_mech: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for filename in sorted(os.listdir(gt_dir)):
        match = GROUND_TRUTH_RE.match(filename)
        if not match:
            continue
        mech = match.group(1)
        sample_idx = int(match.group(2))
        sample_id = filename.replace("_ground_truth", "").replace(".json", "")
        by_mech[mech].append((sample_idx, sample_id))
    return {mech: [sid for _, sid in rows] for mech, rows in by_mech.items()}


def build_rows(
    sample_ids_by_mechanism: Dict[str, List[str]],
    rows_per_mechanism: int,
    sample_indices: List[int] | None,
    iterations: List[int],
    tag_labels: List[Tuple[str, str]],
    tag_label_mechanisms: List[Tuple[str, str, set[str]]] | None = None,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for mech_num in range(1, 21):
        mech = f"M{mech_num}"
        available_sample_ids = sample_ids_by_mechanism.get(mech, [])
        if sample_indices is not None:
            wanted = {f"{mech}_{idx}" for idx in sample_indices}
            sample_ids = [sid for sid in available_sample_ids if sid in wanted]
        else:
            sample_ids = available_sample_ids[:rows_per_mechanism]
        mechanism_tag_labels = list(tag_labels)
        if tag_label_mechanisms is not None:
            mechanism_tag_labels.extend(
                (tag, label)
                for tag, label, mechanisms in tag_label_mechanisms
                if mech in mechanisms
            )
        for sample_id in sample_ids:
            for iteration in iterations:
                for tag, model_label in mechanism_tag_labels:
                    rows.append(
                        {
                            "tag": tag,
                            "model_label": model_label,
                            "sample_id": sample_id,
                            "iteration": str(iteration),
                        }
                    )
    return rows


def main() -> None:
    args = parse_args()
    sample_ids_by_mech = collect_sample_ids(args.gt_dir)
    rows = build_rows(
        sample_ids_by_mechanism=sample_ids_by_mech,
        rows_per_mechanism=args.rows_per_mechanism,
        sample_indices=args.sample_indices,
        iterations=args.iterations,
        tag_labels=[(tag, label) for tag, label in args.tag_label],
        tag_label_mechanisms=[
            (tag, label, parse_mechanism_spec(mechanisms))
            for tag, label, mechanisms in (args.tag_label_mechanisms or [])
        ],
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tag", "model_label", "sample_id", "iteration"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
