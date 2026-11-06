from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from bench.judge.prompts import build_judge_prompt
from bench.judge.rubrics import RUBRIC_ORDER
from bench.judge.schema import JudgeResult
from bench.judge.trace_io import load_trace_record, trace_slug


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score archived reasoning traces with an LLM judge."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help=(
            "CSV manifest describing which traces to judge. Required columns: "
            "tag,sample_id,iteration. Optional: model_label"
        ),
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        required=True,
        help="Directory containing downloaded traces/*.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write prompt dumps and score tables.",
    )
    parser.add_argument(
        "--rubrics",
        nargs="+",
        default=RUBRIC_ORDER,
        help=f"Rubric keys to score. Default: {' '.join(RUBRIC_ORDER)}",
    )
    parser.add_argument(
        "--dump-prompts-only",
        action="store_true",
        help="Save prompts without calling any judge model.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5.4",
        help="Exact judge model name to call for rubric scoring.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=positive_int,
        default=5,
        help="Maximum simultaneous LLM judge calls. Default: 5.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=positive_int,
        default=1024,
        help="Maximum tokens returned by one judge call. Default: 1024.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing results and score only unfinished trace/rubric pairs.",
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


def ensure_output_layout(output_dir: str) -> Dict[str, Path]:
    base = Path(output_dir)
    prompts_dir = base / "prompts"
    results_dir = base / "results"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "prompts": prompts_dir,
        "results": results_dir,
    }


def save_prompt_dump(prompt_payload: Dict[str, str], out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump(prompt_payload, f, indent=2, ensure_ascii=False)


def save_results_jsonl(results: Iterable[JudgeResult], out_path: Path) -> None:
    with open(out_path, "w") as f:
        for result in results:
            f.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")


def save_results_csv(results: Iterable[JudgeResult], out_path: Path) -> None:
    rows = [result.to_row() for result in results]
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_result_jsonl(result: JudgeResult, out_path: Path) -> None:
    with open(out_path, "a") as f:
        f.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")


def append_result_csv(result: JudgeResult, out_path: Path) -> None:
    row = result.to_row()
    file_exists = out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def completed_result_keys(path: Path) -> set[tuple[str, str, int, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    with open(path) as f:
        for line in f:
            result = json.loads(line)
            keys.add(
                (
                    result["tag"],
                    result["sample_id"],
                    int(result["iteration"]),
                    result["rubric_key"],
                    result["judge_model"],
                )
            )
    return keys


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    rubrics = args.rubrics
    layout = ensure_output_layout(args.output_dir)
    backend_config = None
    run_judge_with_retries = None
    if not args.dump_prompts_only:
        from bench.judge.runner import JudgeBackendConfig, run_judge_with_retries

        backend_config = JudgeBackendConfig(
            judge_model=args.judge_model,
            max_completion_tokens=args.max_completion_tokens,
        )

    shutil.copyfile(args.manifest, layout["base"] / "manifest.csv")
    with open(layout["base"] / "run_config.json", "w") as f:
        json.dump(
            {
                "judge_model": args.judge_model,
                "trace_dir": args.trace_dir,
                "rubrics": rubrics,
                "dump_prompts_only": args.dump_prompts_only,
                "max_concurrency": args.max_concurrency,
                "max_completion_tokens": args.max_completion_tokens,
                "resume": args.resume,
            },
            f,
            indent=2,
        )

    jsonl_path = layout["results"] / "judge_results.jsonl"
    csv_path = layout["results"] / "judge_results.csv"
    if not args.dump_prompts_only and not args.resume:
        for path in (jsonl_path, csv_path):
            if path.exists():
                path.unlink()

    completed = completed_result_keys(jsonl_path) if args.resume else set()
    judge_items = []
    for row in rows:
        record = load_trace_record(args.trace_dir, row)
        slug = trace_slug(row)

        for rubric_key in rubrics:
            prompt_parts = build_judge_prompt(
                rubric_key=rubric_key,
                trace_payload=record.to_prompt_payload(),
            )
            save_prompt_dump(
                {
                    "system_prompt": prompt_parts.system_prompt,
                    "user_prompt": prompt_parts.user_prompt,
                },
                layout["prompts"] / f"{slug}__{rubric_key}.json",
            )

            if args.dump_prompts_only:
                continue

            result_key = (
                record.tag,
                record.sample_id,
                record.iteration,
                rubric_key,
                args.judge_model,
            )
            if result_key in completed:
                continue

            judge_items.append((record, rubric_key))

    if args.dump_prompts_only:
        return

    assert backend_config is not None
    assert run_judge_with_retries is not None
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        futures = {
            executor.submit(
                run_judge_with_retries,
                trace_record=record,
                rubric_key=rubric_key,
                backend_config=backend_config,
            ): (record, rubric_key)
            for record, rubric_key in judge_items
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            append_result_jsonl(result, jsonl_path)
            append_result_csv(result, csv_path)
            print(f"Completed {completed_count}/{len(judge_items)} judge calls.", flush=True)


if __name__ == "__main__":
    main()
