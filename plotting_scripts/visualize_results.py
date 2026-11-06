"""Generate the paper's main figure types for explicitly selected local runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tarfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plotting_scripts.neurips.archived_experiment_data import (
    DirectoryExperiment, TarExperiment, build_metric_dataframe,
)
from plotting_scripts.neurips.baseline_accuracy_entropy_comparison import (
    _mean_and_ci, _plot_metric_panel,
)
from plotting_scripts.neurips_rebuttal.camera_ready_reasoning_score_figures import (
    BAR_RUBRIC_ORDER, plot_bar, plot_scatter,
)
from plotting_scripts.neurips_rebuttal.judge_filtering import canonical_model_id
from plotting_scripts.styling import apply_paper_rcparams


def normalize_tag(value):
    return Path(str(value)).name.removesuffix(".tar")


def open_run(results_dir, tag, label, color):
    """Read one exact source, without paper model/source allowlists."""
    if Path(tag).name != tag:
        raise ValueError("Use a batch tag, not a path; set --results-dir for its parent.")
    path = results_dir / tag
    if not path.exists() and not tag.endswith(".tar"):
        path = results_dir / (tag + ".tar")
    if not path.exists():
        raise ValueError(f"Run does not exist: {path}")
    archive = None
    try:
        if path.is_dir():
            config = json.loads((path / "run_config.json").read_text())
            files = [p.relative_to(path).as_posix() for p in path.glob("M*/*/step_*_fitting_results.json")]
            prefix = ""
        else:
            archive = tarfile.open(path, "r")
            names = archive.getnames()
            configs = [name for name in names if Path(name).name == "run_config.json"]
            if len(configs) != 1:
                raise ValueError(f"Expected one batch run_config.json in {path}")
            config = json.load(archive.extractfile(configs[0]))
            prefix = configs[0].removesuffix("run_config.json")
            files = [name[len(prefix):] for name in names if name.startswith(prefix)]
        pairs = set()
        for name in files:
            parts = name.split("/")
            if (len(parts) == 3 and parts[0].startswith("M")
                    and parts[2].startswith("step_") and parts[2].endswith("_fitting_results.json")):
                pairs.add((parts[0], parts[1]))
        methods = {method for _, method in pairs}
        if len(methods) != 1:
            raise ValueError(f"Expected one method with fitting results in {path}; found {sorted(methods)}")
        kwargs = dict(source_path=str(path), proposer_name=methods.pop(), label=label,
                      color=color, linestyle="-", sample_ids=sorted({sample for sample, _ in pairs}))
        if archive is None:
            experiment = DirectoryExperiment(**kwargs, root_dir=str(path))
        else:
            # TarExperiment joins root_prefix and relative_path with '/'. For
            # rootless archives retain their actual member names via a reader.
            experiment = SelectedTarExperiment(**kwargs, root_prefix=prefix,
                                               tar=archive, members=set(names))
        settings = config["RUN_SETTINGS"]["run_args"]
        return experiment, settings
    except Exception:
        if archive is not None:
            archive.close()
        raise


class SelectedTarExperiment(TarExperiment):
    def read_json(self, relative_path):
        name = self.root_prefix + relative_path
        if name not in self.members:
            return None
        return json.load(self.tar.extractfile(name))


def accuracy_summary(metrics, experiments):
    rows = []
    for exp in experiments:
        values = metrics.loc[(metrics["method"] == exp.label)
                             & (metrics["metric"] == "Accuracy from NLL")
                             & (metrics["step"] == 5), "value"]
        if values.empty:
            raise ValueError(f"No iteration-5 accuracy for {exp.label}; run five iterations first.")
        mean, low, high = _mean_and_ci(values)
        rows.append(dict(model_name=exp.label, task_accuracy=mean, accuracy_ci_low=low,
                         accuracy_ci_high=high, accuracy_n=len(values)))
    return pd.DataFrame(rows)


def plot_task_figures(metrics, accuracy, experiments, output):
    apply_paper_rcparams()
    fig, axes = plt.subplots(2, 2, figsize=(182 / 25.4, 105 / 25.4))
    for ax, metric in zip(axes.flat, ["Accuracy from NLL", "Mechanism Entropy",
                                    "Occupancy Entropy", "S-cat Corner Mass"]):
        _plot_metric_panel(ax, metrics, metric, experiments,
                           "Accuracy" if metric == "Accuracy from NLL" else metric, ylim=(0, 1.05))
    axes[0, 0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(output / "baseline_accuracy_entropy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    colors = {exp.label: exp.color for exp in experiments}
    summary = accuracy.sort_values("task_accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(5, max(2.5, len(summary) * 0.35)))
    y = np.arange(len(summary))
    means = summary["task_accuracy"].to_numpy()
    ax.barh(y, means, color=[colors[label] for label in summary["model_name"]],
            edgecolor="black", xerr=np.vstack([means - summary["accuracy_ci_low"],
                                              summary["accuracy_ci_high"] - means]), capsize=2.5)
    ax.set_yticks(y, summary["model_name"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy at Iteration 5")
    ax.grid(axis="x", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output / "model_accuracy_at_iteration5.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def reasoning_scores(csv_paths, runs, iterations):
    """Use eligible judges on a complete shared sample/iteration/rubric grid."""
    scores = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    required = {"tag", "sample_id", "iteration", "rubric_key", "score", "judge_model"}
    if not required.issubset(scores):
        raise ValueError(f"Judge CSVs must contain {sorted(required)}")
    scores["tag_key"] = scores["tag"].map(normalize_tag)
    scores["judge_id"] = scores["judge_model"].map(canonical_model_id)
    scores["score"] = pd.to_numeric(scores["score"], errors="raise")
    if not scores["score"].isin([0, 1, 2]).all():
        raise ValueError("Judge scores must be 0, 1, or 2; finish scoring before plotting.")
    eligible_frames = []
    complete_sets = []
    for exp, settings in runs:
        if settings["proposer"] != "agent":
            continue
        target = canonical_model_id(settings["proposer_model"])
        frame = scores[(scores["tag_key"] == normalize_tag(exp.source_tar))
                       & scores["iteration"].isin(iterations)
                       & scores["sample_id"].isin(exp.sample_ids)
                       & scores["rubric_key"].isin(BAR_RUBRIC_ORDER)
                       & (scores["judge_id"] != target)].copy()
        if frame.empty:
            raise ValueError(f"No eligible judge scores for {exp.label}; supply a different judge from the agent.")
        keys = ["sample_id", "iteration", "rubric_key", "judge_id"]
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate ratings for {exp.label}; do not supply overlapping judge CSVs.")
        expected = len(iterations) * len(BAR_RUBRIC_ORDER) * frame["judge_id"].nunique()
        counts = frame.groupby("sample_id").size()
        complete_sets.append(set(counts[counts == expected].index))
        frame["model_name"] = exp.label
        eligible_frames.append(frame)
    if not eligible_frames:
        raise ValueError("Reasoning figures require at least one agent run.")
    shared = set.intersection(*complete_sets)
    if not shared:
        raise ValueError("No complete samples shared by all selected agents, iterations, rubrics, and eligible judges.")
    eligible = pd.concat(eligible_frames, ignore_index=True)
    per_trace = (eligible[eligible["sample_id"].isin(shared)]
                 .groupby(["model_name", "sample_id", "iteration", "rubric_key"], as_index=False)
                 .agg(reasoning_score=("score", "mean"), n_judges=("judge_id", "nunique")))
    rows = []
    for (model, rubric), group in per_trace.groupby(["model_name", "rubric_key"], sort=False):
        # Keep repeated iterations of each independent system together, as in
        # the camera-ready analysis.
        system_means = group.groupby("sample_id")["reasoning_score"].mean()
        mean, low, high = _mean_and_ci(system_means)
        rows.append(dict(model_name=model, rubric_key=rubric, reasoning_score=mean,
                         score_ci_low=low, score_ci_high=high, n_reasoning_samples=len(group),
                         n_independent_systems=len(system_means),
                         n_iterations=len(iterations), n_shared_sample_ids_per_iteration=len(shared)))
    return pd.DataFrame(rows), per_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=2, action="append", required=True, metavar=("TAG", "LABEL"),
                        help="Exact run tag (directory or .tar) and unique display label. Repeat to compare runs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-csv", type=Path, action="append", default=[],
                        help="Judge result CSV; repeat for multiple judges. Enables reasoning figures.")
    parser.add_argument("--iterations", nargs="+", type=int, choices=range(1, 6), default=[2, 3, 4])
    args = parser.parse_args()
    if len({label for _, label in args.run}) != len(args.run):
        parser.error("Each run needs a unique display label.")
    if len({normalize_tag(tag) for tag, _ in args.run}) != len(args.run):
        parser.error("Select each batch only once.")
    if len(set(args.iterations)) != len(args.iterations):
        parser.error("Iterations must be unique.")
    runs = []
    try:
        for index, (tag, label) in enumerate(args.run):
            runs.append(open_run(args.results_dir, tag, label, plt.get_cmap("tab20")(index % 20)))
        experiments = [exp for exp, _ in runs]
        metrics = build_metric_dataframe(experiments)
        if metrics.empty:
            raise ValueError("No usable metrics found in the selected runs.")
        accuracy = accuracy_summary(metrics, experiments)
        # Validate reasoning inputs before writing any output.
        summary, per_trace = reasoning_scores(args.judge_csv, runs, args.iterations) if args.judge_csv else (None, None)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(args.output_dir / "per_sample_metrics.csv", index=False)
        accuracy.to_csv(args.output_dir / "iteration5_accuracy.csv", index=False)
        plot_task_figures(metrics, accuracy, experiments, args.output_dir)
        if summary is not None:
            colors = {exp.label: exp.color for exp in experiments}
            order = [exp.label for exp, settings in runs if settings["proposer"] == "agent"]
            summary.to_csv(args.output_dir / "reasoning_scores.csv", index=False)
            per_trace.to_csv(args.output_dir / "per_trace_reasoning_scores.csv", index=False)
            plot_bar(summary, order, output_path=args.output_dir / "reasoning_scores.png",
                     colors=colors, display={label: label for label in order})
            bands = plot_scatter(summary, accuracy, order, output_path=args.output_dir / "ded_pr_vs_accuracy.png",
                                 colors=colors, custom=True)
            # Always replace the table, including when a previous invocation
            # in this output directory had an estimable regression.
            band_columns = ["rubric_key", "x", "ols_prediction", "bootstrap_ci_low", "bootstrap_ci_high",
                            "ols_slope", "ols_intercept", "spearman_rho", "spearman_ci_low",
                            "spearman_ci_high", "spearman_pvalue", "n_models"]
            bands.reindex(columns=band_columns).to_csv(
                args.output_dir / "ded_pr_regression_bands.csv", index=False)
        (args.output_dir / "selection.json").write_text(json.dumps({
            "runs": [{"source": str(Path(exp.source_path).resolve()), "label": exp.label,
                      "settings": settings} for exp, settings in runs],
            "judge_csvs": [str(path.resolve()) for path in args.judge_csv],
            "iterations": args.iterations,
        }, indent=2) + "\n")
        print(f"Saved selected-run figures and source tables to {args.output_dir}")
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")
    finally:
        for exp, _ in runs:
            exp.close()


if __name__ == "__main__":
    main()
