"""Selected-run reader, judge alignment, and figure smoke tests (no APIs)."""

import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

import pandas as pd

from plotting_scripts.visualize_results import (
    BAR_RUBRIC_ORDER, accuracy_summary, build_metric_dataframe, open_run, reasoning_scores,
)
from plotting_scripts.neurips_rebuttal.camera_ready_reasoning_score_figures import plot_scatter


class SelectedRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runs = []

    def tearDown(self):
        for exp, _ in self.runs:
            exp.close()
        self.tmp.cleanup()

    def make_run(self, tag, model="unlisted-model", correct=True):
        batch = self.root / tag
        batch.mkdir()
        config = {"RUN_SETTINGS": {"run_args": {
            "proposer": "agent", "proposer_model": model, "steps": 5,
            "gt_dir": "my_custom_samples", "agent_prompt_variant": "standard",
        }}}
        (batch / "run_config.json").write_text(json.dumps(config))
        for sample in ["M1_0", "M2_0"]:
            folder = batch / sample / ("AgentProposer_" + model)
            folder.mkdir(parents=True)
            gt = sample.split("_")[0]
            fit = {"M1": 2, "M2": 2}
            fit[gt] = 0 if correct else 4
            for step in range(1, 6):
                (folder / f"step_{step:03d}_fitting_results.json").write_text(json.dumps(fit))
                (folder / f"step_{step:03d}_proposal.json").write_text(json.dumps({
                    "proposal": {"S": 0.1 * step, "P": 0, "cat": 0.01},
                }))
        return batch

    def open(self, tag, label):
        run = open_run(self.root, tag, label, "blue")
        self.runs.append(run)
        return run

    def write_scores(self, tag, judge="gpt-5.4", missing=None):
        rows = []
        for sample in ["M1_0", "M2_0"]:
            for iteration in [2, 3, 4]:
                for rubric in BAR_RUBRIC_ORDER:
                    if (sample, iteration, rubric) == missing:
                        continue
                    rows.append(dict(tag=tag, sample_id=sample, iteration=iteration,
                                     rubric_key=rubric, score=2, judge_model=judge,
                                     model_label="An unrelated display name"))
        path = self.root / f"scores_{judge}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_directory_and_both_archive_layouts_match(self):
        batch = self.make_run("new_tag")
        directory, _ = self.open("new_tag", "Custom")
        expected = build_metric_dataframe([directory])
        for prefix in ["new_tag", ""]:
            name = "nested.tar" if prefix else "rootless.tar"
            with tarfile.open(self.root / name, "w") as archive:
                for path in batch.rglob("*.json"):
                    archive.add(path, arcname=str(Path(prefix) / path.relative_to(batch)))
            exp, _ = self.open(name, "Custom")
            actual = build_metric_dataframe([exp])
            pd.testing.assert_frame_equal(expected.drop(columns="source_tar"),
                                          actual.drop(columns="source_tar"))
        self.assertEqual(accuracy_summary(expected, [directory]).iloc[0].task_accuracy, 1)

    def test_tag_matching_and_complete_grid(self):
        self.make_run("new_tag")
        run = self.open("new_tag", "Custom")
        csv = self.write_scores("new_tag.tar", missing=("M1_0", 2, BAR_RUBRIC_ORDER[0]))
        summary, per_trace = reasoning_scores([csv], [run], [2, 3, 4])
        self.assertEqual(set(per_trace.sample_id), {"M2_0"})
        self.assertEqual(set(summary.n_reasoning_samples), {3})
        self.assertEqual(set(summary.reasoning_score), {2})

    def test_self_judge_uses_configuration_not_display_label(self):
        self.make_run("new_tag", model="gpt-5.4")
        run = self.open("new_tag", "My custom label")
        self_csv = self.write_scores("new_tag")
        with self.assertRaisesRegex(ValueError, "No eligible"):
            reasoning_scores([self_csv], [run], [2, 3, 4])
        other_csv = self.write_scores("new_tag", judge="claude-4.7-opus")
        _, traces = reasoning_scores([self_csv, other_csv], [run], [2, 3, 4])
        self.assertTrue((traces.n_judges == 1).all())

    def test_overlapping_ratings_are_rejected(self):
        self.make_run("new_tag")
        run = self.open("new_tag", "Custom")
        csv = self.write_scores("new_tag")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            reasoning_scores([csv, csv], [run], [2, 3, 4])

    def test_cli_generates_all_main_figure_types_for_one_new_run(self):
        self.make_run("new_tag")
        csv = self.write_scores("new_tag")
        output = self.root / "figures"
        result = subprocess.run([
            sys.executable, str(Path(__file__).with_name("visualize_results.py")),
            "--results-dir", str(self.root), "--run", "new_tag", "My custom agent",
            "--judge-csv", str(csv), "--output-dir", str(output),
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual({p.name for p in output.glob("*.png")}, {
            "baseline_accuracy_entropy_comparison.png", "model_accuracy_at_iteration5.png",
            "reasoning_scores.png", "ded_pr_vs_accuracy.png",
        })
        self.assertTrue(pd.read_csv(output / "ded_pr_regression_bands.csv").empty)

    def test_scatter_regression_for_three_distinct_methods(self):
        rows = []
        for index, model in enumerate(["Custom A", "Custom B", "Custom C"]):
            for rubric in BAR_RUBRIC_ORDER:
                score = 0.4 + index * 0.6
                rows.append(dict(model_name=model, rubric_key=rubric, reasoning_score=score,
                                 score_ci_low=score - 0.1, score_ci_high=score + 0.1))
        models = ["Custom A", "Custom B", "Custom C"]
        accuracy = pd.DataFrame(dict(model_name=models, task_accuracy=[0.2, 0.5, 0.8],
                                     accuracy_ci_low=[0.1, 0.4, 0.7], accuracy_ci_high=[0.3, 0.6, 0.9]))
        bands = plot_scatter(pd.DataFrame(rows), accuracy, models,
                             output_path=self.root / "scatter.png",
                             colors=dict(zip(models, ["blue", "green", "red"])), custom=True)
        self.assertEqual(len(bands), 400)
        self.assertTrue((bands.n_models == 3).all())
        self.assertTrue((bands.spearman_rho > 0.99).all())


if __name__ == "__main__":
    unittest.main()
