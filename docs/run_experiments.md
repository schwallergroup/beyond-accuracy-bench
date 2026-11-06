# Run and evaluate your own experiments

Complete [environment setup](setup.md) first. Run commands from the repository root. The full benchmark uses 200 systems and five iterations per method; fitting is computationally expensive.

## 1. Run an experiment

Baselines below disable Weave logging with `--no-weave` and require no API keys. The LLM agent example enables Weave logging for the reasoning evaluation in section 3; configure both `OPENROUTER_API_KEY` and `WANDB_API_KEY`. Local experiment logs contain proposals and fitting results, but do not save the full reasoning traces required by the judge. Ensure Weave logging is available and enabled before running the agent.

For a smaller first run, add `--ground-truth-pattern 'M1_0'` and reduce `--jobs` to `1`. A single sample checks the pipeline; it does not estimate benchmark-wide performance.

The paper ground-truth samples are already included; generating new samples is optional. Preserve the bundled samples when reproducing the paper.

Before generating new samples, change `GROUND_TRUTH_GENERATOR_CONFIG["output_dir"]`
in `bench/config.py` to a separate directory, such as `ground_truths_custom`.
The generator currently defaults to the bundled `ground_truths_200_samples_v2`
directory and overwrites existing files. It has no `--output-dir` option.
For experiments using newly generated samples, set `--gt-dir` to that directory.

Sample kinetic constants for benchmark instances:

```bash
poetry run python generate_ground_truths.py \
  --models M1 M2 M3 M4 M5 M6 M7 M8 M9 M10 M11 M12 M13 M14 M15 M16 M17 M18 M19 M20 \
  --num-ground-truths 10 \
  --jobs 70
```

Run the Maximin LHS baseline on all included kinetic-constant samples:

```bash
poetry run python run_experiment.py \
  --mode sample-parallel \
  --jobs 70 \
  --steps 5 \
  --gt-dir ground_truths_200_samples_v2 \
  --proposer maximin-lhs \
  --no-weave
```

Run the Bayes baseline:

```bash
poetry run python run_experiment.py \
  --mode sample-parallel \
  --jobs 70 \
  --steps 5 \
  --gt-dir ground_truths_200_samples_v2 \
  --proposer bayes \
  --no-weave
```

Run the EIG Reference baseline:

```bash
poetry run python run_experiment.py \
  --mode sample-parallel \
  --jobs 70 \
  --steps 5 \
  --gt-dir ground_truths_200_samples_v2 \
  --proposer map-eig \
  --no-weave
```

Run an LLM experimental-design method:

```bash
poetry run python run_experiment.py \
  --mode sample-parallel \
  --jobs 70 \
  --steps 5 \
  --gt-dir ground_truths_200_samples_v2 \
  --proposer agent \
  --proposer-model gemini-3.1-pro \
  --feedback-mode nll \
  --agent-prompt-variant standard
```

For the explicit four-step prompt ablation, use `--agent-prompt-variant step_instruction`. For the mechanism-information-withheld ablation, use `--agent-prompt-variant mechanism_hidden`. Both variants require `--proposer agent --feedback-mode nll`.

The full-run examples use 70 parallel sample workers; adjust this to the
machine's CPU and memory capacity. Ground-truth generation parallelizes over
mechanisms instead, so only the 20 selected mechanisms can run concurrently.

LLM reruns can differ from the archived results as model APIs and routing change.

## 2. Visualize your results

The runner prints `Starting experiment batch with tag: <tag>` and writes the batch under `results/<tag>/`. Sample/method subdirectories contain per-step proposals and fitting results; the batch also contains `run_config.json`.

The runner saves ordinary directories, not `.tar` archives. The code below reads those directories directly; no manual archiving is needed. It expects this layout:

```text
results/<tag>/
  run_config.json
  M1_0/
    AgentProposer_gemini-3.1-pro/
      step_001_fitting_results.json
      step_001_proposal.json
      ...
  ...
```

Use the following command to generate the **main figure types listed in [Paper data and results](reproduce.md#main-figures)** from your own runs. It uses the paper's metric calculations and bootstrap confidence intervals, without the authors' fixed run selection. Both ordinary batch directories and `.tar` archives are supported; no manual archiving or extraction is needed. If a directory and archive share a tag, the directory is used; specify the `.tar` suffix to select the archive explicitly.

Replace the tags with the ones printed by your runs. Each `--run` takes a tag and a unique display label; repeat it for every agent or baseline you want to compare. Only those runs are included. One run is also supported.

`visualize_results.py` is the selected-run plotting entry point: without judge
CSVs it generates Fig. 2–3-style plots; adding judge CSVs also enables the
Fig. 4-style plots. It reuses shared analysis and plotting functions rather
than running the paper scripts with their fixed model/run selections.

```bash
poetry run python plotting_scripts/visualize_results.py \
  --run YOUR_AGENT_TAG "My agent" \
  --run YOUR_LHS_TAG "Maximin LHS" \
  --run YOUR_BAYES_TAG "Bayes" \
  --run YOUR_EIG_TAG "EIG Reference" \
  --output-dir results/my_analysis
```

This creates:

| Paper figure type | Output in `results/my_analysis/` |
| --- | --- |
| Fig. 2: accuracy, normalized mechanism entropy, occupancy entropy, corner mass over iterations | `baseline_accuracy_entropy_comparison.png` |
| Fig. 3: accuracy at iteration 5 | `model_accuracy_at_iteration5.png` |
| Source measurements, iteration-5 summaries, and selected-run settings | `per_sample_metrics.csv`, `iteration5_accuracy.csv`, `selection.json` |

Run five iterations for these figures. As in the paper analysis, a missing or failed fit carries forward the most recent valid earlier fit; samples without any valid fit at or before iteration 5 are excluded from iteration-5 accuracy. Check sample counts and the `fit_step` column in the metric CSV when interpreting results. The command uses the same metrics and bootstrap procedure as the paper scripts, with full-range axes to accommodate your results.

**Fig. 4 requires the reasoning scores produced in section 3.** After judging, run the same command with the judge CSVs:

```bash
poetry run python plotting_scripts/visualize_results.py \
  --run YOUR_AGENT_TAG "My agent" \
  --run YOUR_LHS_TAG "Maximin LHS" \
  --run YOUR_BAYES_TAG "Bayes" \
  --run YOUR_EIG_TAG "EIG Reference" \
  --judge-csv bench/judge/runs/my_gpt54_judge/results/judge_results.csv \
  --judge-csv bench/judge/runs/my_opus47_judge/results/judge_results.csv \
  --iterations 2 3 4 \
  --output-dir results/my_analysis
```

In addition to Fig. 2–3, this writes `reasoning_scores.png` (four-axis scores) and `ded_pr_vs_accuracy.png` (DED/PR versus iteration-5 accuracy), corresponding to Fig. 4. Source tables are `reasoning_scores.csv` and `per_trace_reasoning_scores.csv`; `ded_pr_regression_bands.csv` contains confidence bands and Spearman statistics when a regression is estimable, and otherwise contains only column headers.

Judge rows are matched by batch tag, not display label. Self-judgments are excluded using the generating model in each run's configuration; eligible judges are averaged per trace. The analysis uses samples complete across the selected agent runs, iterations, four rubrics, and eligible judges. Baselines appear in Fig. 2–3 but have no reasoning scores. For one or two agents, or constant scores/accuracy, the scatter still shows the points and confidence intervals but omits regression and correlation estimates. To compare several agents, repeat `--run` for each and supply judge CSVs covering them all.

Reasoning-score confidence intervals resample independent systems, keeping
each system's selected iterations together. Accuracy on the scatter's vertical
axis is always measured at iteration 5; `--iterations` selects the reasoning
iterations only.

## 3. Evaluate agent reasoning

Create a manifest selecting traces from your Weave-recorded agent run. Use the same agent tag as in section 2. This example selects samples 0 and 1 for every mechanism and iterations 2–4, as in the paper's main reasoning analysis. Adjust the selection to the samples and iterations actually completed; baseline runs do not have agent reasoning traces.

```bash
poetry run python bench/judge/create_manifest.py \
  --gt-dir ground_truths_200_samples_v2 \
  --sample-indices 0 1 \
  --iterations 2 3 4 \
  --tag-label YOUR_AGENT_TAG "My agent" \
  --output bench/judge/manifests/my_manifest.csv
```

Repeat `--tag-label TAG "Label"` to evaluate several agent runs together. Download the selected reasoning traces from Weave using `WANDB_API_KEY`:

```bash
poetry run python download_reasoning_traces.py \
  --manifest bench/judge/manifests/my_manifest.csv \
  --output-dir bench/judge/runs/my_traces
```

Score the local traces with both judges used in the paper. These commands use `OPENROUTER_API_KEY` and do not query Weave:

```bash
poetry run python judge_reasoning_traces.py \
  --manifest bench/judge/manifests/my_manifest.csv \
  --trace-dir bench/judge/runs/my_traces/traces \
  --output-dir bench/judge/runs/my_gpt54_judge \
  --judge-model gpt-5.4 \
  --max-concurrency 5

poetry run python judge_reasoning_traces.py \
  --manifest bench/judge/manifests/my_manifest.csv \
  --trace-dir bench/judge/runs/my_traces/traces \
  --output-dir bench/judge/runs/my_opus47_judge \
  --judge-model claude-4.7-opus \
  --max-concurrency 5
```

Each output directory contains `results/judge_results.csv` and `results/judge_results.jsonl`. Lower `--max-concurrency` if needed for provider rate limits. Use the CSVs in the Fig. 4 command in section 2. A single judge CSV also works provided its judge is different from every selected agent; two judges allow self-judgment exclusion when an agent is itself one of the judges.

For the authors' archived experiments and paper-specific figures, see [Paper data and results](reproduce.md).
