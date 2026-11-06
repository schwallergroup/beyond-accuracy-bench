# Judge evaluation

The complete run → visualization → LLM judge workflow is documented in [Run and evaluate your own experiments](run_experiments.md). This reference also covers deriving selections from archived manifests. Your agent run must have been recorded in Weave; runs made with `--no-weave` cannot be downloaded by this pipeline. Configure both API keys as described in [environment setup](setup.md). For archived paper scores and their locations, see [Paper data and results](reproduce.md).

Replace `YOUR_BATCH_TAG` below with the tag printed by your agent run and `My agent` with your model label. Adjust sample indices and iterations to the samples and steps actually completed by that run. Select only LLM agent runs with reasoning traces.

Create a balanced judge manifest:

```bash
poetry run python bench/judge/create_manifest.py \
  --gt-dir ground_truths_200_samples_v2 \
  --sample-indices 0 1 \
  --iterations 3 \
  --tag-label YOUR_BATCH_TAG "My agent" \
  --output bench/judge/manifests/example_manifest.csv
```

To reuse the tag/model-label mapping of an existing manifest while changing
sample indices or iterations, derive a new manifest instead of re-entering all
tags. The command writes both the CSV and a sibling `.provenance.json` file
recording the source-manifest hash and all selection arguments:

```bash
poetry run python -m bench.judge.derive_manifest \
  --source-manifest bench/judge/runs/standard_iter3/manifest.csv \
  --sample-indices 0 1 \
  --iterations 2 3 4 \
  --output bench/judge/manifests/neurips_10models_Mstar0_1_iter2_3_4_llm.csv
```

When different iterations need different sample sets, use explicit per-iteration
selections. For example, the human-annotation set contains `M2_1`, `M7_0`, and
`M12_0` at iteration 3, and a separate set—`M3_0`, `M9_0`, and `M15_0`—at
iteration 4:

```bash
poetry run python -m bench.judge.derive_manifest \
  --source-manifest bench/judge/runs/standard_iter3/manifest.csv \
  --sample-ids-by-iteration \
    3:M2_1,M7_0,M12_0 \
    4:M3_0,M9_0,M15_0 \
  --output bench/judge/manifests/neurips_10models_Mstar0_1_iter3_4_human.csv
```

Download the traces selected by the manifest from Weave. This is the only judge-pipeline command that requires `WANDB_API_KEY`:

```bash
poetry run python download_reasoning_traces.py \
  --manifest bench/judge/manifests/example_manifest.csv \
  --output-dir bench/judge/runs/example_trace_run
```

Run automated judge scoring using the downloaded traces. This command never queries Weave:

```bash
poetry run python judge_reasoning_traces.py \
  --manifest bench/judge/manifests/example_manifest.csv \
  --trace-dir bench/judge/runs/example_trace_run/traces \
  --output-dir bench/judge/runs/example_judge_run \
  --judge-model gpt-5.4 \
  --max-concurrency 5
```

The pipeline is intentionally separated as `manifest → downloaded traces → LLM judge / human judge`. The LLM judge runs up to five independent API calls concurrently by default; lower `--max-concurrency` if your provider rate limit requires it. Both judging commands validate that each local trace matches the manifest row before scoring it. The archived judge outputs used by the paper are already included, so the paper judge figures can be regenerated without rerunning the judge API calls.

The paper main analysis uses iterations 2–4 (1,200 traces), both GPT-5.4 and Claude Opus 4.7, and four rubric axes. Exclude a judge when it is identical to the trace-generating model, then average eligible judge scores per trace. This is implemented in `plotting_scripts/neurips_rebuttal/iteration_reasoning_score_analysis.py`. The iteration-3 manifest example above illustrates creating a new selection; it is not the full paper evaluation.

Per-trace prompt dumps are excluded from Git. Fixed judge templates and rubrics remain in `bench/judge/prompts.py` and `rubrics.py`; saved scores and their existing result JSONL files remain available.
