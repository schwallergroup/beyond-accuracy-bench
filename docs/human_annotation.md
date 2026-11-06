# Human trace annotation

For the annotations reported in the paper, two chemistry labmates volunteered
their assistance without compensation and scored the assigned traces
independently. The authors considered formal ethics review unnecessary for
this voluntary assessment of model outputs, and no formal review was obtained.

This section is for researchers asked to independently annotate reasoning traces. The goal is to score the quality of the reasoning used to select an experiment, **not** whether the proposed experiment or the model ultimately identified the true mechanism. The assigned traces and manifest are already in this repository, so annotation is entirely local: it does **not** use Weave/W&B or any LLM API.

#### Before you begin

1. Complete [Environment Setup](setup.md), including `poetry install --no-root`.
2. You do not need a `.env`, `WANDB_API_KEY`, or `OPENROUTER_API_KEY`.
3. The app creates a random anonymous annotator ID automatically on first launch and remembers it locally for resuming. You do not need to request or enter an ID.

#### Run an annotation session

From the repository root, run the browser-based interface below:

```bash
poetry run python human_judge_reasoning_traces.py \
  --trace-dir bench/judge/runs/neurips_10models_Mstar0_1_iter3_4_human/traces
```

The app opens in a local browser. It reads the bundled `manifest.csv` automatically from the parent of the fixed `traces/` directory. On first launch, it creates a random anonymous ID, uses it to label the results, and writes the results under `bench/judge/human_runs/`. The same command automatically resumes that local assignment; you do not need to choose an output directory, locate the manifest, or enter an ID.

By default, the interface does not show the experiment tag, sample ID, model label, or experimental feedback. It presents the rubric, the model's raw reasoning trace, and a reference list of candidate-mechanism definitions. Assess only that displayed content. In particular, do not try to infer or look up the ground-truth mechanism or source model, and do not discuss or share your ratings with other annotators before submitting.

For each displayed trace and rubric:

1. Read the rubric definition, score anchors, and the trace.
2. Enter a `score` of `0`, `1`, or `2`.
3. Click **Save and show next rating**. The app checks that a score was selected and saves the result immediately.

The command above deliberately scores only these two rubrics for every trace:

- **Discriminative Experiment Design (DED):** whether the proposed experiment directly tests the feature identified in the reasoning.
- **Predictive Reasoning (PR):** whether it gives concrete, contrastive predictions for candidate mechanisms.

Every annotator independently scores the same 60 traces on both criteria (120 ratings total). Plan for approximately 3–4 hours including setup. The rubric text displayed by the program is authoritative for assigning scores. Briefly, `0` means the criterion is absent or unsupported, `1` means it is present but coarse, partial, or indirect, and `2` means it meets the rubric's specific, discriminative standard.

#### Pause, resume, and submit

- Close the browser tab or stop the local app with `Ctrl-C` to pause. Previously saved judgments remain on disk.
- To continue the same assignment, rerun the same command:

  ```bash
  poetry run python human_judge_reasoning_traces.py \
    --trace-dir bench/judge/runs/neurips_10models_Mstar0_1_iter3_4_human/traces
  ```

  The app automatically skips trace/rubric pairs already saved in `results/human_judge_results.jsonl`.
- When the session is complete, send only `bench/judge/human_runs/<anonymous-id>/results/human_judge_results.csv` to the author responsible for the annotation study. Do not edit the result file manually.
