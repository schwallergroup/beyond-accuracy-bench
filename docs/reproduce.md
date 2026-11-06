# Paper data and results

This document catalogs the authors' saved results and their relationship to the paper. To run, visualize, and evaluate your own experiments, follow [the user workflow](run_experiments.md). Its [visualization section](run_experiments.md#2-visualize-your-results) generates the main figure types below from arbitrary tags using `plotting_scripts/visualize_results.py`.

## Repository contents

| Path | Contents |
| --- | --- |
| `bench/` | Simulator, fitting, experimental-design methods, and judge implementation |
| `bench/models/` | COPASI mechanism definitions M1–M20 |
| `ground_truths_200_samples_v2/` | The 200 sampled kinetic-parameter systems |
| `results/` | Archived benchmark and ablation experiments (`*.tar`); new runs write batch directories here |
| `bench/judge/runs/` | Reasoning traces, manifests, and LLM judge scores |
| `bench/judge/human_runs/` | Author and independent expert ratings |
| `plotting_scripts/neurips/` | Accuracy and exploration figures; shared analysis helpers |
| `plotting_scripts/neurips_rebuttal/` | Final reasoning figures, agreement analyses, and prompt ablations |

## Regenerate paper figures and tables

Complete [environment setup](setup.md), then run commands from the repository root. The figure/table numbers below refer to the camera-ready draft. These analyses read local archives and judge scores; they require no model API calls or Weave access.

## Main figures

| Paper result | Command | Output |
| --- | --- | --- |
| Fig. 2: accuracy, mechanism entropy, occupancy entropy, corner mass | `poetry run python plotting_scripts/neurips/baseline_accuracy_entropy_comparison.py` | `plotting_scripts/neurips/imgs_for_paper/baseline_accuracy_entropy_comparison.png` |
| Fig. 3: accuracy at iteration 5 | `poetry run python plotting_scripts/neurips/model_accuracy_at_iteration5.py` | `plotting_scripts/neurips/imgs_for_paper/model_accuracy_at_iteration5.png` |
| Fig. 4: reasoning scores and DED/PR–accuracy relationships | `poetry run python plotting_scripts/neurips_rebuttal/camera_ready_reasoning_score_figures.py` | `plotting_scripts/neurips_rebuttal/imgs_for_paper/camera_ready_reasoning_scores_iterations_2_to_4.png` and `camera_ready_ded_pr_vs_accuracy_iterations_2_to_4.png`; source tables in `iteration_analysis/` |

Archived experiments are in `results/*.tar`. The preferred main-run mapping is defined in `plotting_scripts/neurips/archived_experiment_data.py` (`PREFERRED_SOURCES_BY_LABEL`); paper model labels are in `paper_model_metadata.py` in the same directory. Keep archive filenames unchanged.

The release includes 26 selected archives: ten main LLM runs, three baseline
runs, four mechanism-information-withheld runs, and nine four-step runs.
The `.gitignore` allowlist retains these inputs; other results remain local.

Fig. 4 uses 20 mechanisms × two kinetic-parameter variants × iterations 2–4 = 120 traces per model, across 10 models. Each of the 1,200 traces is scored on four axes by GPT-5.4 and Claude Opus 4.7. Scores from a judge identical to the generating model are excluded; eligible scores are averaged per trace before model-level aggregation. The two judge result CSVs each contain 4,800 ratings.

For the pooled score error bars, resample the 40 independent reaction systems,
retaining their three iterations together. The figure script implements this
by bootstrapping the system-level three-iteration means. Model-level Spearman
intervals use paired resampling of the ten model aggregates. The iteration-3
values in the robustness table use this newer judging pass; historical
iteration-3 exclusion results and the standard-prompt side of Fig. 5 use the
earlier judging pass and have different coefficients.

The older `plotting_scripts/neurips/judge_rubric_scores_by_model.py` and `judge_score_accuracy_scatter.py` contain shared helpers and historical iteration-3 plotting entry points. Use the camera-ready command above for the final Fig. 4.

## Appendix results

| Paper result | Command or saved result |
| --- | --- |
| H.4, Table 5: iteration-specific reasoning–accuracy correlations | `poetry run python plotting_scripts/neurips_rebuttal/iteration_reasoning_score_report.py`; read `iteration_analysis/iteration_score_accuracy_spearman.csv` |
| I.1, Fig. 5: standard vs four-step instructions | `poetry run python plotting_scripts/neurips_rebuttal/step_instruction_standard_vs_step_scatter.py`; output `step_instruction_judge_analysis/standard_vs_step_instruction_accuracy_ded_pr_scatter.png` |
| I.1, Fig. 6: four-step reasoning vs accuracy | `poetry run python plotting_scripts/neurips_rebuttal/step_instruction_judge_score_accuracy_scatter.py`; output `step_instruction_judge_analysis/iteration3_pr_ded_vs_iteration5_accuracy_scatter.png` |
| I.1: four-step correlation estimates and confidence intervals | `poetry run python plotting_scripts/neurips_rebuttal/step_instruction_reasoning_accuracy_correlation.py`; output `step_instruction_judge_analysis/iteration3_pr_ded_vs_iteration5_accuracy_spearman.csv` |
| H.3, Table 4: expert–expert and both expert–LLM agreements | `poetry run python plotting_scripts/neurips_rebuttal/independent_expert_agreement_report.py`; output `judge_agreement_analysis/independent_annotator_and_llm_agreement_spearman.csv` |
| Appendix table: extreme-score expert audit | The same expert-agreement command; outputs `judge_agreement_analysis/blinded_expert_audit_llm_high_low.csv` and `blinded_expert_audit_llm_consensus_extremes.csv` |
| Appendix table: mechanism information withheld, accuracy | `poetry run python plotting_scripts/neurips_rebuttal/mechanism_info_accuracy_comparison.py`; output `mechanism_info_ablation/iteration5_accuracy_with_vs_without_mechanism.csv` |
| Appendix table: mechanism information withheld, exploration | `poetry run python plotting_scripts/neurips_rebuttal/mechanism_info_entropy_comparison.py`; output `mechanism_info_ablation/iteration5_entropy_with_vs_without_mechanism.csv` |
| Appendix table: nine-model paired prompting effects | `poetry run python plotting_scripts/neurips_rebuttal/paired_prompt_effects.py`; output `step_instruction_judge_analysis/nine_model_paired_prompt_effects.csv` |

All abbreviated output directories in this table are under `plotting_scripts/neurips_rebuttal/`.

Run `poetry run python plotting_scripts/neurips_rebuttal/independent_expert_agreement_report.py`
to reproduce the full two-expert agreement table and the consensus-extreme
audit. The two input files identify the two raters; their legacy anonymous-ID
column is the same in both exports and is not used as a rater identifier.
Thirty of the 60 expert-rated traces overlap with the
30-trace author calibration; author ratings are not pooled as independent expert
validation.

Run `poetry run python plotting_scripts/neurips_rebuttal/paired_prompt_effects.py`
for all nine models' paired accuracy and exploration changes. Condition means
use all available records, while paired changes use shared valid system IDs.
The output includes all three sample counts and 4,000-resample percentile
intervals (seed 0). Missing proposal histories reduce some exploration samples;
these exploratory intervals are not adjusted for multiple comparisons. The
ablation figures use the shared model-color map and include explicit legends.

The extreme-score audit reuses the 60 expert-rated traces, excluding the 12
traces generated by either judge model so that consensus uses two non-self
judges. The eligible subset contains 48 traces from eight models. Both judges
assign 2 on 21 DED traces and 17 PR traces, and 0 on three PR traces. No DED
trace receives 0 from both judges; the high/low table reports that empty group
explicitly. The consensus table also includes the DED score-1 group.

The mechanism-information-withheld Appendix tables compare four models with
their standard with-information runs. `MODEL_SOURCES` in the comparison script
fixes the withheld-information archives; standard runs use the preferred
main-run mapping.

The iteration-specific command writes `reasoning_scores_by_iteration.csv` and
`iteration_score_accuracy_spearman.csv`. The Fig. 4 command writes model-level
score and DED/PR correlation summaries; regression bands are drawn without an
additional band-coordinate CSV. The standard-vs-four-step command writes the
figure and its source data, without computing cross-prompt rank correlations.
Expert agreement has one entry point covering both experts and the audit;
no separate expert-1-only CSV is written.

For H.3 Tables 2–3, the author pilot contains 30 traces × four axes. A dedicated script reproducing both final tables is not currently included. For Appendix K's two representative traces, the source traces and judge results are included, but an exact selection manifest for the two published examples is not currently supplied.

## Judge data directory map

All paths in this table are under `bench/judge/`.

| Path | Role |
| --- | --- |
| `runs/standard_iteration2_3_4/traces/` | Main 1,200 traces; parent directory contains manifest and provenance/settings |
| `runs/standard_iteration2_3_4/results/gpt54/` | Main GPT-5.4 scores |
| `runs/standard_iteration2_3_4/results/claude47/` | Main Claude Opus 4.7 scores |
| `runs/standard_iter3/` | Historical standard-prompt 400 traces and both judges; `results/gpt54/` and `results/claude47/`, used by Fig. 5 |
| `runs/four_step_iter3/` | All nine four-step models: 360 traces and both judges; `results/gpt54/` and `results/claude47/`, used by Fig. 5/6 |
| `runs/neurips_10models_Mstar0_1_iter3_4_human/` | The 60 traces assigned to independent experts |
| `human_runs/neurips_10models_pilot_M12_0_M7_0_M2_1_iter3/` | Author pilot: 30 traces, four axes |
| `human_runs/human_judge_results_f6e6ad1c5a1f.csv` | First expert: 60 traces, DED and PR |
| `human_runs/human_judge_results_2.csv` | Second expert: the same 60 traces and two axes |

Fig. 5 uses an earlier standard-prompt judging pass. Its scores differ from the iteration-3 slice of the main iteration-2–4 pass; substituting the latter changes the result.

The main-analysis `traces/` folder contains the agents' reasoning traces.
The `results/gpt54/` and `results/claude47/` folders contain independent scores for the same
1,200 traces: four rubric ratings per trace, or 4,800 ratings per judge.

Four-step data are consolidated in `runs/four_step_iter3/`: nine models ×
40 systems × iteration 3 = 360 traces, with 1,440 ratings per judge. The
original collection used three batches of three models; their original run
configurations are preserved in `source_run_configs.json`. Consolidation
does not change scores or the original batch order.
`STEP_RUNS` in `step_instruction_reasoning_accuracy_correlation.py` specifies
the corresponding experiment archives. The consolidated historical standard
run likewise preserves its original configurations in `source_run_configs.json`.

For expert annotation, follow [the annotation instructions](human_annotation.md).
The app reads `runs/neurips_10models_Mstar0_1_iter3_4_human/traces` and its
parent manifest directly.

## Reproducibility limits

Saved-result reproduction avoids model API variability; rerunning agents or judges can change results as models, routing, or numerical environments change. See [experiment execution](run_experiments.md) and [judge evaluation](judge.md) for reruns.
