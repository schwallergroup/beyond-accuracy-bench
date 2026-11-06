# Saved judge runs

| Directory | Purpose | Traces | Ratings per judge |
| --- | --- | ---: | ---: |
| `standard_iteration2_3_4/` | Main analysis, ten models, iterations 2–4; both judges | 1,200 | 4,800 |
| `standard_iter3/` | Historical standard-prompt pass, ten models; both judges | 400 | 1,600 |
| `four_step_iter3/` | Four-step prompting, all nine models; both judges | 360 | 1,440 |
| `neurips_10models_Mstar0_1_iter3_4_human/` | Fixed expert annotation assignment | 60 | Expert ratings are under `../human_runs/` |

Traces are produced by the evaluated agents. GPT and Claude independently
assign four rubric scores to each of those same traces.

All three standard/four-step analysis directories use:

```text
traces/                         # agent reasoning
manifest.csv                    # trace selection
results/gpt54/judge_results.csv
results/gpt54/judge_results.jsonl
results/claude47/judge_results.csv
results/claude47/judge_results.jsonl
run_config.json
source_run_configs.json         # original collection settings
```

The historical standard pass remains separate from the main iteration-2–4
pass because their scores differ. Fig. 5 uses that historical standard pass.
The four-step directory merges the three original batches without changing
scores: each former batch had 120 traces and 480 ratings per judge.

See [reproduction](../../../docs/reproduce.md) and
[expert annotation](../../../docs/human_annotation.md) for commands.
