# Saved judge runs

| Directory | Purpose | Traces | Ratings per judge |
| --- | --- | ---: | ---: |
| `standard_iteration2_3_4/` | Main analysis, ten models, iterations 2–4; both judges | 1,200 | 4,800 |
| `four_step_ablation_iteration2_3_4/` | Four-step prompting, nine models, iterations 2–4; both judges | 1,078 | 4,312 |
| `neurips_10models_Mstar0_1_iter3_4_human/` | Fixed expert annotation assignment | 60 | Expert ratings are under `../human_runs/` |

Traces are produced by the evaluated agents. GPT and Claude independently
assign four rubric scores to each of those same traces.

Both standard/four-step analysis directories use:

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

All current standard versus four-step comparisons use iterations 2–4. Two
DeepSeek iteration-4 traces are unavailable; see the four-step run README.

See [reproduction](../../../docs/reproduce.md) and
[expert annotation](../../../docs/human_annotation.md) for commands.
