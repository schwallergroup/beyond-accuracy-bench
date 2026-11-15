# Four-step ablation: iterations 2–4

This run combines the original iteration-3 data with newly fetched and judged iteration-2 and iteration-4 traces. The combined files are the canonical data for analysis.

- `manifest.csv` and `traces/`: available combined traces (360 at iteration 2, 360 at iteration 3, 358 at iteration 4).
- `results/gpt54/` and `results/claude47/`: combined CSV and JSONL scores, 4,312 rows per judge (1,078 traces × 4 rubrics).
- `manifest_iteration2_4.csv`: the intended 720 new traces; `available_manifest.csv` lists the 718 obtained.
- `unavailable_traces.json` records the two missing iteration-4 Weave calls for DeepSeek V3.2 (M7_1 and M13_1).
- `fetch_remaining.py` records the retrieval step. Original iteration-3 source configurations are in `source_run_configs.json`; the iteration-2/4 judge configurations and prompt dumps are in the respective `results/` judge directories.

Both judges scored all available traces, including self-judgments. Apply the same self-judge exclusion policy as the standard analysis when comparing results. For complete paired comparisons across all three iterations, account for the two missing DeepSeek traces.
