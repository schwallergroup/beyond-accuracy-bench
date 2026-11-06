# Environment setup

The code was developed for Python 3.11.

```bash
conda create -n kineticmechanismbench python=3.11
conda activate kineticmechanismbench
pip install poetry
poetry env use "$(which python)"
poetry install --no-root
```

All commands below should be run from the repository root with `poetry run`.

For LLM-based experimental-design methods, trace downloads, or LLM-as-a-judge reruns, create a local `.env` file:

```text
OPENROUTER_API_KEY=your_openrouter_api_key
WANDB_API_KEY=your_wandb_or_weave_api_key
```

`OPENROUTER_API_KEY` is required for the LLM method and LLM-as-a-judge reruns. `WANDB_API_KEY` is required to download archived traces from Weave and when Weights & Biases Weave logging is enabled. Add `--no-weave` to `run_experiment.py` commands to disable Weave logging entirely; this mode does not require a Weave/W&B account and was used for local offline verification. Baseline methods do not call LLM APIs. Human annotation of already downloaded traces requires neither key.

Continue with [running and evaluating your own experiments](run_experiments.md#1-run-an-experiment).
