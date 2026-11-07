# Provenance and attribution

The accompanying paper is by Ryo Kuroki, Amin Mansouri, and Philippe Schwaller. Ryo Kuroki and Amin Mansouri contributed equally to the paper. This repository is the public camera-ready release of the benchmark developed as KineticMechanismBench.

## Mechanisms and synthetic systems

The 20 candidate mechanism families implemented in `bench/models/` are described in Jordi Burés and Igor Larrosa, **Organic reaction mechanism classification using machine learning**, Nature 613, 689–695 (2023), [doi:10.1038/s41586-022-05639-4](https://doi.org/10.1038/s41586-022-05639-4). Please credit that source when using the mechanism families.

The authors generated the benchmark systems by sampling kinetic parameters for these mechanism families and applying the validation filters described in the paper and `generate_ground_truths.py`. The 200 systems in `ground_truths_200_samples_v2/` are synthetic parameterizations, not laboratory measurements. The [Hugging Face dataset](https://huggingface.co/datasets/xq8wvm/HypothesisDrivenExperimentPlanningBench) aggregates these systems into JSONL and provides Croissant metadata with an immutable source URL, checksum, and Responsible AI fields. Both this repository and the hosted dataset declare CC-BY-4.0; see the root license and dataset card.

## Experiments and annotations

The 26 archives in `results/` preserve the paper's main experiments, baselines, and ablations. Run configurations, selected trace manifests, LLM judgments, and author/independent expert ratings are retained as described in [Paper data and results](reproduce.md). LLM-generated traces and scores depend on the model versions and settings recorded with those runs. The independent expert ratings concern existing synthetic benchmark traces; they are not additional chemical measurements.

## Software dependencies

Simulation uses [COPASI](https://copasi.org/) through `copasi-basico`; fitting uses CMA-ES, and analysis uses NumPy, SciPy, pandas, scikit-learn, and Matplotlib, among the dependencies listed in `pyproject.toml` and pinned in `poetry.lock`. These third-party packages retain their own licenses. COPASI's foundational reference is Hoops et al., **COPASI—a COmplex PAthway SImulator**, Bioinformatics 22, 3067–3074 (2006), [doi:10.1093/bioinformatics/btl485](https://doi.org/10.1093/bioinformatics/btl485).
