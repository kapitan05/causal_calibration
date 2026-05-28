# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- **Lint & format:** `uv run ruff check . --fix && uv run ruff format .`
- **Type check:** `uv run mypy .` (strict mode, must pass with zero errors)
- **Add dependency:** `uv add <package>` (do NOT use pip)
- **Run code:** `uv run python <script.py>` or `uv run jupyter notebook`

No test suite currently exists.

## Architecture

Research project on **causal calibration for image classifiers** (ResNet50 and ViT-B/16). The pipeline is:

```
Images → Models → Attribution maps → Perturbation sequences → Causal evaluation → Calibration → Metrics
```

**`src/` modules:**

- **`models.py`** — loads pretrained ResNet50 / ViT-B/16 with ImageNet preprocessing
- **`attribution.py`** — `AttributionPipeline` wrapping 8 methods: Saliency, IntegratedGradients, Occlusion, LIME, GradientShap, ShapleyValueSampling, FeatureAblation, RISE
- **`RISE.py`** — RISE attribution implementation (randomized input sampling)
- **`generators.py`** — `TopNDeletion`, `BucketDeletion`, `TopNInsertion`, `BucketInsertion`; implements `SequenceGenerator` protocol; bucketing groups pixels by saliency rank into equal-count buckets
- **`causal_tests.py`** — `evaluate_causal_metric()`: runs a model over a perturbation sequence and returns confidence curve + AUC
- **`calibration.py`** — `ReCalXModel` (temperature-scaled wrapper); fits one temperature per perturbation bin (0–10 bins); `collect_logits_pipeline()` collects binned logits for fitting
- **`metrics.py`** — `calculate_ece()`, `calculate_tace()` for evaluating calibration quality
- **`visualizations.py`** — `visualize_tests()` (Insertion/Deletion curves), `plot_ece_curves()`

Experiments live in `notebooks/`.
