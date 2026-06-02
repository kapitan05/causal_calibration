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
- **`attribution.py`** — `AttributionPipeline` wrapping 8 methods: `saliency`, `ig`, `occlusion`, `lime`, `gradientshap`, `svs`, `feature_ablation`, `rise`. Returns a 2D saliency map `[H, W]` (channels summed). RISE initializes 5000 masks on construction — this is slow. RISE hardcodes `.cuda()`, so it requires a GPU.
- **`RISE.py`** — RISE attribution implementation (randomized input sampling for explanation)
- **`generators.py`** — implements `SequenceGenerator` protocol; all generators return `(Tensor[steps, C, H, W], List[float])` where the list contains per-step perturbation fractions:
  - `TopNDeletion` / `TopNInsertion` — linear pixel-by-pixel removal/reveal (e.g. 1% per step); baseline for insertion is Gaussian-blurred image
  - `BucketDeletion` / `BucketInsertion` — rank-based equal-count buckets; robust to sparse/skewed saliency maps (e.g. IG, Saliency); preferred over TopN for calibration
  - `BucketInpaintingGenerator` — stub, not implemented yet
- **`causal_tests.py`** — `evaluate_causal_metric(model, sequence_tensor, perturbation_levels, target_class)`: runs a model over a perturbation sequence and returns `(confidence_curve, auc_score, all_probs_np)`. Calls `model.set_perturbation_levels()` if present (i.e. `ReCalXModel`).
- **`calibration.py`** — `ReCalXModel` (temperature-scaled wrapper): holds one learned temperature per perturbation bin; call `set_perturbation_levels(list[float])` before each forward pass so the right bin temperature is applied. `collect_logits_for_calibration()` collects binned raw logits over a dataset. `train_and_save_temperatures()` fits temperatures via L-BFGS-B and saves a CSV. Note: this module uses `pandas` for CSV output (not polars).
- **`metrics.py`** — `calculate_ece()`, `calculate_tace()`, `process_calibration_metrics()` (aggregates ECE/TACE across all bins for base vs calibrated model)
- **`visualizations.py`** — `visualize_tests()` (Insertion/Deletion curves), `plot_ece_curves()`

Experiments live in `notebooks/`. Notebook `03_calibration_test.ipynb` is the primary experiment.
