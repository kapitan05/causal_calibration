# Causal Calibration

Research on **ReCalX** — perturbation-level temperature scaling for image classifiers evaluated via causal deletion sequences. The central question: does learning a separate calibration temperature per perturbation bin reduce miscalibration when confidence is measured along a deletion trajectory?

## Method

For each image an attribution method produces a saliency map. A deletion generator uses that ranking to build a sequence of increasingly degraded images. The raw model is run on every step, yielding a confidence curve. **ReCalX** fits one temperature scalar T per perturbation bin (10 equal-width bins over [0, 1]) by minimising cross-entropy on the same sequences, then post-hoc divides logits by the bin's T before softmax. Calibration quality is measured with ECE and TACE.

## Reproducibility

All results are derived from 20 ImageNet images sampled with a fixed seed:

```python
import random, numpy as np, torch
random.seed(42); torch.manual_seed(42); np.random.seed(42)
indices = random.sample(range(len(hf_dataset)), 20)
```

Dataset: ImageNet-1k via HuggingFace (`ILSVRC/imagenet-1k`), parquet shard `train-00000-of-00294.parquet`.

Preprocessing: `Resize(256) → CenterCrop(224) → ToTensor → Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`.

**All calibration is in-sample** — the same 20 images are used for fitting temperatures and evaluating ECE/TACE. Results represent in-sample calibration performance, not generalisation.

## Models

| Model | Source | Notes |
|-------|--------|-------|
| ResNet50 | `torchvision.models.resnet50` | Primary model, Sections 2–9 |
| ViT-B/16 | `torchvision.models.vit_b_16` | Section 7, 7b |
| ResNet101 | `torchvision.models.resnet101` | Section 10 |
| ViT-B/32 | `torchvision.models.vit_b_32` | Section 10 |

All models use pretrained ImageNet weights with all parameters frozen.

## Attribution Methods

RISE, IntegratedGradients, Saliency, GradientShap, FeatureAblation, LIME — all accessed through `AttributionPipeline` in `src/attribution.py`. RISE requires a GPU (hardcodes `.cuda()`).

## Experiments

Primary notebook: `notebooks/03_calibration_test.ipynb`

| Section | Question |
|---------|----------|
| 2 | Single image: deletion curve, temperature profile, reliability diagram |
| 3 | Batch ECE/TACE with RISE + globally fitted temperatures (20 images) |
| 4 | Saliency scale sanity check across methods |
| 5 | Attribution method comparison: AUC, ECE, TACE (ResNet50, 6 methods) |
| 6 | TopN(1%) vs Bucket(25) deletion generator comparison |
| 7 | ResNet50 vs ViT-B/16: ECE/TACE + temperature profiles |
| 7b | ECE/TACE for ResNet50 vs ViT-B/16 across all 6 attribution methods + Wilcoxon test |
| 8 | Per-method Spearman r and monotonicity ratio (raw vs calibrated) |
| 9 | Wilcoxon test: does ReCalX significantly change Spearman r (deletion monotonicity)? |
| 10 | 4-architecture comparison: ResNet50, ResNet101, ViT-B/16, ViT-B/32 |
| 11 | TopN vs Bucket: Spearman + monotonicity + statistical significance |

## Running

```bash
# Install dependencies
uv sync

# Launch notebook
uv run jupyter notebook notebooks/03_calibration_test.ipynb
```

Sections are designed to run sequentially. Section 8 can be run independently — it will automatically recompute batch temperatures from RISE saliency maps if Section 3 hasn't run yet.
