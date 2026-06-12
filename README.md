# CIFAKE: Real vs. Deepfake Detection

A binary image classifier that distinguishes real images from AI-generated ("fake") images, built on a
Vision Transformer (ViT-B/16) with two-stage fine-tuning. The model is trained on real-vs-fake faces
and evaluated on the CIFAKE benchmark under two regimes: as a held-out out-of-distribution (OOD) test,
and with CIFAKE included in training.

## Summary

The model achieves **98.25% accuracy** on the in-distribution face test set and **97.68% accuracy**
(ROC-AUC 0.9977) on the CIFAKE test set when CIFAKE's training split is included in training, exceeding
the 92.98% baseline reported in the original CIFAKE study. When CIFAKE is held out as a strict
out-of-distribution benchmark, accuracy on CIFAKE falls to chance (49.99%), quantifying the
cross-generator generalization gap. Full figures are in [Results](#results).

## Table of contents

- [Architecture and configuration](#architecture-and-configuration)
- [Datasets](#datasets)
- [Method](#method)
- [Results](#results)
- [Analysis](#analysis)
- [Reproduction](#reproduction)
- [Limitations and future work](#limitations-and-future-work)
- [Security](#security)
- [Repository structure](#repository-structure)
- [References](#references)

## Architecture and configuration

| Component | Specification |
|---|---|
| Model | `vit_b_16` (ViT-Base, patch size 16), ImageNet pretrained (`ViT_B_16_Weights.DEFAULT`) |
| Input | 224x224 RGB, ImageNet normalization |
| Classification head | `Linear(768, 2)` |
| Training strategy | Two-stage: (1) linear probe (head only, frozen backbone); (2) fine-tune the last 4 encoder blocks and the final layer norm |
| Loss / optimizer | `CrossEntropyLoss` / `AdamW` (weight decay 1e-4); learning rate 1e-3 (probe) then 1e-5 (fine-tune) |
| Scheduling and stability | AMP mixed precision, `ReduceLROnPlateau`, early stopping, best-checkpoint selection by validation accuracy |
| Augmentation | Train: horizontal flip, light color jitter. Evaluation: none |
| Metrics | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix |
| Epochs / batch size | 2 (probe) + 3 (fine-tune) / 64 |
| Reproducibility | Global seed 42 |
| Export | ONNX (`deepfake_vit_detector.onnx`, opset 14, dynamic batch axis) |

## Datasets

| Dataset | Role | Images | Resolution | Generator |
|---|---|---|---|---|
| [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | Train / validation / test | 100k / 20k / 20k | 256x256 faces | StyleGAN |
| [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | Benchmark (train + test) | 100k / 20k | 32x32 objects | Stable Diffusion |

Both datasets use the label order `fake = 0`, `real = 1`, so labels are aligned and the cross-dataset
evaluation is valid.

## Method

The pipeline downloads both datasets via the Kaggle API, applies ImageNet preprocessing, and trains the
classifier in two stages:

1. **Linear probe.** The pretrained backbone is frozen and only the new `Linear(768, 2)` head is
   trained. This establishes a stable head before any backbone weights are perturbed.
2. **Fine-tuning.** The best probe checkpoint is restored, the last 4 transformer blocks and the final
   layer norm are unfrozen, and training continues at a low learning rate (1e-5). All evaluation uses
   the best-validation checkpoint.

Training uses horizontal-flip and color-jitter augmentation, AMP mixed precision, a plateau-based
learning-rate scheduler, early stopping, and a fixed seed. Evaluation reports accuracy, precision,
recall, F1, ROC-AUC, and a confusion matrix for both the in-distribution face test set and the CIFAKE
test set.

### Evaluation regimes

The notebook exposes an `INCLUDE_CIFAKE_IN_TRAIN` flag (in the data cell):

- `False` - face-only training; CIFAKE is a held-out out-of-distribution benchmark.
- `True` - CIFAKE's training split is concatenated with the face training set (~195k images total),
  making the CIFAKE test set in-distribution. A disjoint 5% slice of CIFAKE train is held out for
  validation, with augmentation applied only to the training portion.

### Changes from the initial baseline

| Area | Initial baseline | Current |
|---|---|---|
| Backbone training | Frozen (linear probe only) | Two-stage: probe, then fine-tune the last 4 blocks at 1e-5 |
| Optimizer | Adam, fixed lr 1e-3 | AdamW with `ReduceLROnPlateau` |
| Metrics | Accuracy only | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix |
| Augmentation | None | Horizontal flip and color jitter on train |
| Precision | FP32 | AMP mixed precision |
| Robustness | Fixed 3 epochs | Early stopping and best-checkpoint selection |
| Reproducibility | No seed | Global seed 42 |
| Credentials | API token hard-coded in a cell | Loaded from Colab Secrets; no secret in the file |
| Device | Ran on CPU without warning | Explicit device check with a warning when no GPU is present |
| CIFAKE usage | Test-only OOD benchmark | Optional inclusion of CIFAKE train via `INCLUDE_CIFAKE_IN_TRAIN` |

## Results

Both experiments ran on a Colab T4 GPU with identical two-stage fine-tuning, differing only in the
`INCLUDE_CIFAKE_IN_TRAIN` flag:

- **Experiment A - CIFAKE held out (OOD):** trained on 140k faces only.
- **Experiment B - CIFAKE in training:** trained on 140k faces and CIFAKE train (~195k images).

### Comparison

| Test set / metric | A: face-only (CIFAKE OOD) | B: CIFAKE in training |
|---|---|---|
| 140k faces, accuracy | 98.79% | 98.25% |
| 140k faces, ROC-AUC | 0.9993 | 0.9994 |
| CIFAKE, accuracy | 49.99% | 97.68% |
| CIFAKE, ROC-AUC | 0.4418 | 0.9977 |

Including CIFAKE's training split raises CIFAKE accuracy from chance to 97.68% while in-distribution
face accuracy decreases by only 0.54 points, yielding a single model competitive on both domains.

### Experiment A - CIFAKE held out (OOD)

| Metric | 140k Test (in-distribution) | CIFAKE (OOD) |
|---|---|---|
| Accuracy | 98.79% | 49.99% |
| Precision | 98.92% | 50.00% |
| Recall | 98.66% | 99.97% |
| F1 | 98.79% | 66.66% |
| ROC-AUC | 0.9993 | 0.4418 |
| Loss | 0.0323 | 6.5597 |

```
140k Test (in-distribution)         CIFAKE (out-of-distribution)
 [[9892   108]                       [[   2  9998]
  [ 134  9866]]                       [   3  9997]]
rows = true [fake, real], cols = predicted [fake, real]
```

On faces the model performs well, with balanced errors (108 fakes and 134 reals misclassified out of
20,000). On CIFAKE it predicts `real` for 19,995 of 20,000 images; the apparently high recall (99.97%)
and F1 (66.66%) are artifacts of this single-class collapse. The ROC-AUC of 0.4418 is below 0.5,
indicating that the score ranking is worse than random and that the face-domain artifact cue is mildly
anti-correlated with CIFAKE's Stable-Diffusion artifacts.

### Experiment B - CIFAKE in training

Validation accuracy rose from 85.83% after the linear probe to 98.13% after fine-tuning the last 4
transformer blocks.

| Metric | 140k Test (in-distribution) | CIFAKE Test (in-distribution) |
|---|---|---|
| Accuracy | 98.25% | 97.68% |
| Precision | 96.92% | 97.56% |
| Recall | 99.66% | 97.81% |
| F1 | 98.27% | 97.68% |
| ROC-AUC | 0.9994 | 0.9977 |
| Loss | 0.0492 | 0.0623 |

```
140k Test (in-distribution)         CIFAKE Test (in-distribution)
 [[9683   317]                       [[9755   245]
  [  34  9966]]                       [ 219  9781]]
rows = true [fake, real], cols = predicted [fake, real]
```

With CIFAKE in training, CIFAKE accuracy reaches 97.68% (ROC-AUC 0.9977), exceeding the 92.98% CNN
baseline from the original CIFAKE study. Face accuracy holds at 98.25%; the face error profile shifts
slightly toward predicting `real` (317 false `real` versus 108 in Experiment A), while F1 is
effectively unchanged at 98.27%.

## Analysis

- **Cross-generator generalization is weak.** A detector trained on one generator and domain (StyleGAN
  faces) provides no usable transfer to a different generator and domain (Stable-Diffusion objects).
  The below-chance ROC-AUC in Experiment A shows the failure is systematic rather than random noise.
- **Domain coverage is the deciding factor.** Adding in-domain training data closes the gap almost
  entirely, and at minimal cost to the original task, supporting a single multi-domain detector over
  separate per-domain models.
- **Metric selection matters.** Accuracy and F1 alone are misleading under class collapse; the
  confusion matrix and ROC-AUC are necessary to characterize behavior, as Experiment A demonstrates.

## Reproduction

The project is a Google Colab notebook (`CIFAKE_Real_vs_Deepfake_.ipynb`).

1. Set a GPU runtime: Runtime > Change runtime type > Hardware accelerator > GPU (T4).
2. Provide a Kaggle token through Colab Secrets (see below); do not paste it into a cell.
3. Run all cells top to bottom. Datasets download to `/content/data/`.
4. Set `INCLUDE_CIFAKE_IN_TRAIN` in the data cell to select the evaluation regime.
5. Training produces the metrics report and exports `deepfake_vit_detector.onnx`.

### Kaggle credentials

1. Open the Secrets panel in the Colab sidebar.
2. Add a secret named `KAGGLE_API_TOKEN` with your Kaggle token and enable notebook access.
3. The credentials cell reads it automatically; the token never appears in the notebook.

Alternatively, upload `kaggle.json` (Kaggle > Settings > API > Create New Token) and use the commented
lines in the credentials cell to move it to `~/.kaggle/`.

## Limitations and future work

- **CIFAKE resolution.** CIFAKE images are 32x32 and are upscaled to 224x224, which discards
  fine-grained generative artifacts. Native-resolution handling or a small-image architecture may
  improve results further.
- **Generator coverage.** Both generators (StyleGAN, Stable Diffusion) are fixed; robustness to unseen
  generators is not evaluated and is expected to be limited, consistent with Experiment A.
- **Backbone capacity.** A larger backbone (for example ViT-L/16) or a forgery-pretrained model may
  raise accuracy where compute permits.

## Security

The Kaggle API token is loaded from Colab Secrets and is not stored in the notebook. The repository
includes a `.gitignore` that excludes `kaggle.json`, model checkpoints (`*.pth`, `*.pt`), exported
models (`*.onnx`), dataset directories (`data/`), and archives (`*.zip`). Any token previously embedded
in the notebook should be revoked and replaced.

## Repository structure

```
.
├── CIFAKE_Real_vs_Deepfake_.ipynb   # Colab notebook: download, preprocess, two-stage fine-tune, evaluate, export ONNX
├── .gitignore                       # excludes secrets, checkpoints, exported models, and data
└── README.md
```

The trained model is exported as `deepfake_vit_detector.onnx` (ViT-B/16, input shape 1x3x224x224) for
portable inference.

## References

- 140k Real and Fake Faces dataset: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
- CIFAKE dataset: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- Bird, J. J. and Lotfi, A. "CIFAKE: Image Classification and Explainable Identification of
  AI-Generated Synthetic Images." IEEE Access, 2024.
- Dosovitskiy et al. "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale."
  ICLR, 2021.
