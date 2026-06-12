# CIFAKE — Real vs. Deepfake Detection

Detecting AI-generated / deepfake images using a **Vision Transformer (ViT-B/16)** fine-tuned on
real-vs-fake faces, with an **out-of-distribution (OOD) benchmark** on the CIFAKE dataset.

> **Status:** Notebook v2 ran on a **Colab T4 GPU** in two configs (full [Results](#results)). Faces
> score **~98%** either way; **CIFAKE goes from 49.99% (held out / OOD) → 97.68% when its train split
> joins training** (`INCLUDE_CIFAKE_IN_TRAIN`), beating the ~92.98% published CIFAKE CNN — one model,
> both domains.

---

## Table of contents
- [Overview](#overview)
- [Datasets](#datasets)
- [What I did — original baseline](#what-i-did--original-baseline)
- [Improvements applied (v2)](#improvements-applied-v2)
- [Results](#results)
- [Is this the best approach? — Review](#is-this-the-best-approach--review)
- [What would improve accuracy (roadmap)](#what-would-improve-accuracy-roadmap)
- [How to run](#how-to-run)
- [Security note](#security-note)
- [Repo structure](#repo-structure)

---

## Overview

The goal is a binary classifier: **`real` (1)` vs `fake` (0)`**. The model is a pretrained
Vision Transformer whose classification head is replaced with a 2-class layer. It is trained on a
large face dataset and then **benchmarked on a different generator/domain (CIFAKE)** to measure how
well it generalizes to images it was never trained on.

| | |
|---|---|
| **Model** | `vit_b_16` (ViT-Base, patch 16), ImageNet pretrained (`ViT_B_16_Weights.DEFAULT`) |
| **Input** | 224×224 RGB, ImageNet normalization |
| **Head** | `Linear(768 → 2)` |
| **Training strategy** | **Two-stage**: (1) linear probe — head only, frozen backbone; (2) fine-tune last 4 encoder blocks + final norm |
| **Loss / Optimizer** | `CrossEntropyLoss` / `AdamW` (`wd=1e-4`); `lr=1e-3` (probe) → `1e-5` (fine-tune) |
| **Training extras** | AMP mixed precision, `ReduceLROnPlateau`, early stopping, best-checkpoint saving, seed=42 |
| **Augmentation** | Train: h-flip + light color jitter. Eval / OOD: none |
| **Metrics** | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix |
| **Epochs / Batch** | 2 (probe) + 3 (fine-tune) / 64 |
| **Export** | ONNX (`deepfake_vit_detector.onnx`, opset 14, dynamic batch) |

---

## Datasets

| Dataset | Role | Images | Resolution | Fake source |
|---|---|---|---|---|
| [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | **Train / Val / Test** | 100k / 20k / 20k | 256×256 faces | StyleGAN |
| [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | **OOD benchmark (test only)** | 20k (test) | **32×32** objects | Stable Diffusion |

**Class mapping (consistent across both):** `fake → 0`, `real → 1`. ✅ Good — the label order matches,
so the OOD evaluation is valid.

---

## What I did — original baseline

> This is the **original** pipeline. See [Improvements applied (v2)](#improvements-applied-v2) for the
> upgraded version now in the notebook.

1. **Downloaded** the 140k faces dataset (3.75 GB) and CIFAKE via the Kaggle API.
2. **Preprocessing** — resize to 224×224, `ToTensor`, ImageNet `Normalize`. Same transform for both
   datasets so the OOD test is apples-to-apples with training.
3. **Model** — loaded ImageNet-pretrained `vit_b_16`, **froze all backbone parameters**, and replaced
   the 1000-class head with `Linear(768, 2)`.
4. **Training** — trained **only the head** (linear probe) with Adam (`lr=1e-3`), CrossEntropyLoss,
   3 epochs, batch size 64, tracking train/val accuracy + loss.
5. **OOD benchmark** — evaluated the trained model on the **CIFAKE test set** (a different generator
   *and* a different image domain) to measure generalization.
6. **Export** — exported the model to **ONNX** for portable inference.

**Design intent:** train on faces, then test on a totally different generator/domain (CIFAKE) to see
whether the learned "fake" signal transfers. This is a legitimate *generalization* study — but note
it is **not** the setup that produces the highest CIFAKE accuracy (see review below).

---

## Improvements applied (v2)

The notebook was upgraded from the linear-probe baseline above. Before → after:

| Area | Baseline | v2 (now in the notebook) |
|---|---|---|
| Backbone training | Frozen (linear probe only) | **Two-stage**: probe, then fine-tune last 4 blocks @ `lr=1e-5` |
| Optimizer | Adam, fixed `lr=1e-3` | **AdamW** + `ReduceLROnPlateau` scheduler |
| Metrics | Accuracy only | **Accuracy, precision, recall, F1, ROC-AUC, confusion matrix** |
| Augmentation | None | **H-flip + color jitter** on train |
| Speed | FP32 | **AMP mixed precision** (~2× on T4) |
| Robustness | 3 epochs, no stopping | **Early stopping + best-checkpoint saving** |
| Reproducibility | No seed | **Seed = 42** |
| Secrets | **Token hard-coded in cell 1** | **Loaded from Colab Secrets** (`KAGGLE_API_TOKEN`) — no secret in the file |
| GPU | Silently ran on CPU | Explicit **device check + warning** if no GPU |

**Observed effect:** in-distribution (140k) accuracy reached **98.79%** with **0.9993 ROC-AUC** — the
fine-tune hit the ~99% target. Full per-class metrics for both the in-distribution test set and the
CIFAKE OOD set are in [Results](#results).

> Still **not done** (deliberate — see roadmap): training *on* CIFAKE, native-resolution handling of
> 32×32 inputs, and a larger backbone.

---

## Results

Both experiments ran on a Colab T4 GPU with the same v2 two-stage fine-tuning; they differ **only** in
the `INCLUDE_CIFAKE_IN_TRAIN` toggle (cell *5. Data*):

- **Experiment A — CIFAKE held out (OOD):** trained on 140k faces only.
- **Experiment B — mixed (recommended):** trained on 140k faces **+ CIFAKE train** (~195k images).

### Headline comparison

| Test set / metric | A · face-only (CIFAKE = OOD) | B · mixed (CIFAKE in train) |
|---|---|---|
| 140k faces — accuracy | 98.79% | 98.25% |
| 140k faces — ROC-AUC | 0.9993 | 0.9994 |
| **CIFAKE — accuracy** | **49.99%** (chance) | **97.68%** |
| **CIFAKE — ROC-AUC** | **0.4418** | **0.9977** |

**Takeaway:** adding CIFAKE's train split lifts CIFAKE from **chance → 97.68%** (beating the published
**~92.98%** CIFAKE CNN) while face accuracy barely moves (**−0.54 pts**). One model can be strong on
**both** domains at almost no cost to the face task — Experiment B is the recommended setup.

---

### Experiment A — CIFAKE held out (OOD)

| Metric | 140k Test (in-dist) | CIFAKE (OOD) |
|---|---|---|
| **Accuracy** | **98.79%** | 49.99% |
| Precision | 98.92% | 50.00% |
| Recall | 98.66% | 99.97% |
| F1 | 98.79% | 66.66% |
| **ROC-AUC** | **0.9993** | 0.4418 |
| Loss | 0.0323 | 6.5597 |

```
140k Test (in-distribution)         CIFAKE (out-of-distribution)
 [[9892   108]                       [[   2  9998]
  [ 134  9866]]                       [   3  9997]]
```

- **Faces: excellent** — 98.79% accuracy, balanced errors (108 fakes + 134 reals missed of 20k).
- **CIFAKE: total failure** — predicts `real` for **19,995 / 20,000**. The high recall (99.97%) / F1
  (66.66%) are **artifacts of that collapse**. ROC-AUC **0.4418 < 0.5** means the ranking is *worse than
  random*: the StyleGAN-face artifact cue is mildly **anti-correlated** with CIFAKE's Stable-Diffusion
  artifacts. A detector trained on one generator + domain gives **zero usable transfer** to another —
  that cross-generator gap is the point of the OOD benchmark, not a bug.

---

### Experiment B — mixed (CIFAKE in training) · recommended

Trained on 140k faces + CIFAKE train (~195k images). Validation climbed **85.83%** (after the linear
probe) → **98.13%** (after fine-tuning the last 4 transformer blocks).

| Metric | 140k Test (in-dist) | CIFAKE Test (now in-dist) |
|---|---|---|
| **Accuracy** | **98.25%** | **97.68%** |
| Precision | 96.92% | 97.56% |
| Recall | 99.66% | 97.81% |
| F1 | 98.27% | 97.68% |
| **ROC-AUC** | **0.9994** | **0.9977** |
| Loss | 0.0492 | 0.0623 |

```
140k Test (in-distribution)         CIFAKE Test (in-distribution)
 [[9683   317]                       [[9755   245]
  [  34  9966]]                       [ 219  9781]]
```

- **CIFAKE jumps to 97.68%** (AUC 0.9977) once the model trains on its artifacts — and **beats the
  ~92.98%** CNN baseline from the original CIFAKE paper.
- **Faces hold at 98.25%** — only −0.54 pts vs Experiment A. The face errors shift slightly toward
  calling fakes `real` (317 vs 108), i.e. the mixed model leans marginally toward "real", but the F1 is
  effectively unchanged (98.27%).

---

## Is this the best approach? — Review

**Short answer: the original was a fast baseline, not tuned for accuracy. Most gaps are now fixed in
v2** — the remaining ones are deliberate trade-offs. Findings, highest impact first:

| # | Issue | Impact | Status |
|---|---|---|---|
| 1 | **Ran on CPU, not GPU** — `cuda.is_available()` was `False`, so nothing trained to completion. | 🔴 Blocker | ✅ **Done** — re-ran on a T4 GPU; full training completed. Notebook also warns if no GPU. |
| 2 | **Linear probing only** — caps accuracy. | 🔴 High | ✅ **Fixed** — two-stage fine-tuning (last 4 blocks @ `lr=1e-5`). |
| 3 | **CIFAKE is an OOD test the model can't win** — faces (StyleGAN, 256px) → objects (SD, 32px). | 🔴 High | ◻️ **By design** — kept as an OOD generalization benchmark; report the drop as a result. |
| 4 | **32×32 → 224×224 upscaling** loses the fine generative artifacts. | 🟡 Med | ◻️ **Open** — would need CIFAKE-native handling / a small-image model. |
| 5 | **Accuracy was the only metric.** | 🟡 Med | ✅ **Fixed** — precision, recall, F1, ROC-AUC + confusion matrix. |
| 6 | **No data augmentation.** | 🟡 Med | ✅ **Fixed** — h-flip + color jitter on train. |
| 7 | **Fixed LR, no scheduler / early-stopping / checkpointing.** | 🟡 Med | ✅ **Fixed** — `ReduceLROnPlateau`, early stop, best-checkpoint save. |
| 8 | **No mixed precision (AMP).** | 🟢 Low | ✅ **Fixed** — AMP enabled on GPU. |
| 9 | **No seed** — not reproducible. | 🟢 Low | ✅ **Fixed** — seed = 42. |
| 10 | **Hard-coded Kaggle token in the notebook.** | 🔴 Security | ✅ **Removed** — loads from Colab Secrets. **Revoke the old token** (see below). |

---

## What would improve accuracy (roadmap)

✅ = done · ◻️ = still open / your call.

1. ✅ **Fine-tune instead of linear-probe** — two-stage (probe → unfreeze last 4 blocks @ `lr=1e-5`).
2. ✅ **Full metrics** — precision, recall, F1, ROC-AUC, confusion matrix on test + OOD.
3. ✅ **Augmentation** — h-flip + color jitter on train.
4. ✅ **Training hygiene** — AdamW, LR scheduler, early stopping, best-checkpoint, seed, AMP.
5. ✅ **Secret-free auth** — Kaggle token via Colab Secrets.
6. ✅ **Ran on GPU** — completed on a Colab T4; in-distribution result **98.79% acc / 0.9993 AUC**.
7. ✅ **Match the benchmark to the goal** — `INCLUDE_CIFAKE_IN_TRAIN` toggle (cell *5. Data*):
   - `True` → CIFAKE's **train** split joins training; the CIFAKE test set becomes **in-distribution**
     (expect high accuracy). *Re-run to populate the CIFAKE column in [Results](#results).*
   - `False` → CIFAKE held out as the **OOD** generalization benchmark (the documented 49.99% run).
8. ◻️ **Native-resolution CIFAKE** — avoid blowing 32×32 up to 224×224, or use a small-image model.
9. ◻️ **(Stretch) Stronger backbone** — ViT-L/16 or a forgery-pretrained model, if compute allows.

---

## How to run

This is a **Google Colab** notebook (`CIFAKE_Real_vs_Deepfake_.ipynb`).

1. **Set a GPU runtime:** *Runtime → Change runtime type → Hardware accelerator → GPU (T4)*.
2. **Add your Kaggle token securely** (do **not** paste it in a cell — see below).
3. Run the cells top to bottom. Datasets download to `/content/data/`.
4. After training, the benchmark + ONNX export cells produce the final artifacts.

### Kaggle token — use Colab Secrets (no hard-coding)
1. Click the **🔑 (Secrets)** icon in Colab's left sidebar.
2. Add a secret named `KAGGLE_API_TOKEN` with your token, and enable notebook access.
3. Cell 2 reads it automatically — the token never appears in the notebook.

Alternative: upload `kaggle.json` (Kaggle → Settings → API → Create New Token); the commented lines
in cell 2 move it to `~/.kaggle/`.

---

## Security note

✅ **Fixed in v2 — the token is no longer in the notebook;** it's loaded from Colab Secrets.

**You still must:**
1. **Revoke the old token** — it existed in plaintext in the previous version: Kaggle →
   *Settings → API → Expire API Token*, then create a new one.
2. Store the new token in **Colab Secrets** as `KAGGLE_API_TOKEN` (never paste it into a cell).
3. A [`.gitignore`](.gitignore) now excludes `kaggle.json`, `*.pth`, `*.onnx`, `data/`, and `*.zip`.

---

## Repo structure

```
.
├── CIFAKE_Real_vs_Deepfake_.ipynb   # Colab notebook (v2): download → preprocess → 2-stage fine-tune → metrics → CIFAKE OOD → ONNX
├── .gitignore                        # excludes secrets, checkpoints, data
└── README.md
```

---

*Model exported as `deepfake_vit_detector.onnx` (ViT-B/16, 1×3×224×224 input) for portable inference.*
