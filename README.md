
# README

Separation and Classification of Heart–Lung Audio from Single-Channel Recordings
*(HLS-CMDS Manikin Dataset)* 

---

## 📌 Project Overview

This repository implements a full pipeline for **separating mixed heart–lung audio** and **classifying each separated track**, using the *Heart & Lung Sounds Dataset Recorded from a Clinical Manikin (HLS-CMDS)*.

The workflow is organized into five stages:

1. **Data Loading & Normalization**
2. **Preprocessing & Feature Engineering**
3. **Source Separation** (masking, NMF, smoothing, deep models)
4. **Feature-space Prediction & Reconstruction**
5. **Downstream Classification & Evaluation**

All notebooks live under `notebooks/`. All reusable code lives under `src/`.

---

## 📂 Repository Structure

### **data/**

HLS-CMDS dataset arranged by source type:

* `heart_sounds/` — 50 clean heart-only WAV files
* `lung_sounds/` — 50 clean lung-only WAV files
* `mixed_sounds/` — 145 mixtures, each with:

  * `heart_ref/` (ground-truth heart)
  * `lung_ref/` (ground-truth lung)
  * `mixed_ref/` (single-channel mixture)

### **documents/**

* `project_proposal.pdf` — motivation, scope, and expected results
* `README.txt`, `prompt.txt` — dataset-level annotation from HLS-CMDS
   

### **notebooks/**

Contains all analysis artifacts:

* **data_analysis/** — distribution plots, dataset checks, spectrogram examples
* **feature_engineering/** — extraction of MFCCs, spectral features, ANOVA sensitivity
* **sound_type_predictions/** — logistic regression / RF models on features
* **source_reconstruction/** — mask-based and NMF reconstruction
* **model_weights/** — currently includes `heart_unet.pth`
* **outputs/**

  * `classification/heart/` and `classification/lung/` — metrics, CIs

### **src/**

All reusable pipeline code:

#### **src/audio/**

Low-level audio I/O utilities (load WAV, resample, normalization)

#### **src/features/**

Feature extraction (MFCCs, spectral stats, zero crossings, energy bands)

#### **src/metadata/**

Dataset metadata loading, joins, and visualization tools

#### **src/classification/**

Cross-validated evaluation, CI estimation, confusion matrices, model wrappers

#### **src/feature_prediction/**

Random-forest regressors that predict latent heart/lung features from mixed sources

#### **src/reconstruction/**

Feature-space → waveform reconstruction, NMF utilities, masks, smoothing

#### **src/prediction_pipeline/**

End-to-end pipeline:
mixed WAV → feature extraction → RF mapping → classifier → predictions

---

## ▶️ How to Run

### **1. Install dependencies**

```bash
pip install -r requirements.txt
```

This will install:

* `numpy`, `scipy`, `librosa`
* `pandas`, `matplotlib`, `seaborn`
* `scikit-learn`
* `tqdm`
* `torch` (for UNet-based separation)

### **2. Recommended execution order**

#### **Stage 1 — Data Exploration**

`notebooks/data_analysis/data_analysis.ipynb`
Checks distributions, metadata consistency, spectrograms, and audio sanity.

#### **Stage 2 — Feature Engineering**

`notebooks/feature_engineering/feature_engineering.ipynb`
Extracts MFCC/spectral features and runs ANOVA factor-sensitivity.

#### **Stage 3 — Source Separation**

`notebooks/source_reconstruction/source_reconstruction.ipynb`
Runs:

* STFT masking
* NMF (standard & KL divergence)
* optional UNet baseline

Outputs reconstructed heart/lung WAV files.

#### **Stage 4 — Feature Prediction / Regression**

`notebooks/source_feature_prediction.ipynb`
Random-forest regression to map mixed features → clean heart/lung features.

#### **Stage 5 — Classification**

`notebooks/classification.ipynb`
Logistic/Random-Forest/GMM/KNN, K-fold CV, bootstrap confidence intervals.

---

## 📝 Formatting Notes

* All notebooks follow the same structure:
  **Goal → Methods → Experiments → Results → Discussion → Next Steps**
* Code is modular: all logic lives in `src/`, notebooks call these functions.
* Figures are exported automatically to `notebooks/outputs/`.

---

## 📚 References

Dataset and proposal descriptions are included:

* Proposal: 
* Dataset README: 
