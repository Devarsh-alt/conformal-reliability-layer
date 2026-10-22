# Proactive Reliability Layer for Conformal Prediction

A **post-hoc, model-agnostic reliability layer** for sequential financial forecasting. The framework estimates the probability that an individual conformal prediction interval will fail, using information available at prediction time.

## Framework

```text
Market Data
    ↓
Black-Box Forecasting Model
    ↓
Point Prediction + Conformal Interval
    ↓
Reliability Signals
    ├── Model Disagreement (D)
    └── Perturbation Sensitivity (PS)
    ↓
Reliability Model
    ↓
Prediction-Level Failure Risk
    ↓
Selective Trust
```

## Current Implementation

### Base Model

Currently implemented using **LightGBM** with:

* Log_Return
* return_lag1
* return_lag5
* return_lag10
* vol_10d
* vol_20d
* volume_ratio
* VIX

Predictions are generated using strict walk-forward evaluation.

### Conformal Prediction

The target is next-day conformal interval miscoverage:

$$
M_{t+1} =
\mathbf{1}(y_{t+1}\notin C_t)
$$

The reliability layer estimates:

$$
P(M_{t+1}=1\mid X_t)
$$

where only information available at prediction time is used.

### Reliability Signals

**Model Disagreement (D)**
Measures dispersion between predictions from multiple fitted model versions.

**Perturbation Sensitivity (PS)**
Perturbs the current feature vector using historically estimated feature-difference covariance and measures how much the fixed model's prediction changes.

Primary PS configuration:

```text
Covariance window = 60
Perturbation scale = 0.50
Perturbations = 30
```

Both signals are causally percentile-normalized.

## Reliability Model

Current primary architecture:

```text
D + PS
   ↓
Logistic Regression
   ↓
Conformal Failure Risk
```

Architectures tested include:

```text
Constant
D
PS
D + PS
```

Additional candidate signals previously evaluated include historical miscoverage, KS drift, feature novelty, and conformal interval width.

## Current Results

Strict OOS evaluation:

* **33,024 predictions**
* Brier Score: **0.0877**
* Log Loss: **0.3168**
* ROC-AUC: **0.603**
* PR-AUC: **0.144**

Selective prediction shows that predictions assigned higher failure risk contain a larger proportion of actual conformal failures.

For example:

| Predictions retained | Miscoverage |
| -------------------: | ----------: |
|                 100% |       9.83% |
|                  90% |       8.86% |
|                  75% |       8.07% |
|                  50% |       7.17% |
|                  25% |       7.03% |
|                  10% |       6.69% |

## Repository Structure

```text
reliability_layer/
├── data/
│   ├── reliability/
│   └── signals/
│
├── src/
│   ├── base_model.py
│   ├── conformal.py
│   ├── signals/
│   │   └── compare_signals.py
│   └── reliability/
│       ├── nested_ablation.py
│       ├── perturbation_sensitivity.py
│       └── fixed_ablation_d_ps.py
│
└── ...
```

## Technical Status

### Completed

* [x] Walk-forward LightGBM forecasting
* [x] Adaptive conformal prediction
* [x] Pointwise miscoverage target
* [x] Model Disagreement
* [x] Perturbation Sensitivity
* [x] Causal signal normalization
* [x] D vs PS vs D+PS ablation
* [x] PS robustness analysis
* [x] Strict OOS evaluation
* [x] Selective prediction analysis
* [x] Failure concentration analysis
* [x] Calibration diagnostics

### Remaining

* [ ] Formal historical-failure baselines
* [ ] Generalize reliability layer to other black-box models
* [ ] Test with a second model family
* [ ] Cross-model robustness experiments
* [ ] Regime-level analysis
* [ ] Final leakage audit
* [ ] Freeze methodology and perform final untouched evaluation
* [ ] Consolidate experiments into a reproducible end-to-end pipeline
* [ ] Generate final tables and figures for the paper

## Core Objective

The final system is intended to answer:

> **“How much should we trust this particular black-box forecast and its conformal interval, given the information available at the time the prediction is made?”**
