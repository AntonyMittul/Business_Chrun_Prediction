# Model Card — Churn Prediction (Calibrated LightGBM)

## Model details
- **Model:** LightGBM binary classifier inside a sklearn Pipeline (one-hot encoding →
  LGBM), wrapped in `CalibratedClassifierCV` (isotonic, 5-fold).
- **Hyperparameters:** 40-trial Optuna TPE study (seed 42), objective = 5-fold CV PR-AUC.
  Notably shallow (8 leaves) — appropriate for weak individual signals.
- **Version/tracking:** MLflow experiment `churn-prediction` (`sqlite:///mlflow.db`);
  artifact `models/churn_model.joblib` + `models/model_meta.json`.
- **Reproducibility:** deterministic (seed 42); the Docker image retrains at build time
  from the committed raw data.

## Intended use
- **Primary:** monthly scoring of the active customer book to build a prioritized
  retention work-queue; each score ships with SHAP drivers and a suggested action.
- **Decision rule:** target customers with calibrated P(churn) ≥ **0.16** — the threshold
  that maximizes expected retention profit on out-of-fold training predictions under the
  documented offer economics (`config.yaml → business`).
- **Out of scope:** credit or pricing decisions; per-customer guarantees; any use where a
  false positive costs more than a discounted retention offer.

## Training data
- Kaggle "Customer Churn Prediction Business Dataset" — **synthetic**, 10,000 customers,
  10.21% churn, snapshot (no event timestamps).
- Cleaning: `total_revenue` dropped (exactly `fee × tenure`); `complaint_type` nulls
  encoded as "None" (informative missingness); schema enforced with pandera.
- Features: 4 statistically-validated threshold flags, compound-risk count, tenure
  buckets, intensity ratios. `city`/`country` dropped (proven independently random),
  `gender` dropped (not significant; fairness-positive).

## Metrics (held-out 20% test, stratified)
| Metric | Value |
|---|---|
| ROC-AUC | 0.816 |
| PR-AUC | 0.351 (baseline 0.102) |
| Brier score | 0.077 (isotonic-calibrated) |
| Recall @ threshold 0.16 | 0.82 |
| Expected profit @ threshold | $6,867 (vs $966 at 0.5) |

Leakage audit: shuffled-target CV AUC ≈ 0.49; max single-feature importance share 8%.

## Explainability
SHAP TreeExplainer on the uncalibrated pipeline (calibration is monotonic → identical
rankings/attributions). Dominant driver: engineered `risk_factor_count`
(mean |SHAP| 0.82). Per-customer top-5 drivers feed the GenAI retention advisor.

## Limitations & risks
1. **Synthetic data** — relationships are cleaner than production data; absolute metric
   values will not transfer to a real book.
2. **Snapshot, not longitudinal** — no true temporal validation is possible; deployment
   on real data must use time-based splits.
3. **Profit numbers depend on assumed economics** (offer $21, save value $420, save rate
   30%) — stated in config, must be re-estimated per business.
4. **Weak individual signals** (concordance 0.65 in survival model) — the model ranks
   risk well but individual probabilities near the threshold are uncertain.
5. **GenAI advisor** can phrase drivers imperfectly; it is constrained to model-provided
   facts and always labeled with its `source`. Human review recommended before customer
   contact.

## Monitoring & retraining
- Evidently data-drift report (`python -m src.monitoring.drift`) — run per scoring batch.
- Prefect flow `churn-retrain` re-runs clean → features → train → drift end-to-end.
- Retrain trigger: drift on any of the 4 core drivers, or profit-at-threshold decay.
