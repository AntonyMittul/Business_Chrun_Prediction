# Resume Bullets & Interview Talking Points

## Resume bullets (pick 3–4, tailor per JD)

- Built an end-to-end churn prediction system (10k-customer SaaS dataset): DuckDB SQL
  analytics, statistically-validated feature engineering, calibrated LightGBM
  (ROC-AUC 0.82), and a profit-optimized decision threshold that made the simulated
  retention campaign **7.1× more profitable** than the default cutoff.
- Applied rigorous statistical inference — chi-square/Mann-Whitney with
  Benjamini-Hochberg correction, effect sizes, bootstrap CIs, and Kaplan-Meier/Cox
  survival analysis — to isolate 4 true churn drivers from 23 candidate features and
  design a powered retention A/B test (5pp MDE, 1,194/arm).
- Shipped the model as a production service: FastAPI endpoint returning score → SHAP
  drivers → Gemini-generated retention plan, Streamlit dashboard, self-training Docker
  image, GitHub Actions CI with container smoke tests, Evidently drift monitoring, and a
  Prefect retrain pipeline (experiments tracked in MLflow).
- Engineered a compound-risk feature from statistical findings that became the model's
  dominant SHAP driver (mean |SHAP| 4× the runner-up), and demonstrated experimentally
  that SMOTE degraded PR-AUC vs handling class imbalance at the decision threshold.

## Interview stories (STAR-ready)

**"Tell me about a data quality issue you found."**
`total_revenue` equaled `monthly_fee × tenure_months` for all 10,000 rows (max diff 0) —
a derived column masquerading as a feature. Also proved `city`×`country` were
independently random (all 49 pairs uniform: "Bangladesh/London"). Dropped both with
evidence, not intuition.

**"How do you handle class imbalance?"**
I don't reach for SMOTE — I tested it. Plain LightGBM beat SMOTE (CV PR-AUC 0.288 vs
0.268) and class weights. The imbalance really lives in the *cost asymmetry* ($84 gain
per caught churner vs $21 per false alarm), so I calibrated probabilities (isotonic) and
moved the decision threshold to the profit maximum — chosen on out-of-fold train
predictions to avoid test leakage. Result: 7.1× profit vs the 0.5 default.

**"A time a metric misled you?"**
`last_login_days_ago` failed the global Mann-Whitney test (p=0.12) yet churn doubled
past 30 days — a threshold-shaped effect a whole-distribution test dilutes. The Cox
model recovered it (p=0.003) once conditioned on other covariates. Lesson: encode the
threshold as a feature; don't trust one test shape.
Also: LightGBM split-count importances buried my binary flags, while SHAP showed the
compound flag was the dominant driver — attribution method changes the story.

**"How did you use statistics beyond modeling?"**
BH-corrected hypothesis testing (only 4/23 features survived), bootstrap CI for the
onboarding gap (20pp, CI 17–23), Bayes framing of compound risk (low CSAT + payment
failures → 37% churn), survival analysis (CSAT +1 → hazard ×0.64), and a powered A/B
design for the retention program with guardrail metrics.

**"What would break in production?"**
The data is synthetic and a snapshot — real deployment needs time-based splits, drift
monitoring per scoring batch (Evidently is wired), re-estimated offer economics, and
human review of GenAI plans. I documented all of this in the model card rather than
hiding it.

## Honest-answer guardrails (things NOT to overclaim)
- Spark is a portability demonstration (PySpark twin of the pandas pipeline, runnable on
  Databricks Free Edition) — 10k rows do not need a cluster. Say "cluster-portable", not
  "big data at scale".
- Metrics come from synthetic data; quote them as evidence of method, not of business
  impact at a real company.
- The A/B test is designed, not run.
