# Business Churn Prediction — End-to-End Data Science Project

Predicting customer churn for a subscription business, built as a **full data-science
lifecycle**: data engineering → statistics → machine learning → MLOps → GenAI → deployment.

> **Status:** 🚧 Phase 5 complete — calibrated LightGBM in MLflow: **ROC-AUC 0.816,
> PR-AUC 0.351**, profit-optimal threshold **7.1× more retention profit** than the
> default 0.5 ($6,867 vs $966 on the test set, 82% churner recall). See [Roadmap](#roadmap).
>
> **Headline findings so far:** churn concentrates in the first 6 months (28% vs 10%
> baseline), after ≥2 payment failures (21–33%), at CSAT ≤ 2 (24–26%), and past 30 days
> of inactivity (~2×). Segment, contract, NPS and geography carry no signal —
> **$35.3k/month MRR at risk.**

---

## Problem

Retaining a customer is far cheaper than acquiring one. This project predicts which
customers are likely to **churn**, explains *why*, and turns each prediction into an
actionable retention recommendation.

## Dataset

[Customer Churn Prediction Business Dataset](https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset)
(Kaggle, synthetic but business-realistic).

- **10,000 customers × 32 columns**
- **Target:** `churn` (binary) — **10.2% churn rate** (imbalanced → evaluated on PR-AUC, not accuracy)
- **Feature families:** demographics · account/contract · product usage & engagement ·
  billing & payments · support interactions · marketing & satisfaction

## Tech stack

| Layer | Tools |
|---|---|
| Data engineering | PySpark, Parquet, pandera (data validation) |
| SQL analytics | DuckDB |
| Analysis & statistics | pandas, SciPy, statsmodels, lifelines (survival analysis) |
| Machine learning | scikit-learn, XGBoost, LightGBM, imbalanced-learn, Optuna |
| Explainability & GenAI | SHAP, Anthropic API (retention recommendations) |
| MLOps | MLflow, Airflow, Docker, GitHub Actions, Evidently |
| Serving | FastAPI, Streamlit, Cloud Run |

## Roadmap

- [x] **Phase 0** — Project setup & scaffolding
- [x] **Phase 1** — Data cleaning & validation (pandas + pandera schema contract)
- [x] **Phase 2** — EDA + SQL analytics (DuckDB — business questions in pure SQL)
- [x] **Phase 3** — Statistics & probability (hypothesis testing, effect sizes, survival analysis)
- [x] **Phase 4** — Feature engineering + leakage audit (pandas → PySpark on Databricks)
- [x] **Phase 5** — Modeling, MLflow & cost-sensitive threshold tuning
- [ ] **Phase 6** — Explainability (SHAP) + GenAI retention advisor
- [ ] **Phase 7** — Deployment, CI/CD & monitoring
- [ ] **Phase 8** — Polish, model card & write-up

## Project structure

```
Business_Chrun_Prediction/
├── config/            # config.yaml — paths, target, column groups
├── data/
│   ├── raw/           # source CSV (gitignored)
│   ├── interim/       # cleaned intermediate data
│   └── processed/     # model-ready features
├── notebooks/         # EDA & analysis notebooks
├── src/
│   ├── data/          # loading & validation
│   ├── features/      # feature engineering
│   ├── models/        # training & evaluation
│   └── utils/         # config & paths
├── tests/             # pytest smoke tests
├── models/            # trained artifacts (gitignored)
├── reports/figures/   # generated plots
└── .github/workflows/ # CI
```

## Setup

```bash
# 1. Create & activate a virtual environment
#    (keep it OUTSIDE OneDrive-synced folders — sync file-locking breaks pip installs)
python -m venv %USERPROFILE%\.venvs\churn
%USERPROFILE%\.venvs\churn\Scripts\activate          # Windows
# python -m venv .venv && source .venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) enable auto-formatting on commit
pre-commit install

# 4. Sanity-check the data
python -m src.data.load_data   # prints a dataset profile
pytest                         # runs smoke tests
```

## License

For educational / portfolio use.
