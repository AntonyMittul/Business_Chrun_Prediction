# Business Churn Prediction — End-to-End Data Science Project

Predicting customer churn for a subscription business, built as a **full data-science
lifecycle**: data engineering → statistics → machine learning → MLOps → GenAI → deployment.

> **Status:** 🚧 Phase 0 complete (project setup). See [Roadmap](#roadmap).

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
- [ ] **Phase 1** — Data cleaning & validation (pandas + pandera schema contract)
- [ ] **Phase 2** — EDA + SQL analytics (DuckDB — business questions in pure SQL)
- [ ] **Phase 3** — Statistics & probability (hypothesis testing, effect sizes, survival analysis)
- [ ] **Phase 4** — Feature engineering + leakage audit (pandas → PySpark on Databricks)
- [ ] **Phase 5** — Modeling, MLflow & cost-sensitive threshold tuning
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
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

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
