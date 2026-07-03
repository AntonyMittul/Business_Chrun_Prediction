"""Data-drift monitoring with Evidently.

Reference = training split; current = the data now being scored. In production
"current" would be the latest scoring batch — here the held-out test split stands
in, which also doubles as a sanity check (same distribution -> little drift
expected; heavy drift here would indicate a broken pipeline).

Run:  python -m src.monitoring.drift
Output: reports/drift_report.html + console summary.
"""
from __future__ import annotations

from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.model_selection import train_test_split

from src.models.train import SEED, load_features
from src.utils.config import REPORTS_DIR


def run_drift_report() -> str:
    X, y = load_features()
    X_ref, X_cur = train_test_split(X, test_size=0.2, stratify=y, random_state=SEED)

    # category dtype -> string for evidently's type inference
    for frame in (X_ref, X_cur):
        for c in frame.select_dtypes("category").columns:
            frame[c] = frame[c].astype(str)

    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=X_cur, reference_data=X_ref)

    out = REPORTS_DIR / "drift_report.html"
    snapshot.save_html(str(out))
    return str(out)


if __name__ == "__main__":
    path = run_drift_report()
    print(f"OK: drift report written -> {path}")
