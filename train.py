"""
Step 1: Train and save a loan risk model.

Run from the project folder:
    python train.py

Output:
    model/model.joblib   -> the trained pipeline (cleaning + model in one object)
    model/metrics.json   -> scores for each model, so you can compare them
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/credit_risk_dataset.csv")
MODEL_DIR = Path("model")
TARGET = "loan_status"  # 1 = defaulted, 0 = repaid

# Only features a real applicant would know when filling in a form.
# We deliberately DROP loan_grade and loan_int_rate: those are set by the
# lender's own risk assessment, so using them would be a kind of "cheating"
# (the model would copy the bank's answer instead of learning risk itself).
NUMERIC = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows_before = len(df)

    df = df.drop_duplicates()
    # Remove impossible values (the raw data has ages like 144)
    df = df[(df["person_age"] >= 18) & (df["person_age"] <= 100)]
    df = df[(df["person_emp_length"].isna()) | (df["person_emp_length"] <= 60)]

    print(f"Rows: {rows_before} -> {len(df)} after cleaning")
    print(f"Default rate: {df[TARGET].mean():.1%}")
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),  # fill missing numbers
        ("scale", StandardScaler()),                   # put numbers on one scale
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_steps, NUMERIC),
        ("cat", categorical_steps, CATEGORICAL),
    ])


def main():
    df = load_and_clean(DATA_PATH)
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]

    # stratify keeps the same default rate in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "lightgbm": LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=42, verbose=-1,
        ),
    }

    results = {}
    fitted = {}
    for name, model in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        pipe.set_output(transform="pandas")         
        pipe.fit(X_train, y_train)

        probs = pipe.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y_test, probs)
        report = classification_report(y_test, preds, output_dict=True)

        results[name] = {
            "roc_auc": round(auc, 4),
            "recall_defaults": round(report["1"]["recall"], 4),
            "precision_defaults": round(report["1"]["precision"], 4),
        }
        fitted[name] = pipe
        print(f"\n{name}: ROC-AUC = {auc:.4f}")
        print(classification_report(y_test, preds, digits=3))

    best = max(results, key=lambda n: results[n]["roc_auc"])
    print(f"Best model: {best}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[best], MODEL_DIR / "model.joblib")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(
            {"best_model": best, "features": NUMERIC + CATEGORICAL, "results": results},
            f, indent=2,
        )
    print(f"Saved to {MODEL_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
