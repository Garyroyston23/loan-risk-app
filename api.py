import warnings

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel

# Hide a harmless SHAP message about its output format
warnings.filterwarnings("ignore", message="LightGBM binary classifier")

# Load the trained model once, when the API starts
model = joblib.load("model/model.joblib")
prep = model.named_steps["prep"]    # the cleaning steps
clf = model.named_steps["model"]    # the LightGBM model itself
explainer = shap.TreeExplainer(clf)  # SHAP explainer for tree models

app = FastAPI(title="Loan Risk API")

# Friendly names to show users instead of column names
NICE_NAMES = {
    "person_age": "Age",
    "person_income": "Income",
    "person_emp_length": "Years employed",
    "loan_amnt": "Loan amount",
    "loan_percent_income": "Loan as % of income",
    "cb_person_cred_hist_length": "Credit history length",
    "person_home_ownership": "Home ownership",
    "loan_intent": "Loan purpose",
    "cb_person_default_on_file": "Previous default",
}


# What a loan application must contain
class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_emp_length: float
    loan_amnt: float
    cb_person_cred_hist_length: int
    person_home_ownership: str
    loan_intent: str
    cb_person_default_on_file: str


def explain(data):
    """Return the top 3 reasons behind one prediction."""
    X = prep.transform(data)  # clean the data the same way as in training
    values = explainer.shap_values(X)
    if isinstance(values, list):  # some SHAP versions return one array per class
        values = values[1]
    row = values[0]

    # One-hot encoding split some columns into several (e.g. RENT, OWN...).
    # Add their effects back together so each original feature has one score.
    totals = {}
    for column, effect in zip(X.columns, row):
        name = column.split("__", 1)[1]  # remove "num__" / "cat__" prefix
        for original in NICE_NAMES:
            if name.startswith(original):
                name = original
                break
        totals[name] = totals.get(name, 0) + effect

    top3 = sorted(totals.items(), key=lambda item: abs(item[1]), reverse=True)[:3]

    reasons = []
    for feature, effect in top3:
        value = data.iloc[0][feature]
        if feature == "loan_percent_income":
            value = f"{value:.0%}"
        elif hasattr(value, "item"):
            value = value.item()  # convert numpy numbers to normal numbers
        reasons.append({
            "factor": NICE_NAMES[feature],
            "value": value,
            "effect": "increases risk" if effect > 0 else "decreases risk",
            "strength": round(abs(float(effect)), 3),
        })
    return reasons


@app.get("/")
def home():
    return {"message": "Loan Risk API is running"}


@app.post("/predict")
def predict(application: LoanApplication):
    data = pd.DataFrame([application.model_dump()])
    # Calculate this ourselves so the user doesn't have to
    data["loan_percent_income"] = data["loan_amnt"] / data["person_income"]

    prob = model.predict_proba(data)[0, 1]
    return {
        "default_probability": round(float(prob), 3),
        "risk": "High risk" if prob >= 0.5 else "Low risk",
        "top_reasons": explain(data),
    }