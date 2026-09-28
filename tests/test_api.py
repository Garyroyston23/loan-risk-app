"""
Automated tests for the Loan Risk API.

Run them from the project folder with:
    python -m pytest
"""
from fastapi.testclient import TestClient

from api import app

# A fake "browser" that sends requests straight to the API, no server needed
client = TestClient(app)

HIGH_RISK = {
    "person_age": 25,
    "person_income": 30000,
    "person_emp_length": 2,
    "loan_amnt": 15000,
    "cb_person_cred_hist_length": 3,
    "person_home_ownership": "RENT",
    "loan_intent": "MEDICAL",
    "cb_person_default_on_file": "Y",
}

LOW_RISK = {
    "person_age": 45,
    "person_income": 120000,
    "person_emp_length": 15,
    "loan_amnt": 5000,
    "cb_person_cred_hist_length": 15,
    "person_home_ownership": "MORTGAGE",
    "loan_intent": "HOMEIMPROVEMENT",
    "cb_person_default_on_file": "N",
}


def test_home_page_works():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Loan Risk API is running"


def test_high_risk_applicant_is_flagged():
    result = client.post("/predict", json=HIGH_RISK).json()
    assert result["risk"] == "High risk"
    assert result["default_probability"] > 0.5


def test_low_risk_applicant_is_approved():
    result = client.post("/predict", json=LOW_RISK).json()
    assert result["risk"] == "Low risk"
    assert result["default_probability"] < 0.5


def test_probability_is_between_0_and_1():
    for applicant in (HIGH_RISK, LOW_RISK):
        prob = client.post("/predict", json=applicant).json()["default_probability"]
        assert 0 <= prob <= 1


def test_three_readable_reasons_are_returned():
    reasons = client.post("/predict", json=HIGH_RISK).json()["top_reasons"]
    assert len(reasons) == 3
    for reason in reasons:
        assert set(reason) == {"factor", "value", "effect", "strength"}
        assert reason["effect"] in ("increases risk", "decreases risk")
        assert "__" not in reason["factor"]  # no raw column names like "cat__..."


def test_missing_field_is_rejected():
    incomplete = dict(HIGH_RISK)
    del incomplete["person_income"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422  # 422 = "invalid input"


def test_wrong_type_is_rejected():
    bad = dict(HIGH_RISK, person_age="twenty-five")
    response = client.post("/predict", json=bad)
    assert response.status_code == 422