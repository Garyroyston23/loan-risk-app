import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

# Where the API lives. On your laptop it's port 8001; later (step 6) we'll
# point this at the online version instead.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

HOME = {"Rent": "RENT", "Own": "OWN", "Mortgage": "MORTGAGE", "Other": "OTHER"}
PURPOSE = {
    "Education": "EDUCATION",
    "Medical": "MEDICAL",
    "Business venture": "VENTURE",
    "Personal": "PERSONAL",
    "Debt consolidation": "DEBTCONSOLIDATION",
    "Home improvement": "HOMEIMPROVEMENT",
}


def nice(value):
    """Show 120000.0 as 120,000 and leave text like 'RENT' alone."""
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    if isinstance(value, (int, float)):
        return f"{value:,}"
    return value


st.set_page_config(page_title="Loan Risk Checker", page_icon="🏦")
st.title("Loan Risk Checker")
st.write(
    "Enter an applicant's details to estimate their risk of defaulting "
    "on a loan, and see the reasons behind the decision."
)

# ---------- The form ----------
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual income", min_value=1000, max_value=2_000_000, value=50_000, step=1000)
    emp_length = st.number_input("Years employed", min_value=0, max_value=60, value=3)
    history = st.number_input("Credit history (years)", min_value=0, max_value=60, value=5)
with col2:
    amount = st.number_input("Loan amount", min_value=500, max_value=100_000, value=10_000, step=500)
    home = st.selectbox("Home ownership", list(HOME))
    purpose = st.selectbox("Loan purpose", list(PURPOSE))
    defaulted = st.radio("Defaulted on a loan before?", ["No", "Yes"], horizontal=True)

# ---------- When the button is clicked ----------
if st.button("Check risk", type="primary"):
    application = {
        "person_age": age,
        "person_income": income,
        "person_emp_length": emp_length,
        "loan_amnt": amount,
        "cb_person_cred_hist_length": history,
        "person_home_ownership": HOME[home],
        "loan_intent": PURPOSE[purpose],
        "cb_person_default_on_file": "Y" if defaulted == "Yes" else "N",
    }

    # Send the application to our API, just like the /docs page did
    try:
        response = requests.post(f"{API_URL}/predict", json=application, timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException:
        st.error(
            "Couldn't reach the API. Is it running? Start it in Terminal with: "
            "`uvicorn api:app --reload --port 8001`"
        )
        st.stop()

    # ---------- Show the result ----------
    prob = result["default_probability"]
    if result["risk"] == "High risk":
        st.error(f"### High risk: {prob:.1%} chance of default")
    else:
        st.success(f"### Low risk: {prob:.1%} chance of default")
    st.progress(prob)

    # ---------- Show the reasons ----------
    st.subheader("Why?")
    reasons = pd.DataFrame(result["top_reasons"])
    reasons["value"] = reasons["value"].apply(nice).astype(str)
    reasons["label"] = reasons["factor"] + ": " + reasons["value"]
    # Positive = pushes risk up, negative = pushes risk down
    reasons["push"] = reasons.apply(
        lambda r: r["strength"] if r["effect"] == "increases risk" else -r["strength"], axis=1
    )

    chart = (
        alt.Chart(reasons)
        .mark_bar()
        .encode(
            x=alt.X("push:Q", title="← lowers risk     |     raises risk →"),
                        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=300)),
            color=alt.condition(alt.datum.push > 0, alt.value("#d9534f"), alt.value("#2e8b57")),
            tooltip=["factor", "effect", "strength"],
        )
    )
    st.altair_chart(chart)

    for _, r in reasons.iterrows():
        icon = "🔴" if r["push"] > 0 else "🟢"
        st.write(f"{icon} **{r['label']}** {r['effect']}")

    st.caption("Reasons are calculated with SHAP. Longer bars had a bigger effect on the decision.")