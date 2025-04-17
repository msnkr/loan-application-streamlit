import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("./loan_approval_dataset.csv")
df.columns = df.columns.str.strip()


# Header
st.header("""
          Loan Prediction Dataset
          """)


# Sidebar
st.sidebar.header("""
                  User Input Parameters
                  """)

# Preprocess the data
df.drop("loan_id", axis=1, inplace=True)
df["education"] = df["education"].map({" Graduate": 1, " Not Graduate": 0})
df["self_employed"] = df["self_employed"].map({" Yes": 1, " No": 0})
df["loan_status"] = df["loan_status"].map({" Approved": 1, " Rejected": 0})


def get_user_input():
    dependants = st.sidebar.slider(
        "Number of dependants: ", 0, 5)
    education = st.sidebar.radio("Education", ("0", "1"))
    employed = st.sidebar.radio("Self-employed", ("0", "1"))
    income_annum = st.sidebar.slider(
        "Income per annum",  200000, 9900000, 5100000)
    loan_amount = st.sidebar.slider(
        "Loan Amount", 300000, 39500000, 14500000)
    loan_term = st.sidebar.slider("Loan Term", 2, 20, 10)
    cibil_score = st.sidebar.slider("Cibil Score", 300, 900, 600)
    res_value = st.sidebar.slider(
        "Residential Asset Value", -100000, 29100000,  5600000)
    com_asset_value = st.sidebar.slider(
        "Commercial Asset Value", 0, 19400000, 3700000)
    lux_asset_value = st.sidebar.slider(
        "Luxury Asset Value", 300000, 39200000, 14600000)
    bank_asset_value = st.sidebar.slider(
        "Bank Asset Value", 0, 14700000, 4600000)

    data = {
        "no_of_dependents": dependants,
        "education": education,
        "self_employed": employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": res_value,
        "commercial_assets_value": com_asset_value,
        "luxury_assets_value": lux_asset_value,
        "bank_asset_value": bank_asset_value
    }

    features = pd.DataFrame(data, index=[0])
    return features


data_df = get_user_input()
st.write(data_df)

clf = RandomForestClassifier()
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

clf.fit(X, y)

prediction = clf.predict(data_df)
predict_proba = clf.predict_proba(data_df)

st.subheader("""
             Prediction
             """)
st.write(prediction)

st.subheader("""
             Prediction Probability
             """)

st.write(predict_proba)
