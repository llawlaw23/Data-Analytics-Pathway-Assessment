import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

month_to_nv = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

def preprocess_input(data):
    binary_cols = ['default', 'housing', 'loan']
    for col in binary_cols:
        data[col] = data[col].map({'yes': 1, 'no': 0})
    
    data['month'] = data['month'].map(month_to_nv)

    data = pd.get_dummies(data, drop_first=True)

    for col in model.feature_names_in_:
        if col not in data.columns:
            data[col] = 0

    return data[model.feature_names_in_]

st.title("Bank Term Deposit Subscription Predictor")
st.write("Fill in the customer information:")

with st.form("input_form"):
    age = st.slider("Age", 15, 90, 35)
    default = st.selectbox("Has credit in default?", ["no", "yes"])
    housing = st.selectbox("Has housing loan?", ["no", "yes"])
    loan = st.selectbox("Has personal loan?", ["no", "yes"])
    month = st.selectbox("Last contact month", list(month_to_nv.keys()))
    duration = st.number_input("Call duration (seconds)", min_value=0, value=100)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "age": age,
        "default": default,
        "housing": housing,
        "loan": loan,
        "month": month,
        "duration": duration
    }])

    processed = preprocess_input(input_df)
    prediction = model.predict(processed)[0]

    if prediction == 1:
        st.success("✅ The client is likely to subscribe to a term deposit.")
    else:
        st.warning("❌ The client is unlikely to subscribe.")
