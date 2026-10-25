
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Credit Card Fraud Detection")

@st.cache_resource
def load_files():
    with open("credit_card_fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("credit_card_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, list(features)

st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to check for possible fraud.")

try:
    model, scaler, features = load_files()

    with st.form("fraud_form"):
        values = {}
        for feature in features:
            values[feature] = st.number_input(
                str(feature), value=0.0, format="%.6f"
            )
        submitted = st.form_submit_button("Check Transaction")

    if submitted:
        data = pd.DataFrame([[values[f] for f in features]],
                            columns=features)
        scaled_data = scaler.transform(data)
        prediction = int(model.predict(scaled_data)[0])

        if prediction == 1:
            st.error("⚠️ Potential Fraudulent Transaction")
        else:
            st.success("✅ Potential Legitimate Transaction")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_data)[0]
            st.write("Prediction probabilities:")
            st.json({
                str(model.classes_[i]): round(float(probabilities[i]), 4)
                for i in range(len(probabilities))
            })

except Exception as e:
    st.error(f"Could not load the model or make a prediction: {e}")
