import streamlit as st
import numpy as np
import joblib
model = joblib.load("rbf_svm_model.pkl")
scaler = joblib.load("scaler.pkl")
st.set_page_config(
    page_title="Rock vs Mine Prediction",
    page_icon="🔍",
    layout="centered"
)
st.title("Rock vs Mine Prediction System")
st.write(
    "Enter 60 sonar measurements to predict "
    "whether the detected object is a Rock or Mine."
)
st.info("This application uses a trained RBF Support Vector Machine.")
st.subheader("Sonar Measurements")
values = []
for i in range(60):
    value = st.number_input(
        f"Feature {i+1}",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.001,
        format="%.4f"
    )
    values.append(value)
if st.button("Predict Object"):
    input_data = np.array(values).reshape(1, -1)
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    if prediction[0] == 0:
        st.error("Prediction: MINE")
    else:
        st.success("Prediction: ROCK")
