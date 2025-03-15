import streamlit as st
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

model = tf.keras.models.load_model("model.h5")

with open("onehot_geo.pkl", "rb") as file:
    onehot_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("label_gender.pkl", "rb") as file:
    label_gender = pickle.load(file)

st.title("Customer Exit Prediction")

geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0,1])
is_active_member = st.selectbox("Is Active Member", [0,1])

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

geo_encoded = onehot_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_geo.get_feature_names_out(["Geography"]))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
pred_prob = prediction[0][0]

st.write(pred_prob)

if pred_prob > 0.5:
    st.write("The Customer is likely to Churn!")
else:
    st.write("The Customer is NOT likely to Churn!")


