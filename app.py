import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ================= LOAD MODEL =====================
model = joblib.load("model.pkl")

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Enter the car details below to estimate the selling price.")

# ================ INPUT FIELDS ====================
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

km_driven = st.number_input("Kilometers Driven", min_value=0)
mileage = st.number_input("Mileage (km/ltr/kg)", min_value=1.0)
engine = st.number_input("Engine CC", min_value=500)
max_power = st.number_input("Max Power (bhp)", min_value=20)
seats = st.selectbox("Number of Seats", [4, 5, 6, 7, 8])

car_age = st.number_input("Car Age (years)", min_value=0)

# ================ PREDICTION ======================
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'km_driven': [km_driven],
        'mileage(km/ltr/kg)': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats],
        'car_age': [car_age]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Selling Price: ₹ {int(prediction):,}")

