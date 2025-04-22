import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Judul aplikasi
st.title("Hotel Booking Cancellation Predictor")

# Load model dan preprocessing
with open("best_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler_encoder.pkl", "rb") as preprocess_file:
    preprocess_obj = pickle.load(preprocess_file)

robust_scaler = preprocess_obj['robust_scaler']
ordinal_encoders = preprocess_obj['ordinal_encoders']
one_hot_columns = preprocess_obj['one_hot_columns']

# Fungsi prediksi
def predict_new_data(new_data_df):
    for col, encoder in ordinal_encoders.items():
        new_data_df[col] = encoder.transform(new_data_df[[col]])
    
    new_data_df = pd.get_dummies(new_data_df, columns=[col.split('_')[0] for col in one_hot_columns], dtype='int32')
    
    for col in one_hot_columns:
        if col not in new_data_df.columns:
            new_data_df[col] = 0

    new_data_df = new_data_df.reindex(sorted(new_data_df.columns), axis=1)
    one_hot_columns_sorted = sorted(one_hot_columns)
    feature_order = list(ordinal_encoders.keys()) + ['avg_price_per_room', 'lead_time'] + one_hot_columns_sorted
    new_data_df = new_data_df[feature_order]

    new_data_df[['avg_price_per_room', 'lead_time']] = robust_scaler.transform(new_data_df[['avg_price_per_room', 'lead_time']])

    predictions = model.predict(new_data_df)
    probabilities = model.predict_proba(new_data_df)

    return predictions, probabilities

# Input manual atau test case
option = st.radio("Pilih input data:", ["Manual Input", "Test Case 1", "Test Case 2"])

if option == "Manual Input":
    user_input = {
        "no_of_adults": st.number_input("Number of Adults", min_value=1, value=2),
        "no_of_children": st.number_input("Number of Children", min_value=0, value=0),
        "no_of_weekend_nights": st.number_input("Weekend Nights", min_value=0, value=1),
        "no_of_week_nights": st.number_input("Week Nights", min_value=0, value=2),
        "type_of_meal_plan": st.selectbox("Meal Plan", ["Not Selected", "Meal Plan 1", "Meal Plan 2", "Meal Plan 3"]),
        "required_car_parking_space": st.selectbox("Car Parking Required", [0, 1]),
        "room_type_reserved": st.selectbox("Room Type Reserved", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"]),
        "lead_time": st.number_input("Lead Time (days)", min_value=0, value=45),
        "arrival_year": st.selectbox("Arrival Year", [2017, 2018]),
        "arrival_month": st.selectbox("Arrival Month", list(range(1, 13))),
        "arrival_date": st.selectbox("Arrival Date", list(range(1, 32))),
        "market_segment_type": st.selectbox("Market Segment", ["Online", "Offline", "Corporate", "Complementary", "Aviation", "Other"]),
        "repeated_guest": st.selectbox("Repeated Guest", [0, 1]),
        "no_of_previous_cancellations": st.number_input("Previous Cancellations", min_value=0, value=0),
        "no_of_previous_bookings_not_canceled": st.number_input("Previous Non-Canceled Bookings", min_value=0, value=0),
        "avg_price_per_room": st.number_input("Average Price per Room", min_value=0.0, value=100.0),
        "no_of_special_requests": st.number_input("Special Requests", min_value=0, value=0)
    }
    input_df = pd.DataFrame([user_input])

elif option == "Test Case 1":
    st.markdown("**Test Case 1:** Guest with basic reservation, likely not canceled.")
    input_df = pd.DataFrame([{
        "no_of_adults": 2,
        "no_of_children": 0,
        "no_of_weekend_nights": 1,
        "no_of_week_nights": 2,
        "type_of_meal_plan": "Meal Plan 1",
        "required_car_parking_space": 0,
        "room_type_reserved": "Room_Type 1",
        "lead_time": 45,
        "arrival_year": 2018,
        "arrival_month": 5,
        "arrival_date": 15,
        "market_segment_type": "Online",
        "repeated_guest": 0,
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 0,
        "avg_price_per_room": 100.0,
        "no_of_special_requests": 1
    }])

elif option == "Test Case 2":
    st.markdown("**Test Case 2:** Guest with long lead time and no car parking, might cancel.")
    input_df = pd.DataFrame([{
        "no_of_adults": 1,
        "no_of_children": 1,
        "no_of_weekend_nights": 0,
        "no_of_week_nights": 1,
        "type_of_meal_plan": "Not Selected",
        "required_car_parking_space": 0,
        "room_type_reserved": "Room_Type 6",
        "lead_time": 120,
        "arrival_year": 2018,
        "arrival_month": 11,
        "arrival_date": 20,
        "market_segment_type": "Corporate",
        "repeated_guest": 0,
        "no_of_previous_cancellations": 1,
        "no_of_previous_bookings_not_canceled": 0,
        "avg_price_per_room": 150.0,
        "no_of_special_requests": 0
    }])

# Tombol prediksi
if st.button("Predict"):
    prediction, probability = predict_new_data(input_df)
    st.write("### Hasil Prediksi")
    st.write(f"**Booking Status:** {'Canceled' if prediction[0] == 1 else 'Not Canceled'}")
    st.write(f"**Probabilities:** {probability[0]}")
