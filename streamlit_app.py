import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model dan preprocessing
with open("best_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler_encoder.pkl", "rb") as preprocess_file:
    preprocess_obj = pickle.load(preprocess_file)

# Preprocessing object
robust_scaler = preprocess_obj['robust_scaler']
ordinal_encoders = preprocess_obj['ordinal_encoders']
one_hot_columns = preprocess_obj['one_hot_columns']

def predict_new_data(new_data_df):
    # 1. Ordinal Encoding
    for col, encoder in ordinal_encoders.items():
        new_data_df[col] = encoder.transform(new_data_df[[col]])

    # 2. One-hot encoding
    new_data_df = pd.get_dummies(new_data_df, columns=[col.split('_')[0] for col in one_hot_columns], dtype='int32')

    # Tambahkan missing one-hot encoded columns agar sesuai dengan training
    for col in one_hot_columns:
        if col not in new_data_df.columns:
            new_data_df[col] = 0

    # Urutkan kolom
    new_data_df = new_data_df.reindex(sorted(new_data_df.columns), axis=1)
    one_hot_columns_sorted = sorted(one_hot_columns)
    feature_order = list(ordinal_encoders.keys()) + ['avg_price_per_room', 'lead_time'] + one_hot_columns_sorted
    new_data_df = new_data_df[feature_order]

    # 3. Scaling
    new_data_df[['avg_price_per_room', 'lead_time']] = robust_scaler.transform(new_data_df[['avg_price_per_room', 'lead_time']])

    # 4. Predict
    predictions = model.predict(new_data_df)
    probabilities = model.predict_proba(new_data_df)

    return predictions, probabilities

# ---------------------- Streamlit UI ----------------------
st.title("Hotel Booking Cancellation Prediction")
st.write("Masukkan data reservasi untuk memprediksi apakah akan dibatalkan atau tidak.")

# Form input data
with st.form("booking_form"):
    no_of_adults = st.number_input("Jumlah Dewasa", min_value=1, max_value=10, value=2)
    no_of_children = st.number_input("Jumlah Anak", min_value=0, max_value=10, value=0)
    no_of_weekend_nights = st.number_input("Malam Akhir Pekan", min_value=0, value=1)
    no_of_week_nights = st.number_input("Malam Hari Kerja", min_value=0, value=2)
    type_of_meal_plan = st.selectbox("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
    required_car_parking_space = st.selectbox("Perlu Parkir?", [0, 1])
    room_type_reserved = st.selectbox("Tipe Kamar", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
    lead_time = st.slider("Lead Time (hari)", min_value=0, max_value=400, value=45)
    arrival_year = st.selectbox("Tahun Kedatangan", [2017, 2018])
    arrival_month = st.selectbox("Bulan Kedatangan", list(range(1, 13)))
    arrival_date = st.selectbox("Tanggal Kedatangan", list(range(1, 32)))
    market_segment_type = st.selectbox("Segmentasi Pasar", ["Online", "Offline", "Corporate", "Complementary", "Aviation", "Other"])
    repeated_guest = st.selectbox("Tamu Berulang?", [0, 1])
    no_of_previous_cancellations = st.number_input("Jumlah Pembatalan Sebelumnya", min_value=0, value=0)
    no_of_previous_bookings_not_canceled = st.number_input("Booking Sebelumnya yang Tidak Dibatalkan", min_value=0, value=0)
    avg_price_per_room = st.number_input("Harga per Kamar", min_value=0.0, value=100.0)
    no_of_special_requests = st.number_input("Permintaan Khusus", min_value=0, max_value=5, value=1)

    submitted = st.form_submit_button("Prediksi")

    if submitted:
        new_data = pd.DataFrame([{
            "no_of_adults": no_of_adults,
            "no_of_children": no_of_children,
            "no_of_weekend_nights": no_of_weekend_nights,
            "no_of_week_nights": no_of_week_nights,
            "type_of_meal_plan": type_of_meal_plan,
            "required_car_parking_space": required_car_parking_space,
            "room_type_reserved": room_type_reserved,
            "lead_time": lead_time,
            "arrival_year": arrival_year,
            "arrival_month": arrival_month,
            "arrival_date": arrival_date,
            "market_segment_type": market_segment_type,
            "repeated_guest": repeated_guest,
            "no_of_previous_cancellations": no_of_previous_cancellations,
            "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
            "avg_price_per_room": avg_price_per_room,
            "no_of_special_requests": no_of_special_requests
        }])

        pred, prob = predict_new_data(new_data)

        st.success(f"**Prediksi:** {'Canceled' if pred[0] == 1 else 'Not Canceled'}")
        st.info(f"Probabilitas Pembatalan: {prob[0][1]:.2%}")
        st.info(f"Probabilitas Tidak Dibatalkan: {prob[0][0]:.2%}")

# ---------------------- Contoh Test Case ----------------------
st.sidebar.header("Contoh Test Case")

if st.sidebar.button("Test Case 1"):
    st.sidebar.write("**Test Case 1 (Tidak Dibatalkan):**")
    st.session_state.update({
        "Jumlah Dewasa": 2,
        "Jumlah Anak": 0,
        "Malam Akhir Pekan": 1,
        "Malam Hari Kerja": 2,
        "Meal Plan": "Meal Plan 1",
        "Perlu Parkir?": 0,
        "Tipe Kamar": "Room_Type 1",
        "Lead Time (hari)": 45,
        "Tahun Kedatangan": 2018,
        "Bulan Kedatangan": 5,
        "Tanggal Kedatangan": 15,
        "Segmentasi Pasar": "Online",
        "Tamu Berulang?": 0,
        "Jumlah Pembatalan Sebelumnya": 0,
        "Booking Sebelumnya yang Tidak Dibatalkan": 0,
        "Harga per Kamar": 100.0,
        "Permintaan Khusus": 1
    })

if st.sidebar.button("Test Case 2"):
    st.sidebar.write("**Test Case 2 (Kemungkinan Dibatalkan):**")
    st.session_state.update({
        "Jumlah Dewasa": 1,
        "Jumlah Anak": 2,
        "Malam Akhir Pekan": 2,
        "Malam Hari Kerja": 5,
        "Meal Plan": "Meal Plan 3",
        "Perlu Parkir?": 1,
        "Tipe Kamar": "Room_Type 3",
        "Lead Time (hari)": 200,
        "Tahun Kedatangan": 2017,
        "Bulan Kedatangan": 12,
        "Tanggal Kedatangan": 20,
        "Segmentasi Pasar": "Corporate",
        "Tamu Berulang?": 0,
        "Jumlah Pembatalan Sebelumnya": 1,
        "Booking Sebelumnya yang Tidak Dibatalkan": 0,
        "Harga per Kamar": 180.0,
        "Permintaan Khusus": 3
    })
