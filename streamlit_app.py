import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model dan preprocessing
with open("best_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler_encoder.pkl", "rb") as preprocess_file:
    preprocess_obj = pickle.load(preprocess_file)

# Objek preprocessing
robust_scaler = preprocess_obj['robust_scaler']
ordinal_encoders = preprocess_obj['ordinal_encoders']
one_hot_columns = preprocess_obj['one_hot_columns']

# Fungsi prediksi
def predict_new_data(new_data_df):
    # Ordinal Encoding
    for col, encoder in ordinal_encoders.items():
        new_data_df[col] = encoder.transform(new_data_df[[col]])

    # One-hot encoding
    new_data_df = pd.get_dummies(new_data_df, columns=[col.split('_')[0] for col in one_hot_columns], dtype='int32')

    # Tambahkan kolom yang hilang
    for col in one_hot_columns:
        if col not in new_data_df.columns:
            new_data_df[col] = 0

    # Urutkan kolom
    new_data_df = new_data_df.reindex(sorted(new_data_df.columns), axis=1)
    one_hot_columns_sorted = sorted(one_hot_columns)
    feature_order = list(ordinal_encoders.keys()) + ['avg_price_per_room', 'lead_time'] + one_hot_columns_sorted
    new_data_df = new_data_df[feature_order]

    # Robust scaling
    new_data_df[['avg_price_per_room', 'lead_time']] = robust_scaler.transform(new_data_df[['avg_price_per_room', 'lead_time']])

    # Prediksi
    predictions = model.predict(new_data_df)
    probabilities = model.predict_proba(new_data_df)

    return predictions, probabilities

# UI Streamlit
st.title("Booking Cancellation Prediction")

# Input user
with st.form("prediction_form"):
    st.subheader("Isi data berikut:")
    no_of_adults = st.number_input("Jumlah Dewasa", min_value=1, value=2)
    no_of_children = st.number_input("Jumlah Anak-anak", min_value=0, value=0)
    no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=1)
    no_of_week_nights = st.number_input("Week Nights", min_value=0, value=2)
    type_of_meal_plan = st.selectbox("Meal Plan", ["Not Selected", "Meal Plan 1", "Meal Plan 2", "Meal Plan 3"])
    required_car_parking_space = st.selectbox("Perlu Parkir?", [0, 1])
    room_type_reserved = st.selectbox("Tipe Kamar", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5"])
    lead_time = st.number_input("Lead Time", min_value=0, value=45)
    arrival_year = st.selectbox("Tahun Kedatangan", [2017, 2018])
    arrival_month = st.selectbox("Bulan Kedatangan", list(range(1, 13)))
    arrival_date = st.selectbox("Tanggal Kedatangan", list(range(1, 32)))
    market_segment_type = st.selectbox("Market Segment", ["Online", "Offline", "Corporate", "Aviation", "Complementary"])
    repeated_guest = st.selectbox("Tamu Berulang?", [0, 1])
    no_of_previous_cancellations = st.number_input("Jumlah Pembatalan Sebelumnya", min_value=0, value=0)
    no_of_previous_bookings_not_canceled = st.number_input("Booking Sebelumnya yang Tidak Dibatalkan", min_value=0, value=0)
    avg_price_per_room = st.number_input("Rata-rata Harga Kamar", min_value=0.0, value=100.0)
    no_of_special_requests = st.number_input("Jumlah Permintaan Khusus", min_value=0, value=1)

    submit = st.form_submit_button("Prediksi")

# Saat tombol ditekan
if submit:
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

    prediction, probability = predict_new_data(new_data)
    status = "❌ Dibatalkan" if prediction[0] == 1 else "✅ Tidak Dibatalkan"
    confidence = np.max(probability[0]) * 100

    st.success(f"Hasil Prediksi: {status}")
    st.info(f"Akurasi keyakinan model: {confidence:.2f}%")
