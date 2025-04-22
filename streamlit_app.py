import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ——— Load model & preprocessing objects ———
@st.cache(allow_output_mutation=True)
def load_objects():
    with open("best_model.pkl", "rb") as mf:
        model = pickle.load(mf)
    with open("scaler_encoder.pkl", "rb") as pf:
        preprocess = pickle.load(pf)
    return model, preprocess['robust_scaler'], preprocess['ordinal_encoders'], preprocess['one_hot_columns']

model, robust_scaler, ordinal_encoders, one_hot_columns = load_objects()

# ——— Sample test cases ———
test_cases = {
    "Case 1 - Business Booking": {
        "no_of_adults": 2,
        "no_of_children": 1,
        "no_of_weekend_nights": 1,
        "no_of_week_nights": 2,
        "type_of_meal_plan": "Meal Plan 2",
        "required_car_parking_space": 1,
        "room_type_reserved": "Room_Type 3",
        "lead_time": 60,
        "arrival_year": 2024,
        "arrival_month": 7,
        "arrival_date": 20,
        "market_segment_type": "Corporate",
        "repeated_guest": 1,
        "no_of_previous_cancellations": 1,
        "no_of_previous_bookings_not_canceled": 2,
        "avg_price_per_room": 150.0,
        "no_of_special_requests": 2
    },
    "Case 2 - Leisure Last-Minute": {
        "no_of_adults": 1,
        "no_of_children": 0,
        "no_of_weekend_nights": 0,
        "no_of_week_nights": 1,
        "type_of_meal_plan": "Not Selected",
        "required_car_parking_space": 0,
        "room_type_reserved": "Room_Type 1",
        "lead_time": 10,
        "arrival_year": 2023,
        "arrival_month": 12,
        "arrival_date": 15,
        "market_segment_type": "Online",
        "repeated_guest": 0,
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 0,
        "avg_price_per_room": 80.0,
        "no_of_special_requests": 0
    }
}

st.title("Hotel Booking Cancellation Prediction")
st.write("Masukkan detail pemesanan, atau pilih salah satu test case untuk auto-fill.")

# ——— Sidebar for selecting test case or custom ———
option = st.sidebar.selectbox("Pilih Test Case atau Custom Input:", ["Custom"] + list(test_cases.keys()))

# Prepare input dict
if option == "Custom":
    user_input = {}
    # Numeric inputs
    user_input['no_of_adults'] = st.sidebar.number_input('Jumlah Dewasa', min_value=0, value=2)
    user_input['no_of_children'] = st.sidebar.number_input('Jumlah Anak', min_value=0, value=0)
    user_input['no_of_weekend_nights'] = st.sidebar.number_input('Malam Akhir Pekan', min_value=0, value=1)
    user_input['no_of_week_nights'] = st.sidebar.number_input('Malam Hari Kerja', min_value=0, value=2)
    user_input['lead_time'] = st.sidebar.number_input('Lead Time (hari)', min_value=0, value=30)
    user_input['arrival_year'] = st.sidebar.number_input('Tahun Kedatangan', min_value=2000, value=2023)
    user_input['arrival_month'] = st.sidebar.number_input('Bulan Kedatangan', min_value=1, max_value=12, value=12)
    user_input['arrival_date'] = st.sidebar.number_input('Tanggal Kedatangan', min_value=1, max_value=31, value=15)
    user_input['repeated_guest'] = st.sidebar.selectbox('Tamu Berulang?', [0,1], index=0)
    user_input['no_of_previous_cancellations'] = st.sidebar.number_input('Jumlah Pembatalan Sebelumnya', min_value=0, value=0)
    user_input['no_of_previous_bookings_not_canceled'] = st.sidebar.number_input('Booking Sukses Sebelumnya', min_value=0, value=0)
    user_input['avg_price_per_room'] = st.sidebar.number_input('Harga Rata-rata per Kamar (€)', min_value=0.0, value=100.0)
    user_input['no_of_special_requests'] = st.sidebar.number_input('Permintaan Khusus', min_value=0, value=0)
    
    # Categorical inputs
    user_input['type_of_meal_plan'] = st.sidebar.selectbox('Jenis Paket Makan', 
        ['Not Selected', 'Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3'])
    user_input['required_car_parking_space'] = st.sidebar.selectbox('Butuh Parkir?', [0,1], index=0)
    user_input['room_type_reserved'] = st.sidebar.selectbox('Jenis Kamar', 
        ['Room_Type 1','Room_Type 2','Room_Type 3','Room_Type 4','Room_Type 5','Room_Type 6','Room_Type 7'])
    user_input['market_segment_type'] = st.sidebar.selectbox('Segmen Pasar', 
        ['Online','Corporate','Offline','Aviation','Complementary','Groups','Direct'])

else:
    user_input = test_cases[option]

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Preprocessing & Prediction
@st.cache()
def preprocess_and_predict(df):
    # 1. Ordinal
    for col, enc in ordinal_encoders.items():
        df[col] = enc.transform(df[[col]])
    # 2. One-hot
    orig_ohe_cols = sorted({'_'.join(c.split('_')[:-1]) for c in one_hot_columns})
    df = pd.get_dummies(df, columns=orig_ohe_cols, dtype='int32')
    for dummy in one_hot_columns:
        if dummy not in df.columns:
            df[dummy] = 0
    # 3. Scale
    df[['avg_price_per_room','lead_time']] = robust_scaler.transform(df[['avg_price_per_room','lead_time']])
    # 4. Reindex
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    # 5. Predict
    preds = model.predict(df)
    probs = model.predict_proba(df)
    return preds, probs

# Run prediction
df_copy = input_df.copy()
preds, probs = preprocess_and_predict(df_copy)
label_map = {0: 'Not_Canceled', 1: 'Canceled'}

st.subheader("Hasil Prediksi")
status = label_map[preds[0]]
confidence = np.round(probs[0].max(), 4)
st.write(f"**Booking Status:** {status}")
st.write(f"**Confidence:** {confidence}")
