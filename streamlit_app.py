import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ——— Load model & preprocessing objects ———
@st.cache(allow_output_mutation=True)
def load_objects():
    with open("best_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler_encoder.pkl", "rb") as scaler_file:
        preprocess = pickle.load(scaler_file)
    return model, preprocess['robust_scaler'], preprocess['ordinal_encoders'], preprocess['one_hot_columns']

model, robust_scaler, ordinal_encoders, one_hot_columns = load_objects()

# ——— Layout to center the input ———
st.title("Hotel Booking Cancellation Prediction")
st.write("Masukkan detail pemesanan untuk memprediksi apakah pemesanan akan dibatalkan.")

# Create two columns for centering the input form
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # ——— Input form for custom data ———
    with st.form("user_input_form"):
        st.subheader("Masukkan Data Pemesanan")
        
        # Numeric inputs
        no_of_adults = st.number_input('Jumlah Dewasa', min_value=0, value=2)
        no_of_children = st.number_input('Jumlah Anak', min_value=0, value=0)
        no_of_weekend_nights = st.number_input('Malam Akhir Pekan', min_value=0, value=1)
        no_of_week_nights = st.number_input('Malam Hari Kerja', min_value=0, value=2)
        lead_time = st.number_input('Lead Time (hari)', min_value=0, value=30)
        arrival_year = st.number_input('Tahun Kedatangan', min_value=2000, value=2023)
        arrival_month = st.number_input('Bulan Kedatangan', min_value=1, max_value=12, value=12)
        arrival_date = st.number_input('Tanggal Kedatangan', min_value=1, max_value=31, value=15)
        repeated_guest = st.selectbox('Tamu Berulang?', [0, 1], index=0)
        no_of_previous_cancellations = st.number_input('Jumlah Pembatalan Sebelumnya', min_value=0, value=0)
        no_of_previous_bookings_not_canceled = st.number_input('Booking Sukses Sebelumnya', min_value=0, value=0)
        avg_price_per_room = st.number_input('Harga Rata-rata per Kamar (€)', min_value=0.0, value=100.0)
        no_of_special_requests = st.number_input('Permintaan Khusus', min_value=0, value=0)
        
        # Categorical inputs
        type_of_meal_plan = st.selectbox('Jenis Paket Makan', 
                                         ['Not Selected', 'Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3'])
        required_car_parking_space = st.selectbox('Butuh Parkir?', [0, 1], index=0)
        room_type_reserved = st.selectbox('Jenis Kamar', 
                                          ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 
                                           'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
        market_segment_type = st.selectbox('Segmen Pasar', 
                                           ['Online', 'Corporate', 'Offline', 'Aviation', 
                                            'Complementary', 'Groups', 'Direct'])

        # Submit button
        submitted = st.form_submit_button("Prediksi Pembatalan")
        
        if submitted:
            # Prepare user input
            user_input = {
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
            }

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
                df[['avg_price_per_room', 'lead_time']] = robust_scaler.transform(df[['avg_price_per_room', 'lead_time']])
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
            st.write(f"**Status Pembatalan:** {status}")
            st.write(f"**Kepercayaan:** {confidence}")

