import streamlit as st
import pandas as pd
import joblib
import random

# Modeli yüklə
model = joblib.load("rf_model10_encoded.pkl")

# Başlıq və inputlar
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("💳 Customer Fraud Detection App")

with st.sidebar:
    # Bütün input sahələri
    amount = st.slider("Amount ($)", 0, 10000, 100)
    merchant = st.selectbox("Merchant", ['Amazon', 'AliExpress',
    'McDonalds', 'Starbucks', 'KFC',
    'Bolt', 'Turkish Airlines',
    'Netflix', 'YouTube Premium', 'Spotify',
    'Aptek+', 'Baku Med Center',
    'Azercell', 'Azəriqaz', 'Azərsu',
    'ABB', 'Kapital Bank'])
    transaction_type = st.selectbox("Transaction Type", ['online', 'pos', 'atm', 'mobile_app'])
    transaction_category = st.selectbox("Category", ['travel', 'food', 'shopping', 'entertainment',
    'healthcare', 'utilities', 'education',
    'charity', 'other'])
    location = st.selectbox("Location", ['Baku', 'Ganja', 'Sumqayit', 'Sheki', 'Shirvan',
    'Mingachevir', 'Lankaran', 'Nakhchivan', 'Xankendi',
    'Quba', 'Gabala', 'Zaqatala'])
    gender = st.radio("Gender", ["Male", "Female"])
    hour = st.slider("Hour", 0, 23, 12)
    dayofweek = st.selectbox("Day of Week", [0, 1, 2, 3, 4, 5, 6])
    month = st.selectbox("Month", [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
])

# Yeni məlumatları DataFrame kimi topla
input_df = pd.DataFrame({
    'amount': [amount],
    'merchant': [merchant],
    'transaction_type': [transaction_type],
    'transaction_category': [transaction_category],
    'location': [location],
    'gender': [gender],
    'hour': [hour],
    'dayofweek': [dayofweek],
    'month': [month]
})

st.subheader("🧾 Daxil etdiyin məlumatlar:")
st.dataframe(input_df)

# Fraud ehtimalinin yoxlanmasi
if st.button("🚨 Fraud ehtimalını yoxla"):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ FRAUD! Bu əməliyyat şübhəlidir. Ehtimal: {proba:.2f}")
    else:
        st.success(f"✅ Etibarlı əməliyyat. Ehtimal: {proba:.2f}")

# 📊 Vizual progress bar əlavə edirik
    st.progress(int(proba * 100))

    # 📌 Ehtimal badge (metrik)
    st.metric("🧮 Fraud Ehtimalı", f"{proba:.2%}")


    import os

    log_file = r"C:\Users\user\Documents\fraud_log_live.csv"

    # DataFrame-ə prediction və proba əlavə edirik
    input_df["prediction"] = prediction[0]
    input_df["fraud_probability"] = proba

    # Faylı varsa əlavə edirik, yoxdursa yaradir
    if os.path.exists(log_file):
        input_df.to_csv(log_file, mode='a', index=False, header=False)
    else:
        input_df.to_csv(log_file, mode='w', index=False, header=True)

    st.success('📁 Əməliyyat log faylına yazıldı!')


    # === 40000-lik datasetə də əlavə et (fraud_transactions7.csv) ===
    input_df['transaction_id'] = random.randint(100000, 999999)
    input_df['card_id'] = random.randint(1000000000000000, 9999999999999999)
    
    from datetime import datetime

    date = st.date_input("Transaction Date")
    time = st.time_input("Transaction Time")
    timestamp = datetime.combine(date, time)

    input_df['timestamp'] = timestamp
    input_df['account_balance'] = round(random.uniform(100, 10000), 2)
    input_df['is_fraud'] = input_df['prediction']
    input_df["prediction"] = prediction[0]
    input_df["fraud_probability"] = proba
    df_final = input_df[[
        'transaction_id', 'card_id', 'timestamp', 'amount', 'merchant',
        'transaction_type', 'transaction_category', 'location', 'gender',
        'account_balance', 'is_fraud','hour','dayofweek','month','fraud_probability'
    ]]

    main_file = r"C:\Users\user\Documents\fraud_predictions_full10.csv"

    if os.path.exists(main_file):
        df_final.to_csv(main_file, mode='a', index=False, header=False)
    else:
        df_final.to_csv(main_file, mode='w', index=False, header=True)

    st.info("✅ Əlavə olaraq əsas baza faylına da yazıldı (fraud_predictions_full10.csv)")
