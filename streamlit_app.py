import streamlit as st
import pandas as pd
import requests
from datetime import datetime

st.set_page_config(page_title="FinLora Fraud Detection", page_icon="🛡️", layout="wide")

CSV_PATH = r"C:\Users\user\Downloads\Fraudulent_Transaction_Detection_for_Finlora_Company\Finlora_dataset\artifacts\cleaned_Data.csv"
API_URL = "http://127.0.0.1:8000"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

df = load_data()

def unique(col):
    return df[col].dropna().unique().tolist()

def default_index(options, value):
    return next((i for i, x in enumerate(options) if str(x) == str(value)), 0)

st.title("🛡️ FinLora Fraud Detection Dashboard")

customer_ids = sorted(df["customer_id"].dropna().astype(str).unique())
customer_id = st.selectbox("Customer ID", customer_ids)

customer_history = df[df["customer_id"].astype(str) == customer_id].sort_values("timestamp")
customer = customer_history.iloc[-1]

st.subheader("Customer Information")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customer ID", customer_id)
c2.metric("Home Country", customer.get("home_country", "N/A"))
c3.metric("KYC Tier", customer.get("kyc_tier", "N/A"))
c4.metric("Account Age", f"{customer.get('account_age_days', 0)} days")

home_countries = unique("home_country")
source_currencies = unique("source_currency")
dest_currencies = unique("dest_currency")
channels = unique("channel")
ip_countries = unique("ip_country")
kyc_tiers = unique("kyc_tier")
new_devices = unique("new_device")
location_mismatches = unique("location_mismatch")

st.subheader("Transaction Details")

with st.form("transaction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        timestamp = st.text_input("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        home_country = st.selectbox("Home Country", home_countries,
                                    index=default_index(home_countries, customer.get("home_country")))
        source_currency = st.selectbox("Source Currency", source_currencies,
                                       index=default_index(source_currencies, customer.get("source_currency")))
        dest_currency = st.selectbox("Destination Currency", dest_currencies,
                                     index=default_index(dest_currencies, customer.get("dest_currency")))
        channel = st.selectbox("Channel", channels,
                               index=default_index(channels, customer.get("channel")))

    with col2:
        amount_src = st.number_input("Amount", min_value=0.0,
                                     value=float(customer.get("amount_src", 0) or 0))
        fee = st.number_input("Fee", min_value=0.0,
                              value=float(customer.get("fee", 0) or 0))
        new_device = st.selectbox("New Device", new_devices,
                                  index=default_index(new_devices, customer.get("new_device")))
        ip_country = st.selectbox("IP Country", ip_countries,
                                  index=default_index(ip_countries, customer.get("ip_country")))
        location_mismatch = st.selectbox("Location Mismatch", location_mismatches,
                                         index=default_index(location_mismatches, customer.get("location_mismatch")))

    with col3:
        ip_risk_score = st.number_input("IP Risk Score", 0.0, 1.0,
                                        float(customer.get("ip_risk_score", 0) or 0))
        kyc_tier = st.selectbox("KYC Tier", kyc_tiers,
                                index=default_index(kyc_tiers, customer.get("kyc_tier")))
        account_age_days = st.number_input("Account Age Days", min_value=0,
                                           value=int(customer.get("account_age_days", 0) or 0))
        device_trust_score = st.number_input("Device Trust Score", 0.0, 1.0,
                                             float(customer.get("device_trust_score", 0) or 0))
        risk_score_internal = st.number_input("Internal Risk Score", 0.0, 1.0,
                                              float(customer.get("risk_score_internal", 0) or 0))
        corridor_risk = st.number_input("Corridor Risk", 0.0, 1.0,
                                        float(customer.get("corridor_risk", 0) or 0))

    submit = st.form_submit_button("Predict Fraud")

if submit:
    payload = {
        "timestamp": timestamp,
        "customer_id": customer_id,
        "home_country": str(home_country),
        "source_currency": str(source_currency),
        "dest_currency": str(dest_currency),
        "channel": str(channel),
        "amount_src": float(amount_src),
        "fee": float(fee),
        "new_device": str(new_device),
        "ip_country": str(ip_country),
        "location_mismatch": str(location_mismatch),
        "ip_risk_score": float(ip_risk_score),
        "kyc_tier": str(kyc_tier),
        "account_age_days": int(account_age_days),
        "device_trust_score": float(device_trust_score),
        "risk_score_internal": float(risk_score_internal),
        "corridor_risk": float(corridor_risk)
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()

            if result["is_fraud"] == 1:
                st.error("⚠️ FRAUDULENT TRANSACTION")
            else:
                st.success("✅ LEGITIMATE TRANSACTION")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fraud Probability", f"{result['fraud_probability']:.2%}")
            c2.metric("1 Hour Velocity", result.get("txn_velocity_1h", "N/A"))
            c3.metric("24 Hour Velocity", result.get("txn_velocity_24h", "N/A"))
            c4.metric("Amount USD",
                      f"${result['amount_usd']:,.2f}" if result.get("amount_usd") is not None else "N/A")

            st.write("Velocity Spike:", result.get("velocity_spike", "N/A"))

            with st.expander("API Response"):
                st.json(result)
        else:
            st.error(f"API Error {response.status_code}")
            st.json(response.json())

    except Exception as e:
        st.error(f"Error connecting to API: {e}")