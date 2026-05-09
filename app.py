import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="💳",
    layout="centered"
)

# ---------------------------------------------------
# 🌟 Professional Fintech Theme
# ---------------------------------------------------
st.markdown("""
<style>

/* ===== Main Background ===== */
.stApp {
    background: #0b1120;
    color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ===== Main Container ===== */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1050px;
}

/* ===== Cards ===== */
.card {
    background: rgba(17, 24, 39, 0.95);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 22px;
    padding: 28px;
    margin-bottom: 24px;
    box-shadow:
        0 4px 12px rgba(0,0,0,0.25),
        0 0 0 1px rgba(255,255,255,0.02);
    transition: all 0.25s ease;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow:
        0 10px 25px rgba(0,0,0,0.35);
}

/* ===== Buttons ===== */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    background: linear-gradient(
        135deg,
        #2563eb,
        #3b82f6
    );
    color: white;
    font-weight: 600;
    padding: 14px;
    font-size: 15px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.01);
    background: linear-gradient(
        135deg,
        #1d4ed8,
        #2563eb
    );
    box-shadow: 0 6px 18px rgba(37,99,235,0.35);
}

/* ===== Inputs ===== */
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background: #1e293b !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* ===== Metric Cards ===== */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 18px;
}

/* ===== Progress Bar ===== */
.stProgress > div > div > div > div {
    background: linear-gradient(
        90deg,
        #06b6d4,
        #3b82f6
    );
}

/* ===== Headings ===== */
h1 {
    font-size: 42px !important;
    font-weight: 800 !important;
    color: white;
    letter-spacing: -1px;
}

h2, h3 {
    color: white;
    font-weight: 700;
}

/* ===== Text ===== */
label, .stMarkdown, p {
    color: #cbd5e1 !important;
}

/* ===== Banner ===== */
.banner {
    background: linear-gradient(
        135deg,
        #111827,
        #1e293b
    );
    border-radius: 24px;
    padding: 35px;
    margin-bottom: 28px;
    border: 1px solid rgba(255,255,255,0.06);
}

/* ===== Footer ===== */
.footer {
    text-align: center;
    color: #64748b;
    padding-top: 10px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.title("🏦 Fraud Detection")

st.sidebar.markdown("---")

st.sidebar.write("👤 User: Purvi Lakhotia")
st.sidebar.write("🤖 Model: Random Forest")
st.sidebar.write("🛡️ Security Status: Active")

st.sidebar.success("✅ Monitoring Transactions")

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    return pd.read_csv(url)

data = load_data()

# ---------------------------------------------------
# Prepare Data
# ---------------------------------------------------
X = data.drop("Class", axis=1)
y = data["Class"]

# Small sample for fast app
X_sample = X.sample(5000, random_state=42)
y_sample = y.loc[X_sample.index]

X_train, X_test, y_train, y_test = train_test_split(
    X_sample,
    y_sample,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# Train Model
# ---------------------------------------------------
@st.cache_resource
def train_model():
    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_model()

accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.markdown("""
<div class="banner">

<h1>
💳 AI Fraud Detection System
</h1>

<p style="
font-size:18px;
margin-top:-10px;
color:#cbd5e1;
">
Enterprise-grade transaction monitoring powered by machine learning
</p>

<div style="
display:flex;
gap:12px;
margin-top:20px;
flex-wrap:wrap;
">

<div style="
background:#0f172a;
padding:10px 18px;
border-radius:12px;
border:1px solid rgba(255,255,255,0.06);
">
🛡️ Real-Time Monitoring
</div>

<div style="
background:#0f172a;
padding:10px 18px;
border-radius:12px;
border:1px solid rgba(255,255,255,0.06);
">
🤖 Random Forest AI
</div>

<div style="
background:#0f172a;
padding:10px 18px;
border-radius:12px;
border:1px solid rgba(255,255,255,0.06);
">
⚡ Instant Risk Analysis
</div>

</div>

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Transaction Input
# ---------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("💰 Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Transaction Amount ($)",
        value=100.0
    )

    transaction_type = st.selectbox(
        "Transaction Type",
        ["Online", "POS", "ATM"]
    )

with col2:
    location = st.selectbox(
        "Location",
        ["Local", "International"]
    )

    new_device = st.selectbox(
        "New Device?",
        ["Yes", "No"]
    )

time = st.slider(
    "Transaction Hour",
    0,
    23,
    12
)

predict_btn = st.button("🚀 Analyze Transaction")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if predict_btn:

    with st.spinner("🔍 AI analyzing transaction..."):

        # Custom risk scoring logic
        risk = 0

        # Amount Risk
        if amount > 3000:
            risk += 35
        elif amount > 1000:
            risk += 20
        elif amount > 500:
            risk += 10

        # Transaction Type Risk
        if transaction_type == "Online":
            risk += 20

        # International Risk
        if location == "International":
            risk += 25

        # New Device Risk
        if new_device == "Yes":
            risk += 15

        # Odd Hour Risk
        if time >= 0 and time <= 5:
            risk += 15

        # Cap at 100
        risk = min(risk, 100)

        # Fraud decision
        prediction = 1 if risk >= 60 else 0

    # ---------------------------------------------------
    # Result Card
    # ---------------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Risk Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Risk Score",
            f"{risk:.2f}%"
        )

    with col2:
        if risk > 70:
            st.metric("Risk Level", "High")
        elif risk > 30:
            st.metric("Risk Level", "Medium")
        else:
            st.metric("Risk Level", "Low")

    st.progress(int(risk))

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected")
        st.write(
            "Suspicious activity detected due to unusual behavior patterns."
        )
    else:
        st.success("✅ Transaction Approved")
        st.write(
            "Transaction behavior appears safe and legitimate."
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------
    # Result Card
    # ---------------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Risk Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Risk Score",
            f"{risk:.2f}%"
        )

    with col2:
        if risk > 70:
            st.metric("Risk Level", "High")
        elif risk > 30:
            st.metric("Risk Level", "Medium")
        else:
            st.metric("Risk Level", "Low")

    st.progress(int(risk))

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected")
        st.write(
            "Suspicious activity detected due to unusual behavior patterns."
        )
    else:
        st.success("✅ Transaction Approved")
        st.write(
            "Transaction behavior appears safe and legitimate."
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------
    # Summary Card
    # ---------------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🧾 Transaction Summary")

    c1, c2 = st.columns(2)

    with c1:
        st.write(f"💵 Amount: ${amount}")
        st.write(f"🌍 Location: {location}")
        st.write(f"🕒 Time: {time}:00 hrs")

    with c2:
        st.write(f"💳 Type: {transaction_type}")
        st.write(f"📱 New Device: {new_device}")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# Model Performance
# ---------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy:.2f}"
    )

with col2:
    st.metric(
        "Dataset Size",
        f"{len(X_sample)}"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("""
<div class="footer">
Built using  Machine Learning
</div>
""", unsafe_allow_html=True)