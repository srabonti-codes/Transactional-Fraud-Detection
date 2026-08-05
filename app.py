# IMPORTS
import datetime
import json
import os
import pickle
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.models import load_model
from pytorch_tabnet.tab_model import TabNetClassifier

# STREAMLIT CONFIGURATION
st.set_page_config(
    page_title="Transactional Fraud Detector | Deep Learning-Based Fraud Detection",
    page_icon=":material/shield:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

def register_health_endpoint():
    try:
        from streamlit.web.server.server import Server
    except Exception:
        return False

    try:
        server = Server.get_current()
    except Exception:
        return False

    if not hasattr(server, "add_route"):
        return False

    def health_handler(*_args, **_kwargs):
        return "ok"

    try:
        server.add_route("/health", health_handler)
        return True
    except Exception:
        return False


def register_seo_routes():
    try:
        from streamlit.web.server.server import Server
    except Exception:
        return False

    try:
        server = Server.get_current()
    except Exception:
        return False

    if not hasattr(server, "add_route"):
        return False

    def robots_handler(*_args, **_kwargs):
        site_url = os.getenv(
            "SITE_URL",
            "https://transactional-fraud-detector.onrender.com"
        ).rstrip("/")
        return (
            f"User-agent: *\n"
            f"Allow: /\n"
            f"Sitemap: {site_url}/sitemap.xml\n"
        )

    def sitemap_handler(*_args, **_kwargs):
        site_url = os.getenv(
            "SITE_URL",
            "https://transactional-fraud-detector.onrender.com"
        ).rstrip("/")

        return (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n"
            f"  <url><loc>{site_url}/</loc><changefreq>weekly</changefreq><priority>1.0</priority></url>\n"
            f"  <url><loc>{site_url}/health</loc><changefreq>monthly</changefreq><priority>0.5</priority></url>\n"
            "</urlset>\n"
        )

    try:
        server.add_route("/robots.txt", robots_handler)
        server.add_route("/sitemap.xml", sitemap_handler)
        return True
    except Exception:
        return False


def start_public_keepalive():
    site_url = os.getenv("SITE_URL", "https://transactional-fraud-detection.onrender.com").rstrip("/")
    target_url = f"{site_url}/health"
    interval = 300

    def ping_loop():
        while True:
            try:
                with urllib.request.urlopen(target_url, timeout=10) as response:
                    if response.status >= 400:
                        raise urllib.error.URLError(f"Keepalive returned {response.status}")
            except Exception:
                pass
            time.sleep(interval)

    threading.Thread(target=ping_loop, daemon=True).start()
    return True


register_health_endpoint()
register_seo_routes()
start_public_keepalive()

site_url = os.getenv("SITE_URL", "https://transactional-fraud-detection.onrender.com").rstrip("/")
st.markdown(
    f"""
    <meta name=\"description\" content=\"AI-powered fraud detection demo for analyzing financial transaction risk using machine learning.\" />
    <meta name=\"keywords\" content=\"fraud detection, transaction fraud, AI fraud detection, financial security, machine learning demo\" />
    <meta name=\"robots\" content=\"index,follow\" />
    <link rel=\"canonical\" href=\"{site_url}/\" />
    """,
    unsafe_allow_html=True,
)

st.title("Financial Transaction Fraud Detection")
st.caption("Predict whether a financial transaction is fraudulent or legitimate using the trained model.")

st.info("Secure review workflow • Instant fraud scoring • Risk-based insights")
st.divider()

# PATHS
ROOT_PATH = Path.cwd().resolve()

if ROOT_PATH.name == "notebooks":
    ROOT_PATH = ROOT_PATH.parent

DATA_DIR = ROOT_PATH / "data"
ARTIFACT_DIR = ROOT_PATH / "artifacts"
MODEL_DIR = ROOT_PATH / "models"

BEST_MODEL_METADATA_PATH = MODEL_DIR / "best_model_metadata.json"
PREPROCESSING_BUNDLE_PATH = ARTIFACT_DIR / "preprocessing.pkl"

# LOAD MODEL
@st.cache_resource
def load_artifacts():

    if not BEST_MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Missing models/best_model_metadata.json"
        )

    with open(BEST_MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        best_model_metadata = json.load(f)

    best_model_name = best_model_metadata["model_name"]
    model_family = best_model_metadata["model_family"]

    model_path = ROOT_PATH / best_model_metadata["artifact_path"]

    if not PREPROCESSING_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            "Missing artifacts/preprocessing.pkl"
        )

    with open(PREPROCESSING_BUNDLE_PATH, "rb") as f:
        preprocessing_bundle = pickle.load(f)

    feature_cols = preprocessing_bundle["feature_cols"]

    metadata = preprocessing_bundle["metadata"]

    scaler = preprocessing_bundle["scaler"]

    decision_thresholds = preprocessing_bundle.get(
        "decision_thresholds",
        {}
    ).copy()

    if (
        best_model_name not in decision_thresholds
        and best_model_metadata.get("decision_threshold")
    ):
        decision_thresholds[best_model_name] = (
            best_model_metadata["decision_threshold"]
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    if model_family == "keras":

        model = load_model(model_path)

    elif model_family == "tabnet":

        model = TabNetClassifier()
        model.load_model(str(model_path))

    else:

        raise ValueError(
            f"Unsupported model family: {model_family}"
        )

    return (
        model,
        best_model_name,
        feature_cols,
        metadata,
        scaler,
        decision_thresholds,
    )

# INITIALIZE
try:

    (
        model,
        best_model_name,
        feature_cols,
        metadata,
        scaler,
        decision_thresholds,
    ) = load_artifacts()

except Exception as e:

    st.error(str(e))
    st.stop()

# PREPROCESSING HELPERS
FEATURE_COLS = [
    "log_amount",
    "error_orig",
    "error_dest",
    "hour",
    "is_night",
    "is_high_amount",
    "type_encoded",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

TYPE_MAPPING = {
    "TRANSFER": 0,
    "CASH_OUT": 1,
}

RAW_NUMERIC_COLS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

# CLEAN RAW DATA
def clean_raw_transactions(df):

    df = df.copy()

    for col in RAW_NUMERIC_COLS:

        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "type" not in df.columns:
        raise ValueError("Missing required column: type")

    if "isFraud" in df.columns:
        df["isFraud"] = (
            pd.to_numeric(df["isFraud"], errors="coerce")
            .astype("Int64")
        )

    df = df.dropna(subset=["type"] + RAW_NUMERIC_COLS)

    return df

# BOOLEAN CONVERTER
def _binary_indicator(series):

    numeric = pd.to_numeric(series, errors="coerce")

    text = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return (
        numeric.fillna(
            text.isin(["true", "yes", "1"]).astype(int)
        )
        .astype(int)
    )

# FEATURE ENGINEERING
def engineer_features(
    df,
    high_amount_threshold,
    type_mapping=None,
):

    df = clean_raw_transactions(df)

    type_mapping = (
        TYPE_MAPPING
        if type_mapping is None
        else type_mapping
    )

    df = df[df["type"].isin(type_mapping)].copy()

    df["log_amount"] = np.log1p(
        df["amount"].clip(lower=0)
    )

    df["error_orig"] = (
        df["oldbalanceOrg"]
        - df["amount"]
        - df["newbalanceOrig"]
    )

    df["error_dest"] = (
        df["oldbalanceDest"]
        + df["amount"]
        - df["newbalanceDest"]
    )

    df["hour"] = (
        df["step"] % 24
    ).astype(int)

    df["is_night"] = (
        (df["hour"] >= 22)
        | (df["hour"] <= 5)
    ).astype(int)

    df["is_high_amount"] = (
        df["amount"] > high_amount_threshold
    ).astype(int)

    df["type_encoded"] = (
        df["type"]
        .map(type_mapping)
        .astype(int)
    )

    if "nameDest" in df.columns:

        df["is_merchant_dest"] = (
            df["nameDest"]
            .astype(str)
            .str.startswith("M")
            .astype(int)
        )

    elif "is_merchant_dest" in df.columns:

        df["is_merchant_dest"] = _binary_indicator(
            df["is_merchant_dest"]
        )

    else:

        df["is_merchant_dest"] = 0

    for col in FEATURE_COLS + ["is_merchant_dest"]:

        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .fillna(0)
        )

    return df

# DATE/TIME INPUT MAPPING

def compute_step_from_datetime(date_obj, time_obj):
    dt = datetime.datetime.combine(date_obj, time_obj)
    epoch = datetime.datetime(1970, 1, 1)
    return int((dt - epoch).total_seconds() // 3600)


# PREPARE MODEL INPUT
def prepare_model_input(input_data):
    engineered = engineer_features(
        input_data,
        metadata["high_amount_threshold"],
        metadata["type_mapping"],
    )

    if engineered.empty:
        raise ValueError(
            "Transaction type must be TRANSFER or CASH_OUT."
        )

    X = engineered.reindex(
        columns=feature_cols,
        fill_value=0,
    )

    X = scaler.transform(X).astype("float32")

    return X

# PREDICTION TRANSACTION
def predict_transaction(
    input_data,
    model,
    best_model_name,
):

    X = prepare_model_input(input_data)

    if best_model_name in {"GRU", "BiLSTM"}:

        X_model = X.reshape(
            (X.shape[0], 1, X.shape[1])
        )

    else:

        X_model = X

    if best_model_name == "TabNet":

        probability = float(
            model.predict_proba(X_model)[0, 1]
        )

    else:

        probability = float(
            model.predict(
                X_model,
                verbose=0,
            ).ravel()[0]
        )

    threshold = decision_thresholds.get(
        best_model_name,
        {},
    ).get(
        "threshold",
        0.5,
    )

    prediction = int(
        probability >= threshold
    )

    return prediction, probability, threshold

# TRANSACTION INPUT
st.subheader("Transaction Details")
st.caption("Complete the transaction fields below to evaluate the transaction risk.")

with st.form("prediction_form"):

    transaction_type = st.selectbox(
        "Transaction Type",
        options=["TRANSFER", "CASH_OUT"],
    )

    col1, col2 = st.columns(2)

    with col1:

        transaction_date = st.date_input(
            "Transaction Date",
            value=datetime.date.today(),
        )

        transaction_time = st.time_input(
            "Transaction Time",
            value=datetime.time(12, 0),
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            format="%.2f",
        )

        oldbalanceOrg = st.number_input(
            "Sender Balance Before",
            min_value=0.0,
            value=5000.0,
            step=100.0,
            format="%.2f",
        )

    with col2:

        newbalanceOrig = st.number_input(
            "Sender Balance After",
            min_value=0.0,
            value=4000.0,
            step=100.0,
            format="%.2f",
        )

        oldbalanceDest = st.number_input(
            "Receiver Balance Before",
            min_value=0.0,
            value=10000.0,
            step=100.0,
            format="%.2f",
        )

        newbalanceDest = st.number_input(
            "Receiver Balance After",
            min_value=0.0,
            value=11000.0,
            step=100.0,
            format="%.2f",
        )

    predict_button = st.form_submit_button(
        "Predict Transaction",
        use_container_width=True,
    )

# CREATE INPUT DATA
if predict_button:

    step = compute_step_from_datetime(
        transaction_date,
        transaction_time,
    )

    transaction = {
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
    }

    input_df = pd.DataFrame([transaction])

# MAKE PREDICTION
if predict_button:

    try:

        prediction, probability, threshold = predict_transaction(
            input_df,
            model,
            best_model_name,
        )

        confidence = probability * 100

        st.divider()

        st.subheader("Prediction Result")

        st.progress(float(probability))

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="Fraud Probability",
                value=f"{confidence:.2f}%"
            )

        with col2:

            st.metric(
                label="Decision Threshold",
                value=f"{threshold:.2f}"
            )

        if prediction == 1:

            st.markdown(
                f"""
<div class="result-card"
     style="
        border-left:4px solid #f87171;
        background:rgba(239,68,68,0.06);
        border-radius:18px;
        padding:22px;
        margin-top:15px;
     ">

<h3 style="color:#ef4444; margin-top:0;">
    Fraudulent Transaction Detected
</h3>

<p><b>Status:</b> High Risk</p>

<p><b>Fraud Probability:</b> {confidence:.2f}%</p>

<p><b>Recommendation:</b> Review or block this transaction before processing.</p>

</div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
<div class="result-card"
     style="
        border-left:4px solid #22c55e;
        background:rgba(34,197,94,0.04);
        border-radius:18px;
        padding:22px;
        margin-top:15px;
     ">

<h3 style="color:#22c55e; margin-top:0;">
    Legitimate Transaction
</h3>

<p><b>Status:</b> Low Risk</p>

<p><b>Fraud Probability:</b> {confidence:.2f}%</p>

<p><b>Recommendation:</b> Transaction appears safe to proceed.</p>

</div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as e:

        st.error(f"Prediction failed: {e}")

# FOOTER
st.markdown(
    """
<div class="footer">

<hr>

Fraud Detection System • Deep Learning Powered • Streamlit Application

</div>
""",
    unsafe_allow_html=True,
)