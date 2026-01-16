# app.py
import streamlit as st
import numpy as np
import tensorflow as tf

from schema import SCHEMA, TF_DTYPES, assert_schema_consistency

st.set_page_config(page_title="SAP Customer Predictor", layout="centered")

MODEL_PATH = "customer_model.keras"

# If your satisfaction target is 0..5 keep this.
# If it's 0..1 set SAT_MAX=1.0
# If it's 0..100 set SAT_MAX=100.0
SAT_MIN = 0.0
SAT_MAX = 5.0

RETENTION_THRESHOLD = 0.50

GEOGRAPHY_OPTIONS = ["APAC", "North America", "Europe", "ASIA", "South America"]


def yn_float(v: str) -> float:
    return 1.0 if v.strip().lower() == "yes" else 0.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sat_to_stars(sat_raw: float) -> float:
    sat_raw = clamp(sat_raw, SAT_MIN, SAT_MAX)
    if SAT_MAX - SAT_MIN == 0:
        return 0.0
    stars = (sat_raw - SAT_MIN) / (SAT_MAX - SAT_MIN) * 5.0
    return clamp(stars, 0.0, 5.0)


def render_stars(stars_0_to_5: float) -> str:
    s = round(stars_0_to_5 * 2) / 2  # nearest 0.5
    full = int(s)
    half = 1 if (s - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + ("⯪" if half else "") + "☆" * empty


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def validate_feature_keys_against_model(model: tf.keras.Model):
    model_inputs = sorted([t.name.split(":")[0] for t in model.inputs])
    schema_inputs = sorted(SCHEMA.model_input_names())
    if model_inputs != schema_inputs:
        raise ValueError(
            "Schema mismatch!\n"
            f"Model expects: {model_inputs}\n"
            f"Schema provides: {schema_inputs}\n"
            "Fix: ensure schema.py is identical for training and inference, then retrain."
        )


def make_feature_dict(raw: dict) -> dict:
    """
    Convert raw Python values into a schema-safe model input dict:
    - Strings -> tf.constant(..., dtype=tf.string)
    - Floats  -> np.float32 arrays
    """
    features = {}
    for name in SCHEMA.model_input_names():
        dtype = TF_DTYPES[name]
        v = raw[name]

        if dtype == "string":
            # IMPORTANT: avoid numpy dtype=object; use tf.string tensor
            features[name] = tf.constant([[str(v)]], dtype=tf.string)
        elif dtype == "float32":
            features[name] = np.array([[float(v)]], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported dtype in schema: {name} -> {dtype}")

    return features


def main():
    assert_schema_consistency()

    st.title("Customer Satisfaction & Retention Predictor")

    model = load_model()
    validate_feature_keys_against_model(model)

    st.subheader("Inputs")

    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            active_flag = st.selectbox("Active Flag", ["Yes", "No"], index=0)
            customer_claims = st.number_input("Customer Claims", min_value=0, step=1, value=0)
            incident_count = st.number_input("Incident Count", min_value=0, step=1, value=0)
            incident_res_hours = st.number_input("Incident Resolution Time (hours)", min_value=0.0, step=1.0, value=0.0)

        with col2:
            in_time_payment = st.selectbox("In-time Payment", ["Yes", "No"], index=0)
            invoice_on_time = st.selectbox("Invoice On-time", ["Yes", "No"], index=0)
            geography = st.selectbox("Geography", GEOGRAPHY_OPTIONS, index=0)

            eu_vat_issue = "No"
            if geography == "Europe":
                eu_vat_issue = st.selectbox("EU VAT Issue", ["Yes", "No"], index=1)

        st.divider()

        customer_id = st.text_input("Customer ID", value="CUST_001")

        # App takes year/month directly (derived from date during training)
        posting_year = st.number_input("Posting Year", min_value=1900, max_value=2100, step=1, value=2026)
        posting_mon = st.number_input("Posting Month (1-12)", min_value=1, max_value=12, step=1, value=1)

        submitted = st.form_submit_button("Submit")

    if submitted:
        raw = {
            "CUSTOMER_ID": customer_id,
            "GEOGRAPHY": geography,

            "ACTIVE_FLAG": yn_float(active_flag),
            "IN_TIME_PAYMENT": yn_float(in_time_payment),
            "INVOICE_ON_TIME": yn_float(invoice_on_time),
            "EU_VAT_ISSUE": yn_float(eu_vat_issue),

            "CUSTOMER_CLAIMS": float(customer_claims),
            "INCIDENT_COUNT": float(incident_count),
            "INCIDENT_RESOLVED_IN_TIME": float(incident_res_hours),

            # Derived date fields (float32)
            "POSTING_YEAR": float(posting_year),
            "POSTING_MON": float(posting_mon),
        }

        features = make_feature_dict(raw)

        pred = model.predict(features, verbose=0)

        sat_raw = float(pred[SCHEMA.OUT_SAT][0][0])
        ret_prob = float(pred[SCHEMA.OUT_RET][0][0])
        ret_label = "Yes" if ret_prob >= RETENTION_THRESHOLD else "No"

        stars = sat_to_stars(sat_raw)

        st.subheader("Results")

        st.markdown("### Customer Satisfaction")
        st.markdown(f"**Stars:** {render_stars(stars)}  \n**Score:** {stars:.2f} / 5.00")
        st.caption(f"Model raw output: {sat_raw:.4f} (scaled with SAT_MIN={SAT_MIN}, SAT_MAX={SAT_MAX})")

        st.markdown("### Customer Retention")
        st.markdown(f"**Retained:** {ret_label}  \n**Probability:** {ret_prob:.3f}")


if __name__ == "__main__":
    main()