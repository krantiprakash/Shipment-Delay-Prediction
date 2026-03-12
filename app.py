from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ── Paths & Constants ────────────────────────────────────────────────────────
MODEL_PATH = Path("models/model_pipeline.joblib")
DATA_PATH  = Path("Training Data.csv")
DEFAULT_DECISION_THRESHOLD = 0.50
DEVELOPER_MODE_UI_slider   = True

REQUIRED_INPUT_COLUMNS = [
    "Origin", "Destination", "Shipment Date", "Planned Delivery Date",
    "Vehicle Type", "Distance (km)", "Weather Conditions", "Traffic Conditions",
]
MODEL_FEATURE_COLUMNS = [
    "Origin", "Destination", "Vehicle Type", "Distance (km)",
    "Weather Conditions", "Traffic Conditions",
    "planned_transit_days", "ship_month", "ship_weekday",
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def risk_band(prob: float) -> tuple[str, str, str]:
    if prob >= 0.70: return "High",   "#FF4B4B", "🔴"
    if prob >= 0.40: return "Medium", "#FFA500", "🟡"
    return                  "Low",    "#21C55D", "🟢"

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

@st.cache_data
def load_reference_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def normalize_text(v: str) -> str:
    return str(v).strip().lower().title()

def safe_feature_dict(df: pd.DataFrame) -> dict:
    row = df.iloc[0].to_dict()
    return {k: str(v) if isinstance(v, pd.Timestamp) else v for k, v in row.items()}

def build_single_row_features(
    origin, destination, shipment_date, planned_delivery_date,
    vehicle_type, distance_km, weather, traffic,
) -> pd.DataFrame:
    p = {
        "Origin":                normalize_text(origin),
        "Destination":           normalize_text(destination),
        "Vehicle Type":          normalize_text(vehicle_type),
        "Distance (km)":         float(distance_km),
        "Weather Conditions":    normalize_text(weather),
        "Traffic Conditions":    normalize_text(traffic),
        "Shipment Date":         pd.to_datetime(shipment_date),
        "Planned Delivery Date": pd.to_datetime(planned_delivery_date),
    }
    p["planned_transit_days"] = int((p["Planned Delivery Date"] - p["Shipment Date"]).days)
    p["ship_month"]   = int(p["Shipment Date"].month)
    p["ship_weekday"] = int(p["Shipment Date"].weekday())
    return pd.DataFrame([p])[MODEL_FEATURE_COLUMNS]

# ── CSS ──────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
        /* ── global ── */
        html, body { background: #0f0c29; }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg,#0f0c29,#302b63,#24243e);
            color: #f0f0f0;
        }
        [data-testid="stHeader"] { background: transparent !important; height:0 !important; }

        /* tighten Streamlit's default padding */
        .block-container {
            padding: 0.6rem 1.2rem 1rem !important;
            max-width: 100% !important;
        }

        /* kill excess margin Streamlit adds between every widget */
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }
        /* tighten individual widget bottom margin */
        div.stSelectbox        { margin-bottom: 4px !important; }
        div.stNumberInput      { margin-bottom: 4px !important; }
        div.stDateInput        { margin-bottom: 4px !important; }
        div.stMarkdown         { margin-bottom: 2px !important; }
        div[data-testid="stForm"] { gap: 0 !important; }

        /* shrink label font */
        label, .stSelectbox label, .stNumberInput label, .stDateInput label {
            color: #c4b5fd !important;
            font-size: 0.75rem !important;
            margin-bottom: 1px !important;
            line-height: 1.2 !important;
        }
        /* shrink select/input height */
        div[data-baseweb="select"] > div { min-height: 34px !important; }
        input[type="number"], input[type="date"] {
            height: 34px !important; font-size: 0.82rem !important;
        }

        /* ── top bar ── */
        .topbar {
            background: linear-gradient(90deg,#4f46e5,#7c3aed,#db2777);
            border-radius: 10px; padding: 0.45rem 1.1rem;
            margin-bottom: 0.55rem;
            display:flex; align-items:center; gap:0.7rem;
        }
        .topbar h1 { font-size:1.1rem; margin:0; color:#fff; font-weight:800; }
        .topbar p  { margin:0; color:#d4d0ff; font-size:0.75rem; }

        /* ── glass panels ── */
        .panel {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 12px; padding: 0.75rem 0.9rem;
            backdrop-filter: blur(10px);
        }
        .panel-title {
            font-size:0.65rem; font-weight:800; letter-spacing:0.12em;
            text-transform:uppercase; color:#a78bfa; margin-bottom:0.5rem;
        }
        .section-label {
            font-size:0.68rem; font-weight:700; color:#818cf8;
            text-transform:uppercase; letter-spacing:0.08em;
            margin: 5px 0 3px 0;
        }

        /* ── submit button ── */
        [data-testid="stFormSubmitButton"] button {
            width:100% !important;
            background: linear-gradient(90deg,#4f46e5,#7c3aed) !important;
            color:#fff !important; font-size:0.88rem !important;
            font-weight:700 !important; border:none !important;
            border-radius:8px !important; padding:0.45rem !important;
            box-shadow: 0 3px 14px rgba(124,58,237,0.5);
            margin-top: 4px !important;
        }

        /* ── result panel internals ── */
        .placeholder {
            text-align:center; padding:1.5rem 0;
            color:rgba(255,255,255,0.25); font-size:2.8rem;
        }
        .placeholder p { font-size:0.82rem; margin-top:0.4rem; }

        .verdict-card {
            border-radius:12px; padding:0.9rem 1rem;
            text-align:center; margin-bottom:0.55rem;
        }
        .verdict-card .vc-label { font-size:0.65rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; opacity:0.8; }
        .verdict-card .vc-val   { font-size:2.8rem; font-weight:900; line-height:1.1; }
        .verdict-card .vc-sub   { font-size:0.78rem; opacity:0.75; margin-top:0.15rem; }

        .mini-row { display:flex; gap:0.55rem; margin-bottom:0.55rem; }
        .mini-c {
            flex:1; border-radius:10px; padding:0.65rem 0.5rem;
            text-align:center; box-shadow:0 3px 12px rgba(0,0,0,0.3);
        }
        .mini-c .ml { font-size:0.6rem; font-weight:700; letter-spacing:0.09em; text-transform:uppercase; opacity:0.8; margin-bottom:3px; }
        .mini-c .mv { font-size:1.3rem; font-weight:800; }

        .gauge-wrap  { margin-bottom:0.55rem; }
        .gauge-lbl   { font-size:0.65rem; color:#a78bfa; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:4px; }
        .gauge-track { background:rgba(255,255,255,0.1); border-radius:99px; height:12px; overflow:hidden; }
        .gauge-fill  { height:100%; border-radius:99px; }

        .alert-box { border-radius:9px; padding:0.6rem 0.85rem; font-size:0.82rem; font-weight:600; border-left:4px solid; }
        .alert-high   { background:rgba(255,75,75,0.15);  border-color:#FF4B4B; color:#fca5a5; }
        .alert-medium { background:rgba(255,165,0,0.15);  border-color:#FFA500; color:#fcd34d; }
        .alert-low    { background:rgba(33,197,93,0.15);  border-color:#21C55D; color:#86efac; }

        .chips { display:flex; gap:0.4rem; flex-wrap:wrap; margin-top:0.5rem; }
        .chip  { border-radius:99px; padding:0.15rem 0.6rem; font-size:0.65rem; font-weight:700; }
    </style>
    """, unsafe_allow_html=True)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Shipment Delay Predictor", page_icon="🚚", layout="wide")
    inject_css()

    # top bar
    st.markdown("""
    <div class="topbar">
        <h1>🚚 Shipment Delay Predictor</h1>
        <p>Single-shipment delay risk · Logistic Regression</p>
    </div>
    """, unsafe_allow_html=True)

    # guard
    if not MODEL_PATH.exists():
        st.error(f"Model not found at `{MODEL_PATH}`. Run `python train.py` first.")
        st.stop()
    if not DATA_PATH.exists():
        st.error(f"Reference data not found at `{DATA_PATH}`.")
        st.stop()

    model  = load_model(MODEL_PATH)
    ref_df = load_reference_data(DATA_PATH)
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in ref_df.columns]
    if missing:
        st.error(f"Reference data missing columns: {missing}")
        st.stop()

    def opts(col):
        return (ref_df[col].dropna().astype(str)
                .map(normalize_text).drop_duplicates().sort_values().tolist())

    left, right = st.columns([1.05, 1], gap="medium")

    # ══════════ LEFT — inputs ══════════
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📋 Shipment Details</div>', unsafe_allow_html=True)

        with st.form("shipment_form"):
            st.markdown('<div class="section-label">📍 Route</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1: origin      = st.selectbox("Origin",      opts("Origin"),      key="origin")
            with c2: destination = st.selectbox("Destination", opts("Destination"), key="dest")

            st.markdown('<div class="section-label">🚛 Vehicle & Distance</div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3: vehicle_type = st.selectbox("Vehicle Type", opts("Vehicle Type"))
            with c4: distance_km  = st.number_input("Distance (km)", min_value=1.0, value=500.0, step=1.0)

            st.markdown('<div class="section-label">📅 Schedule</div>', unsafe_allow_html=True)
            c5, c6 = st.columns(2)
            with c5: shipment_date         = st.date_input("Shipment Date",         value=date.today())
            with c6: planned_delivery_date = st.date_input("Planned Delivery Date", value=date.today())

            st.markdown('<div class="section-label">🌦️ Conditions</div>', unsafe_allow_html=True)
            c7, c8 = st.columns(2)
            with c7: weather = st.selectbox("Weather",  opts("Weather Conditions"))
            with c8: traffic = st.selectbox("Traffic",  opts("Traffic Conditions"))

            threshold = float(DEFAULT_DECISION_THRESHOLD)
            if DEVELOPER_MODE_UI_slider:
                threshold = st.slider("Decision Threshold", 0.10, 0.90,
                                      float(DEFAULT_DECISION_THRESHOLD), 0.01)

            submitted = st.form_submit_button("⚡  Predict Delay Risk")

        st.markdown("""
        <div class="chips">
            <span class="chip" style="background:rgba(33,197,93,0.2);color:#86efac;">🟢 Low &lt;0.40</span>
            <span class="chip" style="background:rgba(255,165,0,0.2);color:#fcd34d;">🟡 Medium 0.40–0.69</span>
            <span class="chip" style="background:rgba(255,75,75,0.2);color:#fca5a5;">🔴 High ≥0.70</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)  # close panel

    # ══════════ RIGHT — results ══════════
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🎯 Prediction Result</div>', unsafe_allow_html=True)

        if not submitted:
            st.markdown("""
            <div class="placeholder">
                📦
                <p>Fill in shipment details<br>and click <b>Predict</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            if planned_delivery_date < shipment_date:
                st.error("🚨 Planned Delivery Date must be on or after Shipment Date.")
            else:
                try:
                    X = build_single_row_features(
                        origin, destination, shipment_date, planned_delivery_date,
                        vehicle_type, distance_km, weather, traffic,
                    )
                    cls         = list(model.classes_)
                    pos_label   = "Yes" if "Yes" in cls else cls[1]
                    neg_label   = [c for c in cls if c != pos_label][0]
                    pos_idx     = cls.index(pos_label)
                    delay_prob  = float(model.predict_proba(X)[:, pos_idx][0])
                    pred_label  = pos_label if delay_prob >= threshold else neg_label
                    band, color, emoji = risk_band(delay_prob)
                    pct         = int(delay_prob * 100)
                    vbg         = "#FF4B4B" if pred_label == pos_label else "#21C55D"

                    st.markdown(f"""
                    <div class="verdict-card"
                         style="background:linear-gradient(135deg,{vbg}44,{vbg}1a);
                                border:1px solid {vbg}66;">
                        <div class="vc-label">Shipment Delayed?</div>
                        <div class="vc-val">{pred_label}</div>
                        <div class="vc-sub">{emoji} {band} Risk &nbsp;·&nbsp; {pct}% probability</div>
                    </div>

                    <div class="mini-row">
                        <div class="mini-c"
                             style="background:linear-gradient(135deg,#4f46e5cc,#7c3aedaa);">
                            <div class="ml">Delay Probability</div>
                            <div class="mv">{delay_prob:.4f}</div>
                        </div>
                        <div class="mini-c"
                             style="background:linear-gradient(135deg,{color}cc,{color}66);">
                            <div class="ml">Risk Band</div>
                            <div class="mv">{emoji} {band}</div>
                        </div>
                    </div>

                    <div class="gauge-wrap">
                        <div class="gauge-lbl">Probability Gauge — {pct}%</div>
                        <div class="gauge-track">
                            <div class="gauge-fill"
                                 style="width:{pct}%;
                                        background:linear-gradient(90deg,#4f46e5,{color});">
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if band == "High":
                        st.markdown('<div class="alert-box alert-high">⚠️ High delay risk — review shipment planning or consider rerouting.</div>', unsafe_allow_html=True)
                    elif band == "Medium":
                        st.markdown('<div class="alert-box alert-medium">👀 Medium delay risk — monitor this shipment closely.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="alert-box alert-low">✅ Low delay risk — shipment looks good to go!</div>', unsafe_allow_html=True)

                    if DEVELOPER_MODE_UI_slider:
                        st.caption(f"Threshold: {threshold:.2f}")
                        with st.expander("Engineered features"):
                            for k, v in safe_feature_dict(X).items():
                                st.write(f"**{k}:** {v}")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        st.markdown('</div>', unsafe_allow_html=True)  # close panel


if __name__ == "__main__":
    main()
#==================================================================
# from __future__ import annotations

# from datetime import date
# from pathlib import Path

# import joblib
# import pandas as pd
# import streamlit as st


# # Paths
# MODEL_PATH = Path("models/model_pipeline.joblib")
# DATA_PATH = Path("Training Data.csv")
# DEFAULT_DECISION_THRESHOLD = 0.50
# DEVELOPER_MODE_UI_slider = False

# # Columns
# REQUIRED_INPUT_COLUMNS = [
#     "Origin",
#     "Destination",
#     "Shipment Date",
#     "Planned Delivery Date",
#     "Vehicle Type",
#     "Distance (km)",
#     "Weather Conditions",
#     "Traffic Conditions",
# ]

# MODEL_FEATURE_COLUMNS = [
#     "Origin",
#     "Destination",
#     "Vehicle Type",
#     "Distance (km)",
#     "Weather Conditions",
#     "Traffic Conditions",
#     "planned_transit_days",
#     "ship_month",
#     "ship_weekday",
# ]

# CATEGORICAL_INPUT_COLUMNS = [
#     "Origin",
#     "Destination",
#     "Vehicle Type",
#     "Weather Conditions",
#     "Traffic Conditions",
# ]


# def risk_band(prob: float) -> str:
#     if prob >= 0.70:
#         return "High"
#     if prob >= 0.40:
#         return "Medium"
#     return "Low"


# @st.cache_resource
# def load_model(model_path: Path):
#     return joblib.load(model_path)


# @st.cache_data
# def load_reference_data(data_path: Path) -> pd.DataFrame:
#     return pd.read_csv(data_path)


# def normalize_text(value: str) -> str:
#     return str(value).strip().lower().title()


# def safe_feature_dict(df: pd.DataFrame) -> dict:
#     """
#     Return a plain Python dict for the first row (Arrow-free display).
#     This avoids Streamlit/pyarrow rendering issues with LargeUtf8 types.
#     """
#     row = df.iloc[0].to_dict()
#     safe_row = {}
#     for k, v in row.items():
#         if isinstance(v, (pd.Timestamp,)):
#             safe_row[k] = str(v)
#         else:
#             safe_row[k] = v
#     return safe_row


# def build_single_row_features(
#     origin: str,
#     destination: str,
#     shipment_date: date,
#     planned_delivery_date: date,
#     vehicle_type: str,
#     distance_km: float,
#     weather: str,
#     traffic: str,
# ) -> pd.DataFrame:
#     # Normalize categorical values
#     payload = {
#         "Origin": normalize_text(origin),
#         "Destination": normalize_text(destination),
#         "Vehicle Type": normalize_text(vehicle_type),
#         "Distance (km)": float(distance_km),
#         "Weather Conditions": normalize_text(weather),
#         "Traffic Conditions": normalize_text(traffic),
#         "Shipment Date": pd.to_datetime(shipment_date),
#         "Planned Delivery Date": pd.to_datetime(planned_delivery_date),
#     }

#     # Feature engineering
#     payload["planned_transit_days"] = int(
#         (payload["Planned Delivery Date"] - payload["Shipment Date"]).days
#     )
#     payload["ship_month"] = int(payload["Shipment Date"].month)
#     payload["ship_weekday"] = int(payload["Shipment Date"].weekday())

#     # Keep only training-time model features
#     row_df = pd.DataFrame([payload])[MODEL_FEATURE_COLUMNS]
#     return row_df


# def main():
#     st.set_page_config(page_title="Shipment Delay Predictor", page_icon="🚚", layout="centered")
#     st.markdown(
#         """
#         <style>
#             .block-container {padding-top: 2rem; padding-bottom: 2rem;}
#             .small-note {font-size: 0.9rem; color: #666;}
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

#     st.title("🚚 Shipment Delay Predictor")
#     st.caption("Single-shipment prediction using the finalized Logistic Regression model.")

#     # Load artifacts
#     if not MODEL_PATH.exists():
#         st.error(
#             f"Model not found at `{MODEL_PATH}`. Please run `python train.py` first."
#         )
#         st.stop()

#     if not DATA_PATH.exists():
#         st.error(
#             f"Reference data not found at `{DATA_PATH}`. It is needed to build dropdown options."
#         )
#         st.stop()

#     model = load_model(MODEL_PATH)
#     ref_df = load_reference_data(DATA_PATH)

#     # Validate required columns in reference data
#     missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in ref_df.columns]
#     if missing:
#         st.error(f"Reference data missing required columns: {missing}")
#         st.stop()

#     # Dropdown options from reference data
#     def options_for(col: str):
#         vals = (
#             ref_df[col]
#             .dropna()
#             .astype(str)
#             .map(normalize_text)
#             .drop_duplicates()
#             .sort_values()
#             .tolist()
#         )
#         return vals

#     origin_options = options_for("Origin")
#     destination_options = options_for("Destination")
#     vehicle_options = options_for("Vehicle Type")
#     weather_options = options_for("Weather Conditions")
#     traffic_options = options_for("Traffic Conditions")

#     st.info(
#         "Provide shipment details below and click **Predict Delay** to get prediction, probability, and risk band."
#     )

#     with st.expander("How prediction is calculated"):
#         st.markdown(
#             "- Model computes **delay probability** = P(Delayed = Yes)\n"
#             "- If probability >= selected threshold -> **Yes**, else **No**\n"
#             "- Risk bands: Low (<0.40), Medium (0.40–0.69), High (>=0.70)"
#         )

#     # Input form
#     with st.form("shipment_form"):
#         st.subheader("Shipment Input")
#         threshold = float(DEFAULT_DECISION_THRESHOLD)
#         col1, col2 = st.columns(2)
#         with col1:
#             origin = st.selectbox("Origin", origin_options)
#             vehicle_type = st.selectbox("Vehicle Type", vehicle_options)
#             weather = st.selectbox("Weather Conditions", weather_options)
#             shipment_date = st.date_input("Shipment Date", value=date.today())
#         with col2:
#             destination = st.selectbox("Destination", destination_options)
#             traffic = st.selectbox("Traffic Conditions", traffic_options)
#             distance_km = st.number_input("Distance (km)", min_value=1.0, value=500.0, step=1.0)
#             planned_delivery_date = st.date_input(
#                 "Planned Delivery Date", value=date.today()
#             )

#         if DEVELOPER_MODE_UI_slider:
#             threshold = st.slider(
#                 "Decision Threshold (for Delayed = Yes)",
#                 min_value=0.10,
#                 max_value=0.90,
#                 value=float(DEFAULT_DECISION_THRESHOLD),
#                 step=0.01,
#                 help="If Delay Probability >= threshold, prediction is Yes; otherwise No.",
#             )

#         submitted = st.form_submit_button("Predict Delay")

#     if submitted:
#         if planned_delivery_date < shipment_date:
#             st.error("Planned Delivery Date must be on or after Shipment Date.")
#             st.stop()

#         try:
#             X_infer = build_single_row_features(
#                 origin=origin,
#                 destination=destination,
#                 shipment_date=shipment_date,
#                 planned_delivery_date=planned_delivery_date,
#                 vehicle_type=vehicle_type,
#                 distance_km=distance_km,
#                 weather=weather,
#                 traffic=traffic,
#             )

#             class_order = list(model.classes_)
#             pos_label = "Yes" if "Yes" in class_order else class_order[1]
#             neg_label = [c for c in class_order if c != pos_label][0]
#             pos_idx = class_order.index(pos_label)
#             delay_prob = float(model.predict_proba(X_infer)[:, pos_idx][0])
#             pred_label = pos_label if delay_prob >= threshold else neg_label
#             band = risk_band(delay_prob)

#             st.subheader("Prediction Result")
#             c1, c2, c3 = st.columns(3)
#             c1.metric("Predicted Delayed", str(pred_label))
#             c2.metric("Delay Probability", f"{delay_prob:.4f}")
#             c3.metric("Risk Band", band)
#             if DEVELOPER_MODE_UI_slider:
#                 st.caption(f"Decision rule used: Predict Yes when probability >= {threshold:.2f}")

#             if DEVELOPER_MODE_UI_slider:
#                 with st.expander("Show engineered features used by model"):
#                     feature_view = safe_feature_dict(X_infer)
#                     for key, value in feature_view.items():
#                         st.write(f"**{key}:** {value}")

#             if band == "High":
#                 st.warning("High delay risk. Consider reviewing shipment planning.")
#             elif band == "Medium":
#                 st.info("Medium delay risk. Monitor shipment closely.")
#             else:
#                 st.success("Low delay risk.")

#         except Exception as e:
#             st.error(f"Prediction failed: {e}")


# if __name__ == "__main__":
#     main()
