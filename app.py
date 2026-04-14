"""
app.py — UrbanNest Analytics: Dynamic House Rent Prediction Engine
Task 2: Streamlit Web Application

HOW TO RUN:
    streamlit run app.py

WHAT THIS FILE DOES:
    1. Loads the trained Random Forest model  →  models/best_rf_model.pkl
    2. Loads the saved encoders               →  models/encoders.pkl
         encoders.pkl contains a dict with TWO keys:
           - "encoders"         → {"city": <LabelEncoder>, "location": <LabelEncoder>, ...}
           - "train_categories" → {"city": {"Mumbai": 0, "Delhi": 1, ...}, ...}
    3. Builds dropdowns dynamically from train_categories (no hardcoding needed!)
    4. On button click: encodes inputs → feeds to model → shows predicted rent

Dataset features (in this exact order):
    ['location', 'city', 'latitude', 'longitude', 'numBathrooms',
     'numBalconies', 'isNegotiable', 'SecurityDeposit', 'Status',
     'Size_ft²', 'BHK', 'rooms_num', 'property_type', 'verification_days']
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: PAGE CONFIG
# Must be the VERY FIRST streamlit call in the entire script.
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="UrbanNest — Rent Predictor",
    page_icon="🏠",
    layout="centered",
)


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: LOAD MODEL & ENCODERS
#
# Your notebook saved encoders.pkl with this structure:
#   {
#     "encoders":         {"city": <LE>, "location": <LE>, "Status": <LE>, "property_type": <LE>},
#     "train_categories": {"city": {"Mumbai": 0, "Delhi": 1, ...},
#                          "location": {"Andheri West": 0, ...},
#                          "Status": {"Ready to Move": 0, ...},
#                          "property_type": {"Apartment": 0, ...}}
#   }
#
# @st.cache_resource → files are read from disk only ONCE when the
# app first loads, then reused from memory on every interaction.
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    model_path    = os.path.join("models", "best_rf_model.pkl")
    encoders_path = os.path.join("models", "encoders.pkl")   # ← your exact filename

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(encoders_path, "rb") as f:
        saved = pickle.load(f)              # loads the outer dict

    # Unpack the two keys your notebook saved
    encoders         = saved["encoders"]          # LabelEncoder objects (used for .transform())
    train_categories = saved["train_categories"]  # plain dicts  (used to build dropdowns)

    return model, encoders, train_categories


# Try loading — show a clear error banner if Task 1 isn't done yet
try:
    model, encoders, train_categories = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError as e:
    artifacts_loaded = False
    missing_file_msg = str(e)


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: BUILD DROPDOWN OPTIONS FROM train_categories
#
# train_categories["city"] looks like: {"Delhi": 0, "Hisar": 1, "Mumbai": 2, "Pune": 3}
# .keys() gives us the human-readable strings for the selectbox options.
# No hardcoding needed — these come directly from your notebook!
#
# City coordinates — used to pre-fill latitude & longitude fields.
# ═══════════════════════════════════════════════════════════════════
CITY_COORDS = {
    "Mumbai": (19.0760, 72.8777),
    "Pune":   (18.5204, 73.8567),
    "Delhi":  (28.6139, 77.2090),
    "Hisar":  (29.1492, 75.7217),
}


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: PAGE HEADER
# ═══════════════════════════════════════════════════════════════════
st.title("🏠 UrbanNest Analytics")
st.subheader("Dynamic House Rent Prediction Engine")
st.markdown(
    "Fill in the property details below and click **Predict Rent** "
    "to get an instant ML-powered estimate."
)
st.divider()

# Show an error and stop rendering if model files are missing
if not artifacts_loaded:
    st.error(
        f"⚠️ Could not load model files. Make sure Task 1 is complete "
        f"and both `models/best_rf_model.pkl` and `models/encoders.pkl` exist.\n\n"
        f"**Details:** {missing_file_msg}"
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: INPUT WIDGETS
#
# We build every widget from the exact keys in train_categories so
# the options always match what the LabelEncoder knows.
# ═══════════════════════════════════════════════════════════════════
st.markdown("### 📍 Location Details")

col1, col2 = st.columns(2)

with col1:
    # ── city ──────────────────────────────────────────────────────
    # train_categories["city"].keys() → ["Delhi", "Hisar", "Mumbai", "Pune"]
    city_options = sorted(train_categories["city"].keys())

    city = st.selectbox(
        label="City",
        options=city_options,
        help="Select the city where the property is located."
    )

with col2:
    # ── location ──────────────────────────────────────────────────
    # All locations the encoder knows about
    location_options = sorted(train_categories["location"].keys())

    location = st.selectbox(
        label="Location / Neighbourhood",
        options=location_options,
        help="Select the neighbourhood of the property."
    )

# ── latitude & longitude ──────────────────────────────────────────
# Pre-filled with city centre coordinates; user can fine-tune
default_lat, default_lon = CITY_COORDS.get(city, (20.5937, 78.9629))  # India centre as fallback

col3, col4 = st.columns(2)

with col3:
    latitude = st.number_input(
        label="Latitude",
        value=float(default_lat),
        format="%.4f",
        help="Pre-filled with city centre. Adjust for the exact property location."
    )

with col4:
    longitude = st.number_input(
        label="Longitude",
        value=float(default_lon),
        format="%.4f",
    )

st.markdown("### 🏗️ Property Details")

col5, col6 = st.columns(2)

with col5:
    # ── property_type ─────────────────────────────────────────────
    property_type_options = sorted(train_categories["property_type"].keys())

    property_type = st.selectbox(
        label="Property Type",
        options=property_type_options,
    )

with col6:
    # ── Status ────────────────────────────────────────────────────
    status_options = sorted(train_categories["Status"].keys())

    status = st.selectbox(
        label="Status",
        options=status_options,
        help="Current construction status of the property."
    )

col7, col8 = st.columns(2)

with col7:
    # ── Size_ft² ──────────────────────────────────────────────────
    size = st.number_input(
        label="Size (ft²)",
        min_value=100,
        max_value=20000,
        value=1000,
        step=50,
        help="Total carpet / built-up area in square feet."
    )

with col8:
    # ── BHK ───────────────────────────────────────────────────────
    bhk = st.number_input(
        label="BHK",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Number of Bedrooms, Hall, and Kitchen."
    )

col9, col10 = st.columns(2)

with col9:
    # ── rooms_num ─────────────────────────────────────────────────
    rooms_num = st.number_input(
        label="Total Rooms",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        help="Total number of rooms (bedroom + hall + kitchen, etc.)"
    )

with col10:
    # ── numBathrooms ──────────────────────────────────────────────
    num_bathrooms = st.number_input(
        label="Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
    )

col11, col12 = st.columns(2)

with col11:
    # ── numBalconies ──────────────────────────────────────────────
    num_balconies = st.number_input(
        label="Balconies",
        min_value=0,
        max_value=10,
        value=1,
        step=1,
    )

with col12:
    # ── verification_days ─────────────────────────────────────────
    verification_days = st.number_input(
        label="Verification Days",
        min_value=0,
        max_value=3650,
        value=30,
        step=1,
        help="Number of days since the listing was last verified."
    )

st.markdown("### 💰 Financial Details")

col13, col14 = st.columns(2)

with col13:
    # ── SecurityDeposit ───────────────────────────────────────────
    security_deposit = st.number_input(
        label="Security Deposit (₹)",
        min_value=0,
        max_value=10_000_000,
        value=50_000,
        step=5000,
        help="Refundable security deposit amount in rupees."
    )

with col14:
    # ── isNegotiable ──────────────────────────────────────────────
    # Stored as 0/1 in dataset; we collect Yes/No and convert below
    is_negotiable_str = st.selectbox(
        label="Is Rent Negotiable?",
        options=["Yes", "No"],
    )

st.divider()


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: PREDICT BUTTON & LOGIC
# ═══════════════════════════════════════════════════════════════════
if st.button("🔮 Predict Rent", use_container_width=True, type="primary"):

    # ── Step A: Encode categoricals using the saved LabelEncoders ──
    #
    # encoders["city"] is a fitted LabelEncoder object.
    # .transform(["Mumbai"]) returns array([2]); [0] extracts the int.
    #
    # Because your notebook fitted on combined train+test values,
    # every option in the dropdowns is guaranteed to be known — so
    # ValueError should never occur. We still catch it just in case.
    # ──────────────────────────────────────────────────────────────
    try:
        city_enc     = encoders["city"].transform([city])[0]
        location_enc = encoders["location"].transform([location])[0]
        status_enc   = encoders["Status"].transform([status])[0]
        prop_enc     = encoders["property_type"].transform([property_type])[0]

    except ValueError as err:
        st.error(
            f"❌ **Encoding failed:** {err}\n\n"
            "A selected value wasn't found in the encoder. "
            "This shouldn't happen if encoders.pkl was built from your full dataset."
        )
        st.stop()

    except KeyError as err:
        st.error(
            f"❌ **Encoder key missing:** {err}\n\n"
            "The encoders.pkl dict is missing an expected key. "
            "Expected keys: `city`, `location`, `Status`, `property_type`."
        )
        st.stop()

    # ── Step B: Convert Yes/No → 1/0 ──────────────────────────────
    is_negotiable = 1 if is_negotiable_str == "Yes" else 0

    # ── Step C: Assemble the feature vector ───────────────────────
    #
    # ORDER MUST EXACTLY match X_train column order from your notebook:
    # ['location', 'city', 'latitude', 'longitude', 'numBathrooms',
    #  'numBalconies', 'isNegotiable', 'SecurityDeposit', 'Status',
    #  'Size_ft²', 'BHK', 'rooms_num', 'property_type', 'verification_days']
    #
    # model.predict() needs a 2D array → shape (1, 14)
    # That's why we use double brackets: np.array([[ ... ]])
    # ──────────────────────────────────────────────────────────────
    features = np.array([[
        location_enc,       # location          (label-encoded int)
        city_enc,           # city              (label-encoded int)
        latitude,           # latitude          (raw float)
        longitude,          # longitude         (raw float)
        num_bathrooms,      # numBathrooms      (raw int)
        num_balconies,      # numBalconies      (raw int)
        is_negotiable,      # isNegotiable      (0 or 1)
        security_deposit,   # SecurityDeposit   (raw int)
        status_enc,         # Status            (label-encoded int)
        size,               # Size_ft²          (raw float)
        bhk,                # BHK               (raw int)
        rooms_num,          # rooms_num         (raw int)
        prop_enc,           # property_type     (label-encoded int)
        verification_days,  # verification_days (raw int)
    ]])
    # Final shape: (1, 14) ✓

    # ── Step D: Get prediction ─────────────────────────────────────
    predicted_rent = model.predict(features)[0]
    # .predict() returns array([35000.0]); [0] gives the scalar

    # ── Step E: Show result ────────────────────────────────────────
    st.success(f"### 💰 Estimated Monthly Rent: ₹ {predicted_rent:,.0f}")
    st.caption("Prediction by Random Forest model · UrbanNest Analytics")

    # Debug table — shows raw vs encoded values sent to the model
    # Very useful for your viva demo to explain what's happening!
    with st.expander("🔍 View encoded inputs sent to the model"):
        debug_df = pd.DataFrame({
            "Feature": [
                "location", "city", "latitude", "longitude",
                "numBathrooms", "numBalconies", "isNegotiable",
                "SecurityDeposit", "Status", "Size_ft²",
                "BHK", "rooms_num", "property_type", "verification_days"
            ],
            "What you selected": [
                location, city, latitude, longitude,
                num_bathrooms, num_balconies, is_negotiable_str,
                security_deposit, status, size,
                bhk, rooms_num, property_type, verification_days
            ],
            "Encoded value (sent to model)": [
                location_enc, city_enc, latitude, longitude,
                num_bathrooms, num_balconies, is_negotiable,
                security_deposit, status_enc, size,
                bhk, rooms_num, prop_enc, verification_days
            ],
        })

        debug_df = debug_df.astype(str) # Convert all to string 
        st.dataframe(debug_df,width="stretch", hide_index=True)

    


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: FOOTER
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption("UrbanNest Analytics · Powered by Random Forest · Built with Streamlit 🚀")
