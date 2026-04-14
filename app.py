
import streamlit as st # used for the  UI
import pickle # used for loading the saved model and encoders
import numpy as np
import pandas as pd
import os

# SECTION 1: setting up PAGE CONFIG

st.set_page_config(
    page_title="UrbanNest — Rent Predictor", page_icon="🏠", layout="centered",
)


# SECTION 2: LOAD MODEL, CATEGORIES & ENCODERS
# We need Model(to predict), Encoders(text to numbers conversion), and Categories (to know what options to show in dropdowns).


# @st.cache_resource = In Streamlit, it's a specialized decorator used to handle "expensive" objects that should only be created or loaded once.
# it happens because python script runs from top to bottom on every user interaction, so without caching, the model would be loaded from disk every time a user changes an input,which would be very slow, hence caching is used.
# hence, @st.cache_resource is perfect for loading machine learning models and encoders, which are typically large and time-consuming to load. By caching them, we ensure that they are loaded only once when the app starts, and subsequent interactions will use the cached versions, resulting in much faster response times.
# if we ever need to update the model or encoders, we can simply change the files on disk and restart the Streamlit app or clear the cache to load the new versions.


@st.cache_resource
def load_artifacts():
    model_path = os.path.join("models", "best_rf_model.pkl")
    encoders_path = os.path.join("models", "encoders.pkl")

    with open(model_path, "rb") as f: #rb is read binary mode, used for binary files (.pkl) 
        model = pickle.load(f) # pickle convert binary file back to python object (RandomForestRegressor in this case)

    with open(encoders_path, "rb") as f:
        saved = pickle.load(f) 

    # Unpack the two keys from "saved"
    encoders = saved["encoders"] # LabelEncoder (used for .transform() later)
    train_categories = saved["train_categories"]  # category lists (used for dropdown options)

    return model, encoders, train_categories


# Try Loading the artifacts. 
try:
    model, encoders, train_categories = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError as e:
    artifacts_loaded = False
    missing_file_msg = str(e)


# SECTION 3: BUILD DROPDOWN OPTIONS FROM train_categories

# train_categories["city"] looks like: {"Delhi": 0, "Hisar": 1, "Mumbai": 2, "Pune": 3}
# .keys() gives us the human-readable strings/labels (not numbers) for the selectbox options.
# City coordinates: used to pre-fill latitude & longitude fields.
# no hardcoding of dropdown options anywhere in the code. If we add a new city to the dataset and rebuild encoders.pkl, it will automatically appear in the app.

CITY_COORDS = {
    "Mumbai": (19.0760, 72.8777),
    "Pune":   (18.5204, 73.8567),
    "Delhi":  (28.6139, 77.2090),
    "Hisar":  (29.1492, 75.7217),
}

# SECTION 4: PAGE HEADER

st.title("🏠 UrbanNest Analytics")
st.subheader("Dynamic House Rent Prediction Engine")
st.markdown("Fill in the property details below and click **Predict Rent** to get an instant ML-powered estimate.")
st.divider()

# Show an error and stop if model files are missing
if not artifacts_loaded:
    st.error(
        f"Could not load model files. Make sure `models/best_rf_model.pkl` and `models/encoders.pkl` exist.\n"
        f"**Details:** {missing_file_msg}"
    )
    st.stop()

# SECTION 5: INPUTS 

st.markdown("### 📍 Location Details")

col1, col2 = st.columns(2) # allows side by side UI, because normally streamlit stacks it vertically. 
with col1:
    # CITY
    city_options = sorted(train_categories["city"].keys())

    city = st.selectbox(
        label="City",
        options=city_options,
        help="Select the city where the property is located."
    )

with col2:
    # LOCATION
    location_options = sorted(train_categories["location"].keys())

    location = st.selectbox(
        label="Location / Neighbourhood",
        options=location_options,
        help="Select the neighbourhood/area of the property."
    )

# latitude & longitude 
# Pre-filled with city centre coordinates, but can be adjusted.

default_lat, default_lon = CITY_COORDS.get(city, (20.5937, 78.9629))  # India centre as fallback, if city not found in CITY_COORDS. 
col3, col4 = st.columns(2)

with col3:
    latitude = st.number_input(
        label="Latitude",
        value=float(default_lat), # display default value, but user can change it. 
        format="%.4f",
        help="Pre-filled with city centre. Adjust for the exact property latitude."
    )

with col4:
    longitude = st.number_input(
        label="Longitude",
        value=float(default_lon),
        format="%.4f",
        help="Pre-filled with city centre. Adjust for the exact property longitude."
    )

st.markdown("### 🏗️ Property Details")

col5, col6 = st.columns(2)

with col5:
    # PROPERTY TYPE
    property_type_options = sorted(train_categories["property_type"].keys())

    property_type = st.selectbox(
        label="Property Type",
        options=property_type_options,
        help="Type of the property (e.g., Apartment, Independent House, etc.)."
    )

with col6:
    # STATUS
    status_options = sorted(train_categories["Status"].keys())

    status = st.selectbox(
        label="Status",
        options=status_options,
        help="State current construction status of the property."
    )

col7, col8 = st.columns(2)

with col7:
    # SIZE (ft²)
    size = st.number_input(
        label="Size (ft²)",
        min_value=100,
        max_value=20000,
        value=1000,
        step=50,
        help="Total carpet / built-up area in square feet."
    )

with col8:
    # BHK
    bhk = st.number_input(
        label="BHK",
        min_value=1,
        max_value=6,
        value=2,
        step=1,
        help="Number of Bedrooms, Hall, and Kitchen."
    )

col9, col10 = st.columns(2)

with col9:
    # NUMBER OF ROOMS
    rooms_num = st.number_input(
        label="Total Rooms",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        help="Total number of rooms (bedroom + hall + kitchen, etc.)"
    )

with col10:
    # NUMBER OF BATHROOMS
    num_bathrooms = st.number_input(
        label="Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
    )

col11, col12 = st.columns(2)

with col11:
    # NUMBER OF BALCONIES
    num_balconies = st.number_input(
        label="Balconies",
        min_value=0,
        max_value=10,
        value=1,
        step=1,
    )

with col12:
    # VERIFICATION DAYS
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
    # SECURITY DEPOSIT
    security_deposit = st.number_input(
        label="Security Deposit (₹)",
        min_value=0,
        max_value=10_000_000,
        value=50_000,
        step=5000,
        help="Refundable security deposit amount in rupees."
    )

with col14:
    # HOW MUCH RENT IS NEGOTIABLE
    # Stored as 0/1 in dataset; we collect Yes/No and convert below
    is_negotiable_str = st.selectbox(
        label="Is Rent Negotiable?",
        options=["Yes", "No"],
    )

st.divider()

# SECTION 6: PREDICT BUTTON & LOGIC

if st.button("🔮 Predict Rent", use_container_width=True, type="primary"): # primary means important action. Only color of button will change, not the functionality. 

    # Encode categoricals using the saved LabelEncoders
    
    try:
        city_enc = encoders["city"].transform([city])[0]
        location_enc = encoders["location"].transform([location])[0]
        status_enc = encoders["Status"].transform([status])[0]
        prop_enc = encoders["property_type"].transform([property_type])[0]

    except ValueError as err:
        st.error(
            f"❌ **Encoding failed:** {err}\n\n"
            "A selected value wasn't found in the encoder. "
        )
        st.stop()

    except KeyError as err:
        st.error(
            f"❌ **Encoder key missing:** {err}\n\n"
            "The encoders.pkl dict is missing an expected key (like city/location/Status/property_type). "
        )
        st.stop()

    # Convert Yes/No to 1/0 for non_ negotiable field, since model was trained on 0/1 values.
    is_negotiable = 1 if is_negotiable_str == "Yes" else 0

    # Creating the feature vector
    # model.predict() needs a 2D array, hence the double brackets is used. Usually, we give features for multiple samples, hence the shape would be (n_samples, n_features).
    features = np.array([[ location_enc, city_enc, latitude, longitude, num_bathrooms, num_balconies, is_negotiable, security_deposit, status_enc,size, bhk, rooms_num, prop_enc, verification_days]]) # Final shape: (1, 14) 

# Creating the feature vector as a DataFrame (preserves feature names)
    input_dict = {
        "location":          location_enc,
        "city": city_enc,
        "latitude":          float(latitude),
        "longitude":         float(longitude),
        "numBathrooms":      int(num_bathrooms),
        "numBalconies":      int(num_balconies),
        "isNegotiable":      int(is_negotiable),
        "SecurityDeposit":   int(security_deposit),
        "Status":            status_enc,
        "Size_ft²":          int(size),
        "BHK":               int(bhk),
        "rooms_num":         int(rooms_num),
        "property_type":     prop_enc,
        "verification_days": int(verification_days),
    }

    features = pd.DataFrame([input_dict])  # shape (1, 14) with column names

    # Get prediction
    predicted_rent = model.predict(features)[0]
    # .predict() returns numpy  array([35000.0]); [0] gives the scalar value, since only 1 output is there. 
    
    # Show result
    st.success(f"### 💰 Estimated Monthly Rent: ₹ {predicted_rent:,.0f}")
    st.caption("Prediction by Random Forest model · UrbanNest Analytics")

    # Debug table: shows raw vs encoded values sent to the model
    with st.expander("🔍 View encoded inputs sent to the model"):
        debug_df = pd.DataFrame({
            "Feature": [ "location", "city", "latitude", "longitude", "numBathrooms", "numBalconies", "isNegotiable","SecurityDeposit", "Status", "Size_ft²", "BHK", "rooms_num", "property_type", "verification_days"],
            "What you selected": [location, city, latitude, longitude, num_bathrooms, num_balconies, is_negotiable_str, security_deposit, status, size, bhk, rooms_num, property_type, verification_days],
            "Encoded value": [location_enc, city_enc, latitude, longitude, num_bathrooms, num_balconies, is_negotiable, security_deposit, status_enc, size,bhk, rooms_num, prop_enc, verification_days], # encoded values are sent to model. 
        })
    
        debug_df = debug_df.astype(str) # Convert all to string 
        st.dataframe(debug_df,width="stretch", hide_index=True)

# SECTION 7: FOOTER

st.markdown("---")
st.caption("UrbanNest Analytics · Powered by Random Forest · Built with Streamlit 🚀")
