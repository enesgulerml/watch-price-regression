import streamlit as st
import requests
import pandas as pd
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Watch Price Predictor",
    page_icon="⏱️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- API Endpoint ---
# This is the address of our v4 Dockerized FastAPI
# CORRECTED: Pointing to the non-zombie port 8001
API_URL = "http://127.0.0.1:8001/predict"

# --- Page Title ---
st.title("⏱️ End-to-End Watch Price Predictor (v5)")
st.markdown(
    "This Streamlit dashboard is the 'Showroom' (v5) that interacts with the 'Store's API' (v3/v4) running in a Docker container.")

# --- Sidebar Inputs ---
st.sidebar.header("Input Watch Features")

# We use the "example" from our app/schema.py as the defaults
example_data = {
    "Case_Diameter": 40.5,
    "Water_Resistance": 10,
    "Warranty_Years": 2,
    "Weight_g": 85.0,
    "Brand": "Seiko",
    "Gender": "Male",
    "Case_Color": "Silver",
    "Glass_Shape": "Flat",
    "Origin": "Japan",
    "Case_Material": "Steel",
    "Additional_Feature": "Luminous",
    "Strap_Color": "Black",
    "Strap_Material": "Leather",
    "Mechanism": "Automatic",
    "Glass_Type": "Sapphire",
    "Dial_Color": "Blue"
}

# --- Create Input Fields ---
# (Input fields code remains the same as before...)

st.sidebar.subheader("Numerical Features")
p_case_diameter = st.sidebar.number_input("Case Diameter (mm)", min_value=20.0, max_value=70.0,
                                          value=example_data["Case_Diameter"], step=0.5)
p_water_resistance = st.sidebar.number_input("Water Resistance (ATM)", min_value=0, max_value=200,
                                             value=example_data["Water_Resistance"], step=1)
p_warranty_years = st.sidebar.slider("Warranty (Years)", min_value=0, max_value=5, value=example_data["Warranty_Years"],
                                     step=1)
p_weight_g = st.sidebar.number_input("Weight (g)", min_value=20.0, max_value=500.0, value=example_data["Weight_g"],
                                     step=1.0)

st.sidebar.subheader("Categorical Features")
p_gender = st.sidebar.selectbox("Gender", ('Male', 'Female', 'Unisex'), index=0)
p_mechanism = st.sidebar.selectbox("Mechanism", ('Automatic', 'Quartz', 'Manual', 'Unknown'), index=0)
p_glass_shape = st.sidebar.selectbox("Glass Shape", ('Flat', 'Domed', 'Curved'), index=0)
p_glass_type = st.sidebar.selectbox("Glass Type", ('Sapphire', 'Mineral', 'Hardlex', 'Acrylic'), index=0)

p_brand = st.sidebar.text_input("Brand", value=example_data["Brand"])
p_case_color = st.sidebar.text_input("Case Color", value=example_data["Case_Color"])
p_origin = st.sidebar.text_input("Origin", value=example_data["Origin"])
p_case_material = st.sidebar.text_input("Case Material", value=example_data["Case_Material"])
p_additional_feature = st.sidebar.text_input("Additional Feature (or 'No')", value=example_data["Additional_Feature"])
p_strap_color = st.sidebar.text_input("Strap Color", value=example_data["Strap_Color"])
p_strap_material = st.sidebar.text_input("Strap Material", value=example_data["Strap_Material"])
p_dial_color = st.sidebar.text_input("Dial Color", value=example_data["Dial_Color"])

# --- Prediction Logic ---
if st.sidebar.button("Predict Price", type="primary"):

    # 1. Collect all inputs into the JSON payload (must match app/schema.py)
    payload = {
        "Case_Diameter": p_case_diameter,
        "Water_Resistance": p_water_resistance,
        "Warranty_Years": p_warranty_years,
        "Weight_g": p_weight_g,
        "Brand": p_brand,
        "Gender": p_gender,
        "Case_Color": p_case_color,
        "Glass_Shape": p_glass_shape,
        "Origin": p_origin,
        "Case_Material": p_case_material,
        "Additional_Feature": p_additional_feature,
        "Strap_Color": p_strap_color,
        "Strap_Material": p_strap_material,
        "Mechanism": p_mechanism,
        "Glass_Type": p_glass_type,
        "Dial_Color": p_dial_color
    }

    # 2. Display the JSON payload (for debugging)
    st.subheader("API Request Payload (JSON)")
    st.json(payload)

    try:
        # 3. Send request to the FastAPI (running in Docker)
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # Raise exception for 4xx/5xx errors

        result = response.json()

        # 4. Display the result
        st.subheader("API Response (Prediction)")
        price = result.get("predicted_price_usd")

        st.success(f"**Predicted Watch Price: ${price:,.2f}**")

    except requests.exceptions.ConnectionError:
        st.error(f"**Connection Error:** Could not connect to the API at `{API_URL}`.")
        st.warning("Did you forget to run the v4 Docker container in Terminal 1? (See README.md)")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.json(response.json())