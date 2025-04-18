import streamlit as st
import xgboost
from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim
from housing_model import predict_price_by_zip
from crime_model import predict_crime_rate_by_zip
from infrastructure_model import predict_infa_rate_by_zip
from weather_model import predict_temp_by_zip

# === FIRST Streamlit command ===
st.set_page_config(page_title="Living Insights", layout="wide")

# === SESSION INIT ===
if "map_zip" not in st.session_state:
    st.session_state["map_zip"] = ""

# === CUSTOM STYLING ===
st.markdown("""
<style>
    .stApp { background-color: #f3f4f6; color: black; }
    .metric-card {
        background-color: #f8f8f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# === SIDEBAR MAP ===
st.sidebar.markdown("# Living Insights")
st.sidebar.markdown("### 📍 Click Location on Map")
m = folium.Map(location=[32.7767, -96.7970], zoom_start=10)
st.sidebar.markdown("<div style='height:250px;'>", unsafe_allow_html=True)
map_data = st_folium(m, height=250, width=250, returned_objects=["last_clicked"], key="sidebar_map")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# === ZIP CODE HANDLING ===
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lng = map_data["last_clicked"]["lng"]
    geolocator = Nominatim(user_agent="zip_finder")
    location = geolocator.reverse((lat, lng), language="en")
    if location and "postcode" in location.raw["address"]:
        st.session_state["map_zip"] = location.raw["address"]["postcode"]

input_zip = st.text_input("Enter Zip Code", value=st.session_state["map_zip"], placeholder="e.g. 75080")
if input_zip != st.session_state["map_zip"]:
    st.session_state["map_zip"] = input_zip

zip_code = st.session_state["map_zip"]

# === MODEL PREDICTION ===
housing_pred = None
crime_pred   = None
infa_pred = None
weather_pred = None

if zip_code and zip_code.isdigit():
    housing_pred = predict_price_by_zip(zip_code)
    if housing_pred is None:
        st.warning(f"No housing data available for ZIP code {zip_code}.")
    crime_pred = predict_crime_rate_by_zip(zip_code)
    if crime_pred is None:
        st.warning(f"No crime data available for ZIP code {zip_code}.")
    infa_pred = predict_infa_rate_by_zip(zip_code)
    if infa_pred is None:
        st.warning(f"No infrastructure data available for ZIP code {zip_code}.")
    weather_pred = predict_temp_by_zip(zip_code)
    if weather_pred is None:
        st.warning(f"No temperature data available for ZIP code {zip_code}.")
    

# === DASHBOARD UI ===
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Housing Prices</h3>
        <p>ZIP: {zip_code if zip_code else 'Not provided'}</p>
        <p><b>{f"${housing_pred:,.2f}" if housing_pred is not None else 'No prediction available'}</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <h3>Weather Temperatures</h3>
        <p>ZIP: {zip_code if zip_code else 'Not Provided'}</p>
        <p><b>{f"{weather_pred:,.2f} degrees fahrenheit on average" if weather_pred is not None else 'No prediction available'}</b></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Crime Rates</h3>
        <p>ZIP: {zip_code if zip_code else 'Not provided'}</p>
        <p><b>{f"{crime_pred:.2f} incidents per 1,000" if crime_pred is not None else 'No prediction available'}</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <h3>Infrastructure Quality</h3>
        <p>ZIP: {zip_code if zip_code else 'Not provided'}</p>
        <p><b>{f'{infa_pred:.3f} AVG EST traffic delay from accidents' if infa_pred is not None else 'No prediction available'}</b></p>
    </div>
    """, unsafe_allow_html=True)
