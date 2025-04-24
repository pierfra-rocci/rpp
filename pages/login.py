import streamlit as st
import requests
import json

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="RAPAS Photometry Factory", page_icon="🔒", layout="wide")

# Use session state to track login status
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

backend_url = "http://localhost:5000"

if not st.session_state.logged_in:
    st.title("🔒 _RAPAS Photometry Factory_")
    menu = ["Login", "Register", "Recover Password"]
    choice = st.sidebar.selectbox("Menu", menu)

    st.sidebar.markdown("## User Credentials")
    username = st.sidebar.text_input("Username", 
                                     value="admin")
    password = st.sidebar.text_input("Password", 
                                     value="admin", type="password")

    if choice == "Login":
        if st.sidebar.button("Login"):
            if username and password:
                response = requests.post(f"{backend_url}/login", data={"username": username, "password": password})
                if response.status_code == 200:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error(response.text)
            else:
                st.warning("Please enter both username and password.")
    elif choice == "Register":
        if st.sidebar.button("Register"):
            if username and password:
                response = requests.post(f"{backend_url}/register", data={"username": username, "password": password})
                if response.status_code == 201:
                    st.success(response.text)
                else:
                    st.error(response.text)
            else:
                st.warning("Please enter both username and password.")
    elif choice == "Recover Password":
        st.sidebar.markdown("## Recover Password")
        new_password = st.sidebar.text_input("New Password", type="password")
        if st.sidebar.button("Reset Password"):
            if username and new_password:
                response = requests.post(f"{backend_url}/recover", data={"username": username, "new_password": new_password})
                if response.status_code == 200:
                    st.success(response.text)
                else:
                    st.error(response.text)
            else:
                st.warning("Please enter your username and new password.")
else:
    st.success(f"Welcome, {st.session_state.username}! Redirecting to the main app...")
    # Load config from backend and update session state before switching page
    try:
        backend_url = "http://localhost:5000/get_config"
        resp = requests.get(backend_url, params={"username": st.session_state.username})
        if resp.status_code == 200 and resp.text and resp.text != '{}':
            config = json.loads(resp.text)
            # Set session state for each parameter group if present
            if "analysis_parameters" in config:
                st.session_state["analysis_parameters"] = config["analysis_parameters"]
                # Also update individual keys for direct use in the app
                ap = config["analysis_parameters"]
                st.session_state["seeing"] = ap.get("seeing")
                st.session_state["threshold_sigma"] = ap.get("threshold_sigma")
                st.session_state["detection_mask"] = ap.get("detection_mask")
            if "gaia_parameters" in config:
                gaia = config["gaia_parameters"]
                st.session_state["gaia_band"] = gaia.get("gaia_band")
                st.session_state["gaia_min_mag"] = gaia.get("gaia_min_mag")
                st.session_state["gaia_max_mag"] = gaia.get("gaia_max_mag")
            if "observatory_parameters" in config:
                st.session_state["observatory_data"] = config["observatory_parameters"]
            if "astro_colibri_api_key" in config:
                st.session_state["colibri_api_key"] = config["astro_colibri_api_key"]
    except Exception as e:
        st.warning(f"Could not load user config: {e}")
    st.switch_page("pages/app.py")