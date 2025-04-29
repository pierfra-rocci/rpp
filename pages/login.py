import streamlit as st
import requests
import json

import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="RAPAS Photometry Pipeline", page_icon="🔒", layout="wide")

# Use session state to track login status
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

backend_url = "http://localhost:5000"

if not st.session_state.logged_in:
    st.title("🔒 _RAPAS Photometry Pipeline_")
    st.sidebar.markdown("## User Credentials")
    username = st.sidebar.text_input("Username", value="admin")
    password = st.sidebar.text_input("Password", value="admin", type="password")
    email = st.sidebar.text_input("Email", value="", help="Required for registration and password recovery.")

    login_col, register_col = st.sidebar.columns([1, 1])
    login_clicked = login_col.button("Login")
    register_clicked = register_col.button("Register")

    if login_clicked:
        if username and password:
            response = requests.post(
                f"{backend_url}/login",
                data={"username": username, "password": password},
            )
            if response.status_code == 200:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful! Redirecting...")
                st.rerun()
            else:
                st.error(response.text)
        else:
            st.warning("Please enter both username and password.")

    if register_clicked:
        if username and password and email:
            response = requests.post(
                f"{backend_url}/register",
                data={"username": username, "password": password, "email": email},
            )
            if response.status_code == 201:
                st.success(response.text)
            else:
                st.error(response.text)
        else:
            st.warning("Please enter username, password, and email.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Recover Password")
    recovery_email = st.sidebar.text_input("Recovery Email", value="", key="recovery_email")
    if "recovery_step" not in st.session_state:
        st.session_state.recovery_step = 0
    if st.session_state.recovery_step == 0:
        if st.sidebar.button("Send Recovery Code"):
            if recovery_email:
                resp = requests.post(f"{backend_url}/recover_request", data={"email": recovery_email})
                if resp.status_code == 200:
                    st.session_state.recovery_step = 1
                    st.success("Recovery code sent to your email.")
                else:
                    st.error(resp.text)
            else:
                st.warning("Please enter your email.")
    elif st.session_state.recovery_step == 1:
        code = st.sidebar.text_input("Enter Recovery Code", key="recovery_code")
        new_password = st.sidebar.text_input("New Password", type="password", key="recovery_new_pw")
        if st.sidebar.button("Reset Password"):
            if recovery_email and code and new_password:
                resp = requests.post(
                    f"{backend_url}/recover_confirm",
                    data={"email": recovery_email, "code": code, "new_password": new_password},
                )
                if resp.status_code == 200:
                    st.success("Password updated successfully. You can now log in.")
                    st.session_state.recovery_step = 0
                else:
                    st.error(resp.text)
            else:
                st.warning("Please enter all fields.")
        if st.sidebar.button("Cancel Recovery"):
            st.session_state.recovery_step = 0
else:
    st.success(f"Welcome, {st.session_state.username}! Redirecting to the main app...")
    # Load config from backend and update session state before switching page
    try:
        backend_url = "http://localhost:5000/get_config"
        resp = requests.get(backend_url, params={"username": st.session_state.username})
        if resp.status_code == 200 and resp.text and resp.text != "{}":
            config = json.loads(resp.text)
            # Restore all parameter groups
            if "analysis_parameters" in config:
                st.session_state["analysis_parameters"] = config["analysis_parameters"]
                # Also update individual keys for direct use in the app
                ap = config["analysis_parameters"]
                for key in ["seeing", "threshold_sigma", "detection_mask"]:
                    if key in ap:
                        st.session_state[key] = ap[key]
            if "gaia_parameters" in config:
                gaia = config["gaia_parameters"]
                st.session_state["filter_band"] = gaia.get("filter_band")
                st.session_state["filter_max_mag"] = gaia.get("filter_max_mag")
            if "observatory_parameters" in config:
                st.session_state["observatory_data"] = config["observatory_parameters"]
                # Also update individual keys for direct use
                obs = config["observatory_parameters"]
                st.session_state["observatory_name"] = obs.get("name", "")
                st.session_state["observatory_latitude"] = obs.get("latitude", 0.0)
                st.session_state["observatory_longitude"] = obs.get("longitude", 0.0)
                st.session_state["observatory_elevation"] = obs.get("elevation", 0.0)
            if "astro_colibri_api_key" in config:
                st.session_state["colibri_api_key"] = config["astro_colibri_api_key"]
    except Exception as e:
        st.warning(f"Could not load user config: {e}")
    st.switch_page("pages/app.py")
