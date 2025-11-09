import streamlit as st
import requests
import json

import warnings

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="RAPAS Photometry Pipeline", page_icon="🔒", layout="wide"
)

# Use session state to track login status - Keep these basic initializations
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "recovery_step" not in st.session_state:  # Keep for password recovery logic
    st.session_state.recovery_step = 0

backend_url = "http://localhost:5000"

if not st.session_state.logged_in:
    st.title("🔒 _RAPAS Photometry Pipeline_")
    st.sidebar.markdown("## User Credentials")

    username = st.sidebar.text_input(
        "Username",
        value="",
        placeholder="Enter your username",
        help="Your registered username",
    )
    password = st.sidebar.text_input(
        "Password",
        value="",
        type="password",
        placeholder="Enter your password",
        help="Your account password",
    )

    email = st.sidebar.text_input(
        "Email", value="", help="Required for registration and password recovery."
    )

    login_col, register_col = st.sidebar.columns([1,1], gap=None)
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
                st.rerun()
            else:
                st.error(response.text)
        else:
            st.warning("Please enter both username and password.")

    if register_clicked:
        if username and password and email:
            if len(password) < 8:
                st.warning("Password must be at least 8 characters long.")
            else:
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
    recovery_email = st.sidebar.text_input("Email", value="", key="recovery_email")
    if st.session_state.recovery_step == 0:
        if st.sidebar.button("Send Recovery Code"):
            if recovery_email:
                resp = requests.post(
                    f"{backend_url}/recover_request", data={"email": recovery_email}
                )
                if resp.status_code == 200:
                    st.session_state.recovery_step = 1
                    st.success("Recovery code sent to your email.")
                else:
                    st.error(resp.text)
            else:
                st.warning("Please enter your email.")
    elif st.session_state.recovery_step == 1:
        code = st.sidebar.text_input("Enter Recovery Code", key="recovery_code")
        new_password = st.sidebar.text_input(
            "New Password", type="password", key="recovery_new_pw"
        )
        if st.sidebar.button("Reset Password"):
            if recovery_email and code and new_password:
                resp = requests.post(
                    f"{backend_url}/recover_confirm",
                    data={
                        "email": recovery_email,
                        "code": code,
                        "new_password": new_password,
                    },
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
    # Load config from backend and store the raw dictionaries in session state
    try:
        config_url = f"{backend_url}/get_config"  # Corrected URL path
        resp = requests.get(config_url, params={"username": st.session_state.username})
        if resp.status_code == 200 and resp.text and resp.text != "{}":
            config = json.loads(resp.text)

            if "analysis_parameters" in config:
                st.session_state["analysis_parameters"] = config["analysis_parameters"]

            if "observatory_data" in config:  # Use 'observatory_data' key
                st.session_state["observatory_data"] = config["observatory_data"]

            if "colibri_api_key" in config:  # Use 'colibri_api_key' key
                st.session_state["colibri_api_key"] = config["colibri_api_key"]
            st.info("User configuration loaded.")
        else:
            st.info("No user configuration found or error loading. Using defaults.")

            if "analysis_parameters" not in st.session_state:
                st.session_state["analysis_parameters"] = {}
            if "observatory_data" not in st.session_state:
                st.session_state["observatory_data"] = {}

    except Exception as e:
        st.warning(f"Could not load user config: {e}")

        if "analysis_parameters" not in st.session_state:
            st.session_state["analysis_parameters"] = {}
        if "observatory_data" not in st.session_state:
            st.session_state["observatory_data"] = {}

    st.switch_page("pages/app.py")
