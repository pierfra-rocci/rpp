import os
import json
import warnings

import requests  # type: ignore[import]
import streamlit as st

from pages.api_client import ApiClient, ApiError, detect_backend

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="RAPAS Photometry Pipeline - Login",
    page_icon=":sparkles:",
    layout="wide",
)

# Use session state to track login status - Keep these basic initializations
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "recovery_step" not in st.session_state:  # Keep for password recovery logic
    st.session_state.recovery_step = 0
if "api_mode" not in st.session_state:
    st.session_state.api_mode = False
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = os.getenv(
        "RPP_API_URL",
        "http://localhost:8000",
    )
if "legacy_backend_url" not in st.session_state:
    st.session_state.legacy_backend_url = os.getenv(
        "RPP_LEGACY_URL",
        "http://localhost:5000",
    )
if "backend_mode" not in st.session_state:
    st.session_state.backend_mode = "legacy"
if "api_credentials" not in st.session_state:
    st.session_state.api_credentials = None
if "backend_status_message" not in st.session_state:
    st.session_state.backend_status_message = "Determining backend..."
if "backend_initialized" not in st.session_state:
    st.session_state.backend_initialized = False

if not st.session_state.backend_initialized:
    backend_info = detect_backend()
    st.session_state.api_base_url = backend_info["api_base_url"]
    st.session_state.legacy_backend_url = backend_info["legacy_backend_url"]
    st.session_state.api_mode = backend_info["mode"] == "api"
    st.session_state.backend_mode = backend_info["mode"]
    st.session_state.backend_status_message = backend_info["message"]
    st.session_state.backend_initialized = True


def validate_password(password):
    """Validate password meets all requirements"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter."
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter."
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit."
    return True, ""


if not st.session_state.logged_in:
    st.title("**RAPAS Photometry Pipeline** \n *[UNDER DEVELOPMENT]*")
    st.markdown("")
    st.markdown(
        "_Report feedback and bugs to_ : "
        "[rpp_support](mailto:rpp_support@saf-astronomie.fr)"
    )

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
        "Email",
        value="",
        help="Required for registration and password recovery.",
    )

    st.sidebar.caption(st.session_state.backend_status_message)

    login_col, register_col = st.sidebar.columns([1, 1])

    with login_col:
        login_clicked = st.button("Login")
    with register_col:
        register_clicked = st.button("Sign Up")

    if login_clicked:
        if username and password:
            if st.session_state.api_mode:
                client = ApiClient(st.session_state.api_base_url)
                try:
                    message = client.login(
                        username=username,
                        password=password,
                    )
                except ApiError as exc:
                    st.error(f"API error: {exc.message}")
                else:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.backend_mode = "api"
                    st.session_state.api_credentials = {
                        "username": username,
                        "password": password,
                        "base_url": st.session_state.api_base_url,
                    }
                    st.session_state.backend_status_message = (
                        "Using API backend"
                    )
                    st.success(message)
                    st.rerun()
            else:
                legacy_url = st.session_state.legacy_backend_url
                try:
                    response = requests.post(
                        f"{legacy_url}/login",
                        data={"username": username, "password": password},
                    )
                    if response.status_code == 200:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.backend_mode = "legacy"
                        st.session_state.backend_status_message = (
                            "Using legacy backend"
                        )
                        st.rerun()
                    else:
                        st.error(response.text)
                except requests.exceptions.RequestException as e:
                    st.error(
                        f"Connection error: Could not reach the backend server. {e}"
                    )
        else:
            st.warning("Please enter both username and password.")

    if register_clicked:
        if username and password and email:
            # Validate password before sending to backend
            is_valid, error_message = validate_password(password)
            if not is_valid:
                st.warning(error_message)
            else:
                if st.session_state.api_mode:
                    client = ApiClient(st.session_state.api_base_url)
                    try:
                        message = client.register(
                            username=username,
                            password=password,
                            email=email,
                        )
                    except ApiError as exc:
                        st.error(f"API error: {exc.message}")
                    else:
                        st.success(message)
                else:
                    legacy_url = st.session_state.legacy_backend_url
                    try:
                        response = requests.post(
                            f"{legacy_url}/register",
                            data={
                                "username": username,
                                "password": password,
                                "email": email,
                            },
                        )
                        if response.status_code == 201:
                            st.success(response.text)
                        else:
                            st.error(response.text)
                    except requests.exceptions.RequestException as e:
                        st.error(
                            f"Connection error: Could not reach the backend server. {e}"
                        )
        else:
            st.warning("Please enter username, password, and email.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Password Recovery")
    recovery_email = st.sidebar.text_input(
        "Email",
        key="recovery_email",
    )

    if st.session_state.recovery_step == 0:
        if st.sidebar.button("Send Request"):
            if recovery_email:
                if st.session_state.backend_mode == "api":
                    client = ApiClient(st.session_state.api_base_url)
                    try:
                        message = client.request_password_recovery(recovery_email)
                    except ApiError as exc:
                        st.error(f"API error: {exc.message}")
                    else:
                        st.session_state.recovery_step = 1
                        st.success(message)
                else:
                    try:
                        resp = requests.post(
                            (f"{st.session_state.legacy_backend_url}/recover_request"),
                            data={"email": recovery_email},
                        )
                        if resp.status_code == 200:
                            st.session_state.recovery_step = 1
                            st.success("Recovery code sent to your email.")
                        else:
                            st.error(resp.text)
                    except requests.exceptions.RequestException as e:
                        st.error(f"Connection error: Backend unreachable. {e}")
            else:
                st.warning("Please enter your email.")

    elif st.session_state.recovery_step == 1:
        code = st.sidebar.text_input(
            "Enter Recovery Code",
            key="recovery_code",
        )
        new_password = st.sidebar.text_input(
            "New Password",
            type="password",
            key="recovery_new_pw",
        )
        if st.sidebar.button("Reset Password"):
            if recovery_email and code and new_password:
                is_valid, error_message = validate_password(new_password)
                if not is_valid:
                    st.warning(error_message)
                else:
                    if st.session_state.backend_mode == "api":
                        client = ApiClient(st.session_state.api_base_url)
                        try:
                            message = client.confirm_password_recovery(
                                email=recovery_email,
                                code=code,
                                new_password=new_password,
                            )
                        except ApiError as exc:
                            st.error(f"API error: {exc.message}")
                        else:
                            st.success(message)
                            st.session_state.recovery_step = 0
                    else:
                        try:
                            resp = requests.post(
                                f"{st.session_state.legacy_backend_url}"
                                "/recover_confirm",
                                data={
                                    "email": recovery_email,
                                    "code": code,
                                    "new_password": new_password,
                                },
                            )
                            if resp.status_code == 200:
                                st.success("Password updated. You can now log in.")
                                st.session_state.recovery_step = 0
                            else:
                                st.error(resp.text)
                        except requests.exceptions.RequestException as e:
                            st.error(f"Connection error: Backend unreachable. {e}")
            else:
                st.warning("Please enter all fields.")

        if st.sidebar.button("Cancel Recovery"):
            st.session_state.recovery_step = 0
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**License:** MIT")
    st.sidebar.markdown(
        "**GDPR:** By using this application, you agree to the processing of your data under GDPR."
    )
else:
    st.success(f"Welcome, {st.session_state.username}! Redirecting to the app...")
    st.sidebar.caption(st.session_state.backend_status_message)
    # Load config from backend and store the raw dictionaries in session state
    try:
        if st.session_state.backend_mode == "api":
            creds = st.session_state.get("api_credentials")
            if creds:
                client = ApiClient(creds.get("base_url"))
                try:
                    config = client.get_config(
                        username=creds["username"],
                        password=creds["password"],
                    )
                except ApiError as exc:
                    st.warning(f"Could not load user config: {exc.message}")
                    config = None
            else:
                config = None
        else:
            config_url = f"{st.session_state.legacy_backend_url}/get_config"
            resp = requests.get(
                config_url,
                params={"username": st.session_state.username},
            )
            if resp.status_code == 200 and resp.text and resp.text != "{}":
                config = json.loads(resp.text)
            else:
                config = None

        if config:
            if "analysis_parameters" in config:
                st.session_state["analysis_parameters"] = config["analysis_parameters"]

            if "observatory_data" in config:
                st.session_state["observatory_data"] = config["observatory_data"]

            if "colibri_api_key" in config:
                st.session_state["colibri_api_key"] = config["colibri_api_key"]
            st.info("User configuration loaded.")
        else:
            st.info("No stored user configuration found. Using defaults.")

            if "analysis_parameters" not in st.session_state:
                st.session_state["analysis_parameters"] = {}
            if "observatory_data" not in st.session_state:
                st.session_state["observatory_data"] = {}

    except requests.exceptions.RequestException as e:
        st.warning(f"Could not load user config: Connection error - {e}")

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
