# Show the contents of app log files on server

import streamlit as st


def show_file(filepath):
    try:
        # Read the file
        with open(filepath, "r") as file:
            content = file.read()

        # Display the content
        st.text_area("File Content", content, height=300)
    except FileNotFoundError:
        st.error("File not found. Check the path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")


st.set_page_config(
    page_title="RPP Server Logs",
    page_icon=":sparkles:",
    layout="centered",
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.warning("You must log in to access this page.")
    try:
        st.switch_page("pages/login.py")
    except Exception:
        st.warning(
            "Automatic redirect is not available in this launch mode. "
            "Open the login page directly or start the app with frontend.py."
        )
    st.stop()

# Add a button
if st.button("Show Backend Log"):
    show_file('backend.log')

# Add a button
if st.button("Show Frontend Log"):
    show_file('frontend.log')