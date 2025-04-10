import streamlit.web.cli as stcli
import os
import sys

def resolve_path(path):
    return os.path.abspath(os.path.join(os.getcwd(), path))

if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("pfr_app.py"),
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())