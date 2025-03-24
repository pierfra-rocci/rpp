import streamlit.web.cli as stcli
import os
import sys

def resolve_path(path):
    return os.path.abspath(os.path.join(os.getcwd(), path))

if __name__ == '__main__':
    sys.argv = ['streamlit', 'run', 
                resolve_path('rapas_app.py')]
    
    sys.exit(stcli.main())