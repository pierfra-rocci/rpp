from streamlit.web import cli

# This import path depends on your Streamlit version
if __name__ == '__main__':
    cli._main_run_clExplicit('rapas_app.py', args=['run'])