from cx_Freeze import setup, Executable

# Specify the modules to include
build_exe_options = {
    'packages': ['streamlit.runtime.scriptrunner'],
    'includes': ['streamlit.runtime.scriptrunner.magic_funcs'],
}

setup(
    name='PFR_RAPAS',
    version='1.0',
    description='Photometry Factory for RAPAS',
    options={'build_exe': build_exe_options},
    executables=[Executable('pfr_app.py')],
)
