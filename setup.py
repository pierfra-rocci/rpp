from cx_freeze import setup, Executable

# Specify the modules to include
build_exe_options = {
    "packages": ["os", "streamlit"],
    "includes": ["streamlit.runtime.scriptrunner.magic_funcs"],
    "excludes": ["tkinter", "unittest", "email", "http", "html", "xml", "distutils"],
    "include_files": [],
}

setup(
    name='pfr_rapas',
    version='1.0',
    description='Photometry Factory for RAPAS',
    options={'build_exe': build_exe_options},
    executables=[Executable('pfr_app.py')],
)
