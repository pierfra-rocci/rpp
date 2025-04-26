from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_data_files

datas = copy_metadata("streamlit")
datas += collect_data_files("streamlit")
