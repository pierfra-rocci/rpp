from cx_Freeze import setup, Executable
import sys

if getattr(sys, "frozen", False):
    import importlib.metadata

    importlib.metadata.distributions = lambda **kwargs: []

import streamlit.web.cli as stcli
import streamlit.runtime.scriptrunner.magic_funcs
import astropy.constants.codata2018
import astropy.constants.iau2015

from flask import Flask, request
import sqlite3
from flask_cors import CORS

import subprocess
import os
import zipfile
from datetime import datetime, timedelta
import base64
import json
import requests
import tempfile
from urllib.parse import quote

import astroscrappy
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy.coordinates import get_sun
from typing import Union, Any, Optional, Dict, Tuple

from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

from astropy.modeling import models, fitting
import streamlit as st
import streamlit.components.v1 as components

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip, SigmaClip

import astropy.units as u
from astroquery.gaia import Gaia
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, SExtractorBackground
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from astropy.visualization import ZScaleInterval, ImageNormalize, simple_norm
from io import StringIO, BytesIO
from astropy.wcs import WCS

from photutils.psf import EPSFBuilder, extract_stars, IterativePSFPhotometry
from astropy.nddata import NDData

from stdpipe import photometry, astrometry, catalogs, pipeline

from __version__ import version

import warnings

warnings.filterwarnings("ignore")


def resolve_path(path):
    return os.path.abspath(os.path.join(os.getcwd(), path))

# Include additional files and folders (e.g., doc, rpp_results, etc.)
include_files = [
    "requirements.txt",
    "pages/app.py",
    "pages/login.py",
    "backend.py",
    "run_frontend.py",
    "__version__.py",
    "tools.py",
    "plate_solve.ps1",
    "plate_solve.sh",
]

# Build options
build_exe_options = {
    "packages": [
        "os", "sys", "streamlit", "flask", "sqlite3", "requests", "numpy", "pandas",
        "astropy", "photutils", "matplotlib", "astroquery", "stdpipe", "astroscrappy"
    ],
    "includes": [],
    "include_files": include_files,
    "excludes": [],
}

# Main entry point for the frontend (Streamlit)
executables = [
    Executable("run_frontend.py", base=None, target_name="rpp.exe"),
    # Optionally, add backend as a separate executable:
    # Executable("backend.py", base=None, target_name="pfr_backend.exe"),
]

setup(
    name="RAPAS Photometry Pipeline",
    version="0.1.0",
    description="Standalone RAPAS Photometry Pipeline (Streamlit + Flask)",
    options={"build_exe": build_exe_options},
    executables=executables,
)