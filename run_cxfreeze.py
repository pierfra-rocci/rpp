import sys

if getattr(sys, "frozen", False):
    import importlib.metadata
    importlib.metadata.distributions = lambda **kwargs: []

import streamlit.web.cli as stcli
import streamlit.runtime.scriptrunner.magic_funcs
import astropy.constants.codata2018
import astropy.constants.iau2015

import subprocess
import os
import zipfile
from datetime import datetime, timedelta
import base64
import json
import requests
import tempfile
from urllib.parse import quote

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


if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("pfr_app.py"),
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())
