import sys
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from flask import Flask

# Import existing dependencies
import os
import datetime
import base64
import json
import requests
import platform
import subprocess
from urllib.parse import quote
import numpy as np
from astropy.io import fits
import plotly.graph_objects as go
from io import StringIO, BytesIO

# Add astronomy imports
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time 
from astropy.coordinates import get_sun
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astropy.modeling import models, fitting
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip, SigmaClip
import astropy.units as u
from astroquery.gaia import Gaia
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, SExtractorBackground
import pandas as pd
from astropy.table import Table
from astropy.visualization import ZScaleInterval, ImageNormalize, simple_norm
from astropy.wcs import WCS
from photutils.psf import EPSFBuilder, extract_stars, IterativePSFPhotometry
from astropy.nddata import NDData

# Initialize Flask and Dash
server = Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Constants
APP_VERSION = "1.0.0"

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Photometry Factory for RAPAS</title>
        {%favicon%}
        {%css%}
        <style>
            .dash-plot > div {
                display: flex;
                justify-content: center;
                min-height: 400px;
            }
            .container {
                max-width: 1200px;
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            img {
                max-width: 100% !important;
                height: auto !important;
            }
            .sidebar {
                border-right: 1px solid #dee2e6;
                height: 100%;
            }
            .upload-box {
                border: 2px dashed #cccccc;
                border-radius: 5px;
                padding: 20px;
                text-align: center;
                margin-bottom: 10px;
            }
            .upload-box:hover {
                border-color: #666666;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Define standard figure sizes (in pixels for plotly)
FIGURE_SIZES = {
    "small": {"width": 600, "height": 500},
    "medium": {"width": 800, "height": 600},
    "large": {"width": 1000, "height": 800},
    "wide": {"width": 1200, "height": 600},
    "stars_grid": {"width": 1000, "height": 800},
}

# Add visualization configuration
PLOTLY_THEME = {
    'layout': {
        'paper_bgcolor': 'rgb(250, 250, 250)',
        'plot_bgcolor': 'rgb(250, 250, 250)',
        'font': {'family': 'Arial, sans-serif'},
        'margin': {'t': 50, 'b': 50, 'l': 50, 'r': 50}
    }
}

# Session storage for data (replace streamlit session state)
class SessionData:
    def __init__(self):
        self.calibrated_data = None
        self.calibrated_header = None
        self.final_phot_table = None
        self.epsf_model = None
        self.epsf_photometry_result = None
        self.log_buffer = None
        self.base_filename = "photometry"
        self.output_dir = "pfr_results"
        self.manual_ra = ""
        self.manual_dec = ""
        self.valid_ra = None
        self.valid_dec = None
        self.analysis_parameters = {
            "seeing": 3.5,
            "threshold_sigma": 3.0,
            "detection_mask": 50,
            "gaia_band": "phot_g_mean_mag",
            "gaia_min_mag": 11.0,
            "gaia_max_mag": 19.0,
        }
        self.files_loaded = {
            "science_file": None,
            "bias_file": None,
            "dark_file": None,
            "flat_file": None,
        }
        self.gaia_catalog = None
        self.matched_sources = None
        self.photometry_results = None

session = SessionData()

# Enhanced Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Photometry Factory for RAPAS 🔭", className="text-center mb-4"),
            html.Div([
                html.Small(f"App Version: {APP_VERSION}", className="text-muted")
            ], className="text-right"),
            html.Hr()
        ])
    ]),
    
    # Main content area with sidebar and content
    dbc.Row([
        # Sidebar
        dbc.Col([
            html.Div([
                # File Upload Section
                html.H4("Upload FITS Files"),
                dcc.Upload(
                    id='upload-bias',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Master Bias (optional)')
                    ]),
                    className='upload-box mb-3',
                    multiple=False
                ),
                dcc.Upload(
                    id='upload-dark',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Master Dark (optional)')
                    ]),
                    className='upload-box mb-3',
                    multiple=False
                ),
                dcc.Upload(
                    id='upload-flat',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Master Flat (optional)')
                    ]),
                    className='upload-box mb-3',
                    multiple=False
                ),
                dcc.Upload(
                    id='upload-science',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Science Image (required)')
                    ]),
                    className='upload-box mb-3',
                    multiple=False
                ),
                
                # Calibration Options
                html.H4("Calibration Options", className="mt-4"),
                dbc.Switch(id="apply-bias", label="Apply Bias", value=False),
                dbc.Switch(id="apply-dark", label="Apply Dark", value=False),
                dbc.Switch(id="apply-flat", label="Apply Flat Field", value=False),
                
                # Astrometry.net Settings
                html.H4("Astrometry.net", className="mt-4"),
                dbc.Input(
                    id="astrometry-key",
                    type="password",
                    placeholder="Enter API key",
                    className="mb-2"
                ),
                html.Small([
                    "Get your key at ",
                    html.A(
                        "nova.astrometry.net",
                        href="http://nova.astrometry.net",
                        target="_blank"
                    )
                ]),
                
                # Analysis Parameters
                html.H4("Analysis Parameters", className="mt-4"),
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Seeing (arcsec)"),
                            dbc.Input(
                                id="seeing-input",
                                type="number",
                                value=3.5,
                                min=0.5,
                                max=10,
                                step=0.1
                            )
                        ])
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Detection Threshold (σ)"),
                            dbc.Input(
                                id="threshold-input",
                                type="number",
                                value=3.0,
                                min=1.0,
                                max=10.0,
                                step=0.1
                            )
                        ])
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Border Mask (pixels)"),
                            dcc.Slider(
                                id="mask-slider",
                                min=25,
                                max=200,
                                step=25,
                                value=50,
                                marks={i: str(i) for i in range(25, 201, 25)},
                                className="mb-4"
                            )
                        ])
                    ])
                ]),
                
                # Observatory Information
                html.H4("Observatory Location", className="mt-4"),
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Observatory Name"),
                            dbc.Input(
                                id="obs-name-input",
                                type="text",
                                value="TJMS",
                                placeholder="Enter observatory name"
                            )
                        ])
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Latitude (°)"),
                            dbc.Input(
                                id="latitude-input",
                                type="number",
                                value=48.29166,
                                min=-90,
                                max=90,
                                step=0.00001
                            )
                        ])
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Longitude (°)"),
                            dbc.Input(
                                id="longitude-input",
                                type="number",
                                value=2.43805,
                                min=-180,
                                max=180,
                                step=0.00001
                            )
                        ])
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Elevation (m)"),
                            dbc.Input(
                                id="elevation-input",
                                type="number",
                                value=94.0,
                                min=0,
                                step=0.1
                            )
                        ])
                    ], className="mb-3")
                ])
            ], className="sidebar bg-light p-4"),
            
            # Quick Links
            html.Div([
                html.H4("Quick Links", className="mt-4"),
                dbc.ButtonGroup([
                    dbc.Button("GAIA Archive", href="https://gea.esac.esa.int/archive/", target="_blank", color="secondary", size="sm", className="mr-1"),
                    dbc.Button("Simbad", href="http://simbad.u-strasbg.fr/simbad/", target="_blank", color="secondary", size="sm", className="mr-1"),
                    dbc.Button("VizieR", href="http://vizier.u-strasbg.fr/viz-bin/VizieR", target="_blank", color="secondary", size="sm")
                ], vertical=True)
            ], className="mt-4")
            
        ], width=3),
        
        # Main content area
        dbc.Col([
            # Messages and alerts area
            html.Div(id="alerts-area"),
            
            # Image display and analysis area
            html.Div([
                # Science image display
                html.Div(id="science-image-container", className="mb-4"),
                
                # Image statistics
                html.Div(id="image-stats-container"),
                
                # Analysis controls
                dbc.Card([
                    dbc.CardHeader("Analysis Controls"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "Run Image Calibration",
                                    id="run-calibration-btn",
                                    color="primary",
                                    className="me-2"  # Side margin only
                                ),
                                dbc.Button(
                                    "Run Zero Point Calibration",
                                    id="run-zp-calibration-btn",
                                    color="primary",
                                    className="me-2"  # Side margin only
                                ),
                                dbc.Button(
                                    "Match with GAIA Catalog",
                                    id="run-catalog-match-btn",
                                    color="primary",
                                    className="me-2"  # Side margin only
                                ),
                            ])
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("GAIA magnitude range"),
                                dbc.Row([
                                    dbc.Col(
                                        dbc.Input(
                                            id="gaia-min-mag",
                                            type="number",
                                            value=11.0,
                                            step=0.5
                                        ),
                                        width=6
                                    ),
                                    dbc.Col(
                                        dbc.Input(
                                            id="gaia-max-mag",
                                            type="number",
                                            value=19.0,
                                            step=0.5
                                        ),
                                        width=6
                                    )
                                ])
                            ])
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Match radius (arcsec)"),
                                dbc.Input(
                                    id="match-radius",
                                    type="number",
                                    value=5.0,
                                    min=1.,
                                    max=3600.0,
                                    step=1.
                                )
                            ])
                        ], className="mb-3"),
                        dbc.Button(
                            "Download Results",
                            id="download-results-btn",
                            color="success",
                            className="mt-2"
                        ),
                        dcc.Download(id="download-data")
                    ])
                ], className="mb-4 mt-4"),  # Added top margin
                
                # Results display
                html.Div(id="results-container"),
                
                # Astrometry controls
                dbc.Card([
                    dbc.CardHeader("Astrometry Controls"),
                    dbc.CardBody([
                        dbc.Button(
                            "Solve Field",
                            id="solve-wcs-btn",
                            color="primary",
                            className="me-2"
                        ),
                        dbc.Button(
                                        "Submit Coordinates",
                                        id="submit-coords-btn",
                                        color="secondary",
                                        className="ms-2"
                                    ),
                        html.Div(id="manual-coords-form", children=[
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("RA (deg)"),
                                    dbc.Input(
                                        id="ra-input",
                                        type="number",
                                        min=0,
                                        max=360,
                                        step=0.0001
                                    )
                                ]),
                                dbc.Col([
                                    dbc.Label("Dec (deg)"),
                                    dbc.Input(
                                        id="dec-input",
                                        type="number",
                                        min=-90,
                                        max=90,
                                        step=0.0001
                                    )
                                    ])  
                                ])
                            ]),
                            html.Div(id="manual-coords-status")
                    ])
                ], className="mb-4")
            ], id="content-area")
        ], width=9)
    ])
], fluid=True)

# Add the core data processing functions (refactored for Dash)
def parse_fits_contents(contents, filename):
    """Parse uploaded FITS file contents"""
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        with BytesIO(decoded) as bio:
            with fits.open(bio) as hdul:
                data = hdul[0].data
                header = hdul[0].header
                
        return data, header
    except Exception as e:
        print(f'Error parsing {filename}: {str(e)}')
        return None, None

# File upload callbacks
@app.callback(
    [Output('science-image-container', 'children'),
     Output('image-stats-container', 'children'),
     Output('alerts-area', 'children', allow_duplicate=True)],  # Add allow_duplicate
    [Input('upload-science', 'contents')],
    [State('upload-science', 'filename')],
    prevent_initial_call=True
)
def update_science_image(contents, filename):
    if contents is None:
        return html.Div("No image loaded"), html.Div()
    
    data, header = parse_fits_contents(contents, filename)
    if data is None:
        return html.Div("Error loading image"), html.Div()
    
    # Store data in session
    session.files_loaded["science_file"] = (data, header)
    
    # Create image figure
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale='Viridis',
        colorbar=dict(title='Pixel Value')
    ))
    
    fig.update_layout(
        title=f"Science Image: {filename}",
        **PLOTLY_THEME['layout']
    )
    
    # Calculate statistics
    stats = dbc.Card([
        dbc.CardHeader("Image Statistics"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Mean"),
                    html.P(f"{np.mean(data):.3f}")
                ]),
                dbc.Col([
                    html.H6("Median"),
                    html.P(f"{np.median(data):.3f}")
                ]),
                dbc.Col([
                    html.H6("Std Dev"),
                    html.P(f"{np.std(data):.3f}")
                ])
            ])
        ])
    ])
    
    return dcc.Graph(figure=fig), stats

def create_alert(message, color="primary", dismissable=True):
    """Utility function to create consistent alert messages"""
    return dbc.Alert(
        message,
        color=color,
        dismissable=dismissable,
        duration=4000
    )

# Add calibration callback
@app.callback(
    [Output("alerts-area", "children", allow_duplicate=True),
     Output("science-image-container", "children", allow_duplicate=True)],
    [Input("run-calibration-btn", "n_clicks")],
    [State("apply-bias", "value"),
     State("apply-dark", "value"),
     State("apply-flat", "value")],
    prevent_initial_call=True
)
def run_calibration(n_clicks, apply_bias, apply_dark, apply_flat):
    if n_clicks is None:
        return None, dash.no_update
    
    if not any([apply_bias, apply_dark, apply_flat]):
        return create_alert("No calibration steps selected", "warning"), dash.no_update
    
    if session.files_loaded["science_file"] is None:
        return create_alert("No science image loaded", "danger"), dash.no_update
    
    try:
        science_data, science_header = session.files_loaded["science_file"]
        bias_data = session.files_loaded.get("bias_file", (None, None))[0]
        dark_data = session.files_loaded.get("dark_file", (None, None))[0]
        flat_data = session.files_loaded.get("flat_file", (None, None))[0]
        
        # Perform calibration
        calibrated_data = science_data.copy()
        steps_applied = []
        
        if apply_bias and bias_data is not None:
            calibrated_data -= bias_data
            steps_applied.append("Bias")
        
        if apply_dark and dark_data is not None:
            exposure_ratio = science_header.get("EXPTIME", 1.0) / session.files_loaded["dark_file"][1].get("EXPTIME", 1.0)
            calibrated_data -= dark_data * exposure_ratio
            steps_applied.append("Dark")
            
        if apply_flat and flat_data is not None:
            calibrated_data /= (flat_data / np.median(flat_data))
            steps_applied.append("Flat")
            
        # Store calibrated data
        session.calibrated_data = calibrated_data
        session.calibrated_header = science_header.copy()
        
        # Create new figure
        fig = go.Figure(data=go.Heatmap(
            z=calibrated_data,
            colorscale='Viridis',
            colorbar=dict(title='Pixel Value')
        ))
        
        fig.update_layout(
            title="Calibrated Science Image",
            **PLOTLY_THEME['layout']
        )
        
        return (
            create_alert(f"Applied calibrations: {', '.join(steps_applied)}", "success"),
            dcc.Graph(figure=fig)
        )
        
    except Exception as e:
        return create_alert(f"Calibration error: {str(e)}", "danger"), dash.no_update

# Add source detection callback
@app.callback(
    [Output("results-container", "children", allow_duplicate=True),
     Output("alerts-area", "children", allow_duplicate=True)],
    [Input("run-zp-calibration-btn", "n_clicks")],
    [State("seeing-input", "value"),
     State("threshold-input", "value"),
     State("mask-slider", "value")],
    prevent_initial_call=True
)
def run_source_detection(n_clicks, seeing, threshold, mask_size):
    if n_clicks is None:
        return dash.no_update, dash.no_update
    
    # Use calibrated data if available, otherwise use raw science data
    data = session.calibrated_data if session.calibrated_data is not None else session.files_loaded["science_file"][0]
    header = session.calibrated_header if session.calibrated_header is not None else session.files_loaded["science_file"][1]
    
    if data is None:
        return None, create_alert("No image data available", "danger")
    
    try:
        # Estimate background
        bkg = Background2D(
            data,
            box_size=100,
            filter_size=5,
            sigma_clip=SigmaClip(sigma=3.0)
        )
        
        # Create detection mask
        detection_mask = np.zeros_like(data, dtype=bool)
        detection_mask[mask_size:-mask_size, mask_size:-mask_size] = True
        
        # Detect sources
        mean_fwhm = seeing / header.get("PIXSCALE", 1.0)
        daofind = DAOStarFinder(
            fwhm=mean_fwhm,
            threshold=threshold * bkg.background_rms.mean()
        )
        
        sources = daofind(data - bkg.background, mask=~detection_mask)
        
        if sources is None or len(sources) == 0:
            return None, create_alert("No sources detected", "warning")
        
        # Create source overlay plot
        fig = go.Figure()
        
        # Add image
        fig.add_trace(go.Heatmap(
            z=data,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Pixel Value')
        ))
        
        # Add source markers
        fig.add_trace(go.Scatter(
            x=sources['xcentroid'],
            y=sources['ycentroid'],
            mode='markers',
            marker=dict(
                color='red',
                size=10,
                symbol='circle-open'
            ),
            name='Detected Sources'
        ))
        
        fig.update_layout(
            title=f"Detected Sources (N={len(sources)})",
            **PLOTLY_THEME['layout']
        )
        
        # Create results table
        table = dbc.Table.from_dataframe(
            pd.DataFrame(sources)[:10],  # Show first 10 sources
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            className="mt-4"
        )
        
        results_div = html.Div([
            dcc.Graph(figure=fig),
            html.H5("Source Catalog (first 10 entries)", className="mt-4"),
            table
        ])
        
        return results_div, create_alert(f"Detected {len(sources)} sources", "success")
        
    except Exception as e:
        return None, create_alert(f"Source detection error: {str(e)}", "danger")

# Add WCS solving callbacks
@app.callback(
    [Output("alerts-area", "children", allow_duplicate=True),
     Output("results-container", "children", allow_duplicate=True)],
    [Input("solve-wcs-btn", "n_clicks")],
    [State("astrometry-key", "value")],
    prevent_initial_call=True
)
def solve_astrometry(n_clicks, api_key):
    if n_clicks is None:
        return dash.no_update, dash.no_update
    
    if not api_key:
        return create_alert("Please enter your Astrometry.net API key", "warning"), dash.no_update
    
    try:
        from astroquery.astrometry_net import AstrometryNet
        ast = AstrometryNet()
        ast.api_key = api_key
        
        # Get current data
        data = session.calibrated_data if session.calibrated_data is not None else session.files_loaded["science_file"][0]
        header = session.calibrated_header if session.calibrated_header is not None else session.files_loaded["science_file"][1]
        
        if data is None:
            return create_alert("No image data available", "danger"), dash.no_update
        
        # Try to solve
        try_idx = 1
        max_tries = 3
        wcs_header = None
        
        while try_idx <= max_tries and wcs_header is None:
            try:
                # Create progress indicator
                progress = html.Div([
                    dbc.Spinner(size="sm"),
                    f" Attempt {try_idx}/{max_tries}: Submitting to astrometry.net..."
                ])
                
                wcs_header = ast.solve_from_image(data, force_image_upload=True)
                
            except Exception as e:
                try_idx += 1
                
        if wcs_header is None:
            return create_alert(f"Astrometry.net failed after {max_tries} attempts", "danger"), dash.no_update
            
        # Update header with WCS solution
        for key in wcs_header:
            if key not in ['SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND']:
                header[key] = wcs_header[key]
        
        header['ASTRSRC'] = 'astrometry.net'
        
        # Store updated header
        if session.calibrated_data is not None:
            session.calibrated_header = header
        else:
            session.files_loaded["science_file"] = (data, header)
            
        # Create WCS info display
        wcs = WCS(header)
        ra_center, dec_center = wcs.wcs.crval
        
        wcs_info = dbc.Card([
            dbc.CardHeader("WCS Solution"),
            dbc.CardBody([
                html.P(f"Field center: RA = {ra_center:.6f}°, Dec = {dec_center:.6f}°"),
                html.P(f"Pixel scale: {wcs.wcs.cdelt[0]*3600:.3f} arcsec/pixel")
            ])
        ])
        
        return (
            create_alert("Astrometric solution successful!", "success"),
            wcs_info
        )
        
    except Exception as e:
        return create_alert(f"Astrometry error: {str(e)}", "danger"), dash.no_update

# Add coordinate input callbacks
@app.callback(
    Output("manual-coords-status", "children"),
    [Input("submit-coords-btn", "n_clicks")],
    [State("ra-input", "value"),
     State("dec-input", "value")]
)
def update_manual_coordinates(n_clicks, ra, dec):
    if n_clicks is None:
        return dash.no_update
        
    try:
        ra_val = float(ra)
        dec_val = float(dec)
        
        if not (0 <= ra_val < 360):
            return create_alert("RA must be between 0 and 360 degrees", "warning")
        if not (-90 <= dec_val <= 90):
            return create_alert("Dec must be between -90 and 90 degrees", "warning")
            
        # Store coordinates in session
        session.manual_ra = ra_val
        session.manual_dec = dec_val
        
        # Update header if it exists
        if session.files_loaded["science_file"] is not None:
            data, header = session.files_loaded["science_file"]
            header["RA"] = ra_val
            header["DEC"] = dec_val
            session.files_loaded["science_file"] = (data, header)
            
        return create_alert(f"Coordinates updated: RA={ra_val}°, Dec={dec_val}°", "success")
        
    except ValueError:
        return create_alert("Invalid coordinate values", "danger")

# Add new query and analysis functions
def query_gaia_catalog(ra, dec, radius, mag_range):
    """Query GAIA catalog around field center"""
    try:
        coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
        width = radius * u.arcmin
        
        Gaia.ROW_LIMIT = 5000  # Adjust as needed
        
        # Build query
        query = f"""
        SELECT source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius/60.0})
        )
        AND phot_g_mean_mag BETWEEN {mag_range[0]} AND {mag_range[1]}
        """
        
        job = Gaia.launch_job_async(query)
        result = job.get_results()
        
        return result
    except Exception as e:
        print(f"GAIA query error: {str(e)}")
        return None

# Add catalog matching callback
@app.callback(
    [Output("results-container", "children", allow_duplicate=True),
     Output("alerts-area", "children", allow_duplicate=True)],
    [Input("run-catalog-match-btn", "n_clicks")],
    [State("gaia-min-mag", "value"),
     State("gaia-max-mag", "value"),
     State("match-radius", "value")],
    prevent_initial_call=True
)
def run_catalog_matching(n_clicks, min_mag, max_mag, match_radius):
    if n_clicks is None:
        return dash.no_update, dash.no_update
    
    # Check for WCS solution
    if session.calibrated_header is None or 'CRVAL1' not in session.calibrated_header:
        return None, create_alert("WCS solution required before catalog matching", "warning")
    
    try:
        wcs = WCS(session.calibrated_header)
        image_center = wcs.wcs.crval
        
        # Query GAIA catalog
        catalog = query_gaia_catalog(
            image_center[0], 
            image_center[1], 
            30.0,  # 30 arcmin radius
            (min_mag, max_mag)
        )
        
        if catalog is None or len(catalog) == 0:
            return None, create_alert("No GAIA sources found in field", "warning")
            
        # Convert catalog coordinates to pixel coordinates
        pixels = wcs.all_world2pix(
            np.column_stack((catalog['ra'], catalog['dec'])), 
            0
        )
        
        # Create visualization
        fig = go.Figure()
        
        # Plot image
        data = session.calibrated_data if session.calibrated_data is not None else session.files_loaded["science_file"][0]
        fig.add_trace(go.Heatmap(
            z=data,
            colorscale='Viridis',
            showscale=True
        ))
        
        # Plot GAIA sources
        fig.add_trace(go.Scatter(
            x=pixels[:, 0],
            y=pixels[:, 1],
            mode='markers',
            marker=dict(
                color='red',
                size=12,
                symbol='circle-open'
            ),
            name='GAIA Sources'
        ))
        
        fig.update_layout(
            title="GAIA Catalog Overlay",
            **PLOTLY_THEME['layout']
        )
        
        # Create results summary
        results_div = html.Div([
            dcc.Graph(figure=fig),
            html.H5(f"Found {len(catalog)} GAIA sources in field", className="mt-4"),
            dbc.Table.from_dataframe(
                pd.DataFrame({
                    'RA': catalog['ra'],
                    'Dec': catalog['dec'],
                    'G mag': catalog['phot_g_mean_mag']
                }).head(10),
                striped=True,
                bordered=True,
                hover=True
            )
        ])
        
        session.gaia_catalog = catalog
        
        return results_div, create_alert("Catalog matching completed", "success")
        
    except Exception as e:
        return None, create_alert(f"Catalog matching error: {str(e)}", "danger")

# Add photometry analysis and zero-point calibration functions and callbacks
def perform_aperture_photometry(data, sources, fwhm, gain=1.0, readnoise=10.0):
    """Perform aperture photometry on detected sources"""
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=2.0*fwhm)
    annulus_apertures = CircularAnnulus(positions, r_in=4.0*fwhm, r_out=6.0*fwhm)
    
    phot_table = aperture_photometry(data, apertures)
    bkg_table = aperture_photometry(data, annulus_apertures)
    
    # Calculate background per pixel
    bkg_mean = bkg_table['aperture_sum'] / annulus_apertures.area
    bkg_sum = bkg_mean * apertures.area
    
    # Subtract background
    final_sum = phot_table['aperture_sum'] - bkg_sum
    phot_table['net_flux'] = final_sum
    phot_table['mag_inst'] = -2.5 * np.log10(final_sum)
    
    # Add error calculation
    mag_errors, snr, quality_flags = calculate_photometry_uncertainties(
        data, sources, phot_table, gain, readnoise
    )
    phot_table['mag_inst_err'] = mag_errors
    phot_table['snr'] = snr
    phot_table['quality_flag'] = quality_flags
    
    return phot_table

def calculate_photometry_uncertainties(data, sources, phot_table, gain=1.0, readnoise=10.0):
    """Calculate photometric uncertainties including:
    - Photon noise
    - Sky background noise
    - Read noise
    - Gain variations
    """
    
    # Extract measurements
    signal = phot_table['net_flux']
    sky_per_pixel = phot_table['aperture_sum_1'] / phot_table['aperture_area_1']
    npix = phot_table['aperture_area_0']
    
    # Calculate various noise contributions
    shot_noise = np.sqrt(signal/gain)
    sky_noise = np.sqrt(sky_per_pixel * npix) / gain
    readnoise_total = readnoise * np.sqrt(npix) / gain
    
    # Total uncertainty (add in quadrature)
    total_error = np.sqrt(shot_noise**2 + sky_noise**2 + readnoise_total**2)
    
    # Convert to magnitude errors
    mag_errors = 2.5 * np.log10(1.0 + total_error/signal)
    
    # Add SNR calculation
    snr = signal / total_error
    
    # Add quality flags
    quality_flags = np.zeros(len(signal), dtype=int)
    quality_flags[snr < 5] |= 1  # Low SNR
    quality_flags[total_error/signal > 0.2] |= 2  # High relative error
    
    return mag_errors, snr, quality_flags

def calculate_zero_point(inst_mags, ref_mags, inst_errors):
    """Calculate zero point with sigma clipping"""
    differences = ref_mags - inst_mags
    
    # Apply sigma clipping to remove outliers
    clipped_diffs = sigma_clip(differences, sigma=3, maxiters=5)
    good_indices = ~clipped_diffs.mask
    
    # Calculate weighted mean zero point
    weights = 1.0 / (inst_errors[good_indices]**2)
    zp = np.average(differences[good_indices], weights=weights)
    zp_std = np.sqrt(1.0 / np.sum(weights))
    
    return zp, zp_std, good_indices

# Modify the zero-point calibration callback
@app.callback(
    [Output("results-container", "children", allow_duplicate=True),
     Output("alerts-area", "children", allow_duplicate=True)],
    [Input("run-zp-calibration-btn", "n_clicks")],
    [State("seeing-input", "value")],
    prevent_initial_call=True
)
def run_zp_calibration(n_clicks, seeing):
    if n_clicks is None:
        return dash.no_update, dash.no_update
    
    if session.gaia_catalog is None:
        return None, create_alert("GAIA catalog matching required first", "warning")
        
    try:
        # Get current data and WCS
        data = session.calibrated_data if session.calibrated_data is not None else session.files_loaded["science_file"][0]
        header = session.calibrated_header if session.calibrated_header is not None else session.files_loaded["science_file"][1]
        wcs = WCS(header)
        
        # Convert GAIA coordinates to pixels
        gaia_pixels = wcs.all_world2pix(
            np.column_stack((session.gaia_catalog['ra'], session.gaia_catalog['dec'])),
            0
        )
        
        # Create source table for photometry
        sources = Table({
            'xcentroid': gaia_pixels[:, 0],
            'ycentroid': gaia_pixels[:, 1],
            'gaia_mag': session.gaia_catalog['phot_g_mean_mag']
        })
        
        # Perform photometry
        phot_table = perform_aperture_photometry(
            data,
            sources,
            seeing / abs(header.get('CDELT1', 1.0) * 3600.0)  # Convert seeing to pixels
        )
        
        # Calculate zero point
        good_sources = np.isfinite(phot_table['mag_inst']) & (phot_table['snr'] > 5)
        zp, zp_std, zp_indices = calculate_zero_point(
            phot_table['mag_inst'][good_sources],
            sources['gaia_mag'][good_sources],
            phot_table['mag_inst_err'][good_sources]
        )
        
        # Enhanced results DataFrame
        results_df = pd.DataFrame({
            'X': phot_table['xcenter'][good_sources],
            'Y': phot_table['ycenter'][good_sources],
            'Inst_Mag': phot_table['mag_inst'][good_sources],
            'Inst_Mag_Err': phot_table['mag_inst_err'][good_sources],
            'Cal_Mag': phot_table['mag_inst'][good_sources] + zp,
            'Cal_Mag_Err': np.sqrt(
                phot_table['mag_inst_err'][good_sources]**2 + zp_std**2
            ),
            'GAIA_G': sources['gaia_mag'][good_sources],
            'SNR': phot_table['snr'][good_sources],
            'Quality_Flag': phot_table['quality_flag'][good_sources]
        })
        
        # Add magnitude difference statistics
        mag_diff = results_df['GAIA_G'] - results_df['Cal_Mag']
        mag_diff_stats = {
            'median': np.median(mag_diff),
            'std': np.std(mag_diff),
            'mad': np.median(np.abs(mag_diff - np.median(mag_diff)))
        }
        
        # Enhanced results display
        results_div = html.Div([
            dcc.Graph(figure=fig),
            dbc.Card([
                dbc.CardHeader("Photometric Calibration Results"),
                dbc.CardBody([
                    html.P([
                        html.Strong("Zero Point: "),
                        f"{zp:.3f} ± {zp_std:.3f} mag"
                    ]),
                    html.P([
                        html.Strong("Median error: "),
                        f"{np.median(results_df['Cal_Mag_Err']):.3f} mag"
                    ]),
                    html.P([
                        html.Strong("Number of stars used: "),
                        f"{len(results_df)}"
                    ]),
                    html.P([
                        html.Strong("Photometric Quality: "),
                        f"σMAD = {mag_diff_stats['mad']:.3f} mag"
                    ]),
                    html.P([
                        html.Strong("Median SNR: "),
                        f"{np.median(results_df['SNR'])::.1f}"
                    ])
                ])
            ]),
            html.H5("Calibrated Source Catalog (with uncertainties)", className="mt-4"),
            dbc.Table.from_dataframe(
                results_df.round(3).head(10),
                striped=True,
                bordered=True,
                hover=True
            )
        ])
        
        # Update session storage with enhanced results
        session.photometry_results = {
            'zp': zp,
            'zp_std': zp_std,
            'phot_table': phot_table,
            'calibrated_mags': results_df['Cal_Mag'],
            'calibrated_errors': results_df['Cal_Mag_Err'],
            'results_df': results_df,
            'mag_diff_stats': mag_diff_stats,
            'quality_summary': {
                'n_high_quality': np.sum(results_df['Quality_Flag'] == 0),
                'n_low_snr': np.sum(results_df['Quality_Flag'] & 1),
                'n_high_error': np.sum(results_df['Quality_Flag'] & 2)
            }
        }
        
        return results_div, create_alert(
            f"Zero point calibration complete: ZP = {zp:.3f} ± {zp_std:.3f} "
            f"(σMAD = {mag_diff_stats['mad']:.3f})",
            "success"
        )
        
    except Exception as e:
        return None, create_alert(f"Zero point calibration error: {str(e)}", "danger")

# Update the download results callback
@app.callback(
    Output("download-data", "data"),
    [Input("download-results-btn", "n_clicks")]
)
def download_results(n_clicks):
    if n_clicks is None or session.photometry_results is None:
        return dash.no_update
        
    try:
        # Use the enhanced results DataFrame
        results_df = session.photometry_results['results_df']
        
        return dict(
            content=results_df.to_csv(index=False),
            filename=f"photometry_results_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv",
            type="text/csv"
        )
        
    except Exception as e:
        print(f"Download error: {str(e)}")
        return None

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)
