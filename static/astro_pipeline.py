import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from astropy.io import fits
from ccdproc import CCDData, subtract_bias, subtract_dark, flat_correct, combine
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.psf import PSFPhotometry, GaussianPSF
from astropy.stats import sigma_clipped_stats
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from photutils.detection import DAOStarFinder, IRAFStarFinder

class ImageCalibrationPhotometryApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Astronomical Image Calibration and Photometry Pipeline")

        self.science_image_path = tk.StringVar()
        self.bias_frames_path = tk.StringVar()
        self.dark_frames_path = tk.StringVar()
        self.flat_frames_path = tk.StringVar()
        self.calibrated_image_data = None

        self.create_widgets()

    def create_widgets(self):
        # --- File Loading Section ---
        file_frame = ttk.LabelFrame(self.master, text="1. Load Files")
        file_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(file_frame, text="Science Image:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(file_frame, textvariable=self.science_image_path, width=50).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(file_frame, text="Browse", command=self.load_science_image).grid(row=0, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(file_frame, text="Bias Frames Directory:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(file_frame, textvariable=self.bias_frames_path, width=50).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(file_frame, text="Browse", command=self.load_bias_frames).grid(row=1, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(file_frame, text="Dark Frames Directory:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(file_frame, textvariable=self.dark_frames_path, width=50).grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(file_frame, text="Browse", command=self.load_dark_frames).grid(row=2, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(file_frame, text="Flat Frames Directory:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(file_frame, textvariable=self.flat_frames_path, width=50).grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(file_frame, text="Browse", command=self.load_flat_frames).grid(row=3, column=2, padx=5, pady=5, sticky="w")

        # --- Calibration Section ---
        calibration_frame = ttk.LabelFrame(self.master, text="2. Calibration")
        calibration_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.apply_bias = tk.BooleanVar(value=True)
        ttk.Checkbutton(calibration_frame, text="Apply Bias Correction", variable=self.apply_bias).grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.apply_dark = tk.BooleanVar(value=True)
        ttk.Checkbutton(calibration_frame, text="Apply Dark Correction", variable=self.apply_dark).grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.apply_flat = tk.BooleanVar(value=True)
        ttk.Checkbutton(calibration_frame, text="Apply Flat Field Correction", variable=self.apply_flat).grid(row=2, column=0, padx=5, pady=5, sticky="w")

        ttk.Button(calibration_frame, text="Calibrate Image", command=self.calibrate).grid(row=3, column=0, padx=5, pady=10, sticky="ew")

        # --- Photometry Section ---
        photometry_frame = ttk.LabelFrame(self.master, text="3. Photometry")
        photometry_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        self.photometry_type = tk.StringVar(value="aperture")
        ttk.Radiobutton(photometry_frame, text="Aperture Photometry", variable=self.photometry_type, value="aperture").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(photometry_frame, text="PSF Photometry", variable=self.photometry_type, value="psf").grid(row=1, column=0, padx=5, pady=5, sticky="w")

        # Aperture Photometry Parameters
        self.aperture_radius = tk.DoubleVar(value=5.0)
        self.inner_annulus_radius = tk.DoubleVar(value=10.0)
        self.outer_annulus_radius = tk.DoubleVar(value=15.0)

        ttk.Label(photometry_frame, text="Aperture Radius (pixels):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(photometry_frame, textvariable=self.aperture_radius, width=10).grid(row=2, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(photometry_frame, text="Inner Annulus Radius (pixels):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(photometry_frame, textvariable=self.inner_annulus_radius, width=10).grid(row=3, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(photometry_frame, text="Outer Annulus Radius (pixels):").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(photometry_frame, textvariable=self.outer_annulus_radius, width=10).grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # PSF Photometry Parameters (Simplified)
        ttk.Label(photometry_frame, text="PSF Photometry (Simplified):").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        ttk.Label(photometry_frame, text="Assumes a Gaussian PSF for demonstration.").grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        ttk.Button(photometry_frame, text="Perform Photometry", command=self.perform_photometry).grid(row=7, column=0, padx=5, pady=10, sticky="ew")

        # --- Zero Point Calibration Section ---
        zeropoint_frame = ttk.LabelFrame(self.master, text="4. Zero Point Calibration")
        zeropoint_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(zeropoint_frame, text="Standard Star Coordinates (RA, Dec - comma separated) and Magnitudes (comma separated):").grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        self.standard_star_info = tk.Text(zeropoint_frame, height=5, width=60)
        self.standard_star_info.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        self.standard_star_info.insert(tk.END, "e.g., 10.68375, +41.26917, 9.5\n10.68583, +41.27139, 10.2")

        ttk.Button(zeropoint_frame, text="Calculate Zero Point", command=self.calculate_zeropoint).grid(row=2, column=0, padx=5, pady=10, sticky="ew")

        # --- Image Display Section ---
        image_frame = ttk.LabelFrame(self.master, text="Image Display")
        image_frame.grid(row=0, column=1, rowspan=4, padx=10, pady=10, sticky="nsew")
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.plot = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=image_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # --- Results Section ---
        results_frame = ttk.LabelFrame(self.master, text="Results")
        results_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.results_text = tk.Text(results_frame, height=10, width=80)
        self.results_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_rowconfigure(3, weight=1)
        image_frame.grid_rowconfigure(0, weight=1)
        image_frame.grid_columnconfigure(0, weight=1)

    def load_science_image(self):
        filepath = filedialog.askopenfilename(title="Select Science Image", filetypes=[("FITS files", "*.fits"), ("All files", "*.*")])
        if filepath:
            self.science_image_path.set(filepath)
            try:
                with fits.open(filepath) as hdul:
                    self.display_image(hdul.data)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load science image: {e}")

    def load_bias_frames(self):
        directory = filedialog.askdirectory(title="Select Directory Containing Bias Frames")
        if directory:
            self.bias_frames_path.set(directory)

    def load_dark_frames(self):
        directory = filedialog.askdirectory(title="Select Directory Containing Dark Frames")
        if directory:
            self.dark_frames_path.set(directory)

    def load_flat_frames(self):
        directory = filedialog.askdirectory(title="Select Directory Containing Flat Frames")
        if directory:
            self.flat_frames_path.set(directory)

    def display_image(self, image_data):
        self.plot.clear()
        if image_data is not None:
            self.plot.imshow(image_data, cmap='gray', origin='lower')
        self.canvas.draw()

    def calibrate(self):
        science_path = self.science_image_path.get()
        bias_dir = self.bias_frames_path.get()
        dark_dir = self.dark_frames_path.get()
        flat_dir = self.flat_frames_path.get()

        if not science_path:
            messagebox.showerror("Error", "Please load a science image.")
            return

        try:
            with fits.open(science_path) as hdul:
                science_image = CCDData(hdul.data, unit="adu")

            if self.apply_bias.get() and bias_dir:
                bias_fnames = [f for f in tk.os.listdir(bias_dir) if f.endswith(".fits")]
                if bias_fnames:
                    bias_list = [CCDData.read(tk.os.path.join(bias_dir, fname), unit="adu") for fname in bias_fnames]
                    master_bias = combine(bias_list, method='median', unit='adu')
                    science_image = subtract_bias(science_image, master_bias)
                    self.results_text.insert(tk.END, "Bias correction applied.\n")
                else:
                    self.results_text.insert(tk.END, "No bias frames found in the specified directory.\n")

            if self.apply_dark.get() and dark_dir:
                dark_fnames = [f for f in tk.os.listdir(dark_dir) if f.endswith(".fits")]
                if dark_fnames:
                    dark_list = [CCDData.read(tk.os.path.join(dark_dir, fname), unit="adu") for fname in dark_fnames]
                    master_dark = combine(dark_list, method='median', unit='adu')
                    science_image = subtract_dark(science_image, master_dark, exposure_time='exptime', exposure_unit='s', scale=True)
                    self.results_text.insert(tk.END, "Dark correction applied.\n")
                else:
                    self.results_text.insert(tk.END, "No dark frames found in the specified directory.\n")

            if self.apply_flat.get() and flat_dir:
                flat_fnames = [f for f in tk.os.listdir(flat_dir) if f.endswith(".fits")]
                if flat_fnames:
                    flat_list = [CCDData.read(tk.os.path.join(flat_dir, fname), unit="adu") for fname in flat_fnames]
                    master_flat = combine(flat_list, method='median', unit='adu')
                    science_image = flat_correct(science_image, master_flat)
                    self.results_text.insert(tk.END, "Flat field correction applied.\n")
                else:
                    self.results_text.insert(tk.END, "No flat frames found in the specified directory.\n")

            self.calibrated_image_data = science_image.data
            self.display_image(self.calibrated_image_data)
            self.results_text.insert(tk.END, "Calibration complete.\n")

        except FileNotFoundError:
            messagebox.showerror("Error", "One or more calibration files not found.")
        except Exception as e:
            messagebox.showerror("Error", f"Calibration error: {e}")

    def perform_photometry(self):
        if self.calibrated_image_data is None:
            messagebox.showerror("Error", "Please calibrate the image first.")
            return

        try:
            if self.photometry_type.get() == "aperture":
                radius = self.aperture_radius.get()
                inner_radius = self.inner_annulus_radius.get()
                outer_radius = self.outer_annulus_radius.get()

                # For demonstration, let's find some bright sources using a simple threshold
                threshold = np.percentile(self.calibrated_image_data, 95)
                y, x = np.where(self.calibrated_image_data > threshold)
                positions = np.transpose((x, y))[:10] # Take top 10 for example

                apertures = CircularAperture(positions, r=radius)
                annulus_apertures = CircularAnnulus(positions, r_in=inner_radius, r_out=outer_radius)

                phot_table = aperture_photometry(self.calibrated_image_data, apertures)
                bkg_mean = []
                for i in range(len(positions)):
                    annulus_mask = annulus_apertures.to_mask(method='center')[i]
                    annulus_data = annulus_mask.multiply(self.calibrated_image_data)
                    bkg_mean.append(np.median(annulus_data[annulus_mask.data > 0]))

                bkg_mean = np.array(bkg_mean)
                phot_table['bkg_mean'] = bkg_mean
                phot_table['aperture_sum_bkgsub'] = phot_table['aperture_sum'] - (bkg_mean * apertures.area)

                self.results_text.insert(tk.END, "\nAperture Photometry Results:\n")
                for row in phot_table:
                    self.results_text.insert(tk.END, f"Position: ({row['xcenter']:.2f}, {row['ycenter']:.2f}), Counts (Background Subtracted): {row['aperture_sum_bkgsub']:.2f}\n")

            elif self.photometry_type.get() == "psf":
                # Simplified PSF photometry using Gaussian PSF for demonstration
                mean, median, std = sigma_clipped_stats(self.calibrated_image_data, sigma=3.0)
                daofind = DAOStarFinder(fwhm=3.0, threshold=5.0*std)
                sources = daofind(self.calibrated_image_data - median)

                if sources is not None:
                    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
                    psf_model = GaussianPSF(fwhm=3.0)
                    finder = IRAFStarFinder(threshold=5.0*std, fwhm=3.0)
                    detected_sources = finder(self.calibrated_image_data - median)

                    if detected_sources:
                        psf_photometry = PSFPhotometry(psf_model, fitshape=9, finder=finder)
                        result_table = psf_photometry(self.calibrated_image_data)

                        self.results_text.insert(tk.END, "\nPSF Photometry Results (Simplified):\n")
                        for row in result_table:
                            self.results_text.insert(tk.END, f"Position: ({row['x_0']:.2f}, {row['y_0']:.2f}), Flux: {row['flux_0']:.2f}\n")
                    else:
                        self.results_text.insert(tk.END, "No sources found for PSF photometry.\n")
                else:
                    self.results_text.insert(tk.END, "No sources found for PSF photometry.\n")

        except Exception as e:
            messagebox.showerror("Error", f"Photometry error: {e}")

    def calculate_zeropoint(self):
        if self.calibrated_image_data is None:
            messagebox.showerror("Error", "Please calibrate the image first and perform photometry.")
            return

        standard_star_info = self.standard_star_info.get("1.0", tk.END).strip().split('\n')
        if not standard_star_info or not any(standard_star_info):
            messagebox.showerror("Error", "Please enter standard star coordinates and magnitudes.")
            return

        try:
            standard_stars = []
            for line in standard_star_info:
                ra_dec_str, mag_str = line.split(',')[:2], line.split(',')[-1]
                ra = float(ra_dec_str.strip())
                dec = float(ra_dec_str[1].strip())
                mag = float(mag_str.strip())
                standard_stars.append(((ra, dec), mag))

            if not standard_stars:
                messagebox.showerror("Error", "No valid standard star information provided.")
                return

            # Perform aperture photometry on the standard stars (assuming known positions)
            radius = self.aperture_radius.get()
            apertures = CircularAperture([star for star in standard_stars], r=radius)
            phot_table = aperture_photometry(self.calibrated_image_data, apertures)

            instrumental_magnitudes = []
            standard_magnitudes = [star[1] for star in standard_stars]

            for row in phot_table:
                counts = row['aperture_sum']
                exposure_time = 1.0 # Assuming 1 second exposure for simplicity, adjust if needed
                if counts > 0:
                    instrumental_magnitude = -2.5 * np.log10(counts / exposure_time)
                    instrumental_magnitudes.append(instrumental_magnitude)
                else:
                    self.results_text.insert(tk.END, f"Warning: Zero or negative counts for a standard star.\n")
                    return

            if len(instrumental_magnitudes)!= len(standard_magnitudes):
                messagebox.showerror("Error", "Number of detected standard stars does not match the input.")
                return

            zero_points = np.array(standard_magnitudes) - np.array(instrumental_magnitudes)
            mean_zero_point = np.mean(zero_points)

            self.results_text.insert(tk.END, f"\nZero Point Calibration Results:\n")
            for i in range(len(standard_magnitudes)):
                self.results_text.insert(tk.END, f"Standard Star {i+1}: Standard Mag = {standard_magnitudes[i]:.2f}, Instrumental Mag = {instrumental_magnitudes[i]:.2f}, Zero Point = {zero_points[i]:.2f}\n")
            self.results_text.insert(tk.END, f"Mean Zero Point: {mean_zero_point:.2f}\n")

        except ValueError:
            messagebox.showerror("Error", "Invalid format for standard star information. Please use 'RA, Dec, Magnitude' per line.")
        except Exception as e:
            messagebox.showerror("Error", f"Zero point calculation error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCalibrationPhotometryApp(root)
    root.mainloop()