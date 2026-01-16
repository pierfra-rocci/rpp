from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import numpy as np
import os
import cv2
from astropy.io import fits
import tensorflow as tf
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import pandas as pd
import argparse
import time


class ASTA:
    def __init__(self, model_path):
        """
        Initializes the ASTA class by loading a pre-trained model for astronomical image analysis.

        Parameters:
        model_path (str): Path to the pre-trained model file.
        """
        self.model = tf.keras.models.load_model(model_path, custom_objects={'dice_BCE_loss': self.dice_BCE_loss,
                                                                            'dice_coeff': self.dice_coeff})

    def zscale_image(self, image_data):
        """
        Scales image data using the ZScale algorithm for better visualization.

        Parameters:
        image_data (numpy.ndarray): Input image data.

        Returns:
        numpy.ndarray: Scaled image data.
        """
        interval = ZScaleInterval()
        zmin, zmax = interval.get_limits(image_data)
        scaled_img = (image_data - zmin) / (zmax - zmin)
        return np.clip(scaled_img, 0, 1)

    def remove_small_objects(self, mask, min_size=100):
        """
        Removes small connected components from a binary mask.

        Parameters:
        mask (numpy.ndarray): Input binary mask.
        min_size (int): Minimum size of objects to keep.

        Returns:
        numpy.ndarray: Mask with small objects removed.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask,
                                                                        connectivity=8)
        new_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                new_mask[labels == i] = 1
        return new_mask

    def calculate_line_angle(self, line):
        """
        Calculates the angle of a line segment in degrees.

        Parameters:
        line (tuple): Coordinates of the line segment (x1, y1, x2, y2).

        Returns:
        float: Angle of the line segment in degrees.
        """
        x1, y1, x2, y2 = line
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        return angle if angle >= 0 else angle + 360

    def find_trail_extremities(self, lines):
        """
        Finds the extremities of a set of line segments.

        Parameters:
        lines (numpy.ndarray): Array of line segments.

        Returns:
        tuple: Coordinates of the start and end points of the trail.
        """
        points = np.vstack([lines[:, :2], lines[:, 2:4]])
        dist_matrix = squareform(pdist(points, 'euclidean'))
        start_idx, end_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        return points[start_idx], points[end_idx]

    def predict_on_batch(self, patches):
        """
        Predicts masks for a batch of image patches using the pre-trained model.

        Parameters:
        patches (numpy.ndarray): Batch of image patches.

        Returns:
        numpy.ndarray: Predicted masks for the input patches.
        """
        patches = np.expand_dims(patches, axis=-1)
        preds = self.model.predict(patches)
        return preds[..., 0]

    def dice_coeff(self, y_true, y_pred, smooth=1e-4):
        """
        Computes the Dice coefficient, a measure of overlap between two binary masks.

        Parameters:
        y_true (tensor): Ground truth mask.
        y_pred (tensor): Predicted mask.
        smooth (float): Smoothing factor to avoid division by zero.

        Returns:
        tensor: Dice coefficient.
        """
        y_true = tf.cast(y_true, tf.float32)
        intersection = tf.keras.backend.sum(y_true * y_pred, axis=(0, 1))
        union = tf.keras.backend.sum(y_true, axis=(0, 1)) + tf.keras.backend.sum(y_pred, axis=(0, 1))
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice

    def dice_BCE_loss(self, y_true, y_pred, smooth=1e-4):
        """
        Combines Dice loss and Binary Cross Entropy (BCE) loss for training.

        Parameters:
        y_true (tensor): Ground truth mask.
        y_pred (tensor): Predicted mask.
        smooth (float): Smoothing factor to avoid division by zero.

        Returns:
        tensor: Combined Dice-BCE loss.
        """
        y_true = tf.cast(y_true, tf.float32)
        numerator = 2. * tf.keras.backend.sum(y_true * y_pred, axis=(0, 1)) + smooth
        denominator = tf.keras.backend.sum(y_true ** 2, axis=(0, 1)) + tf.keras.backend.sum(y_pred ** 2, axis=(0, 1)) + smooth
        dice_loss = 1 - tf.keras.backend.mean(numerator / denominator)
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        BCE = bce(y_true, y_pred)
        Dice_BCE = BCE + dice_loss
        return Dice_BCE

    def apply_threshold(self, full_predicted_mask, threshold):
        """
        Applies a threshold to a predicted mask to obtain a binary mask.

        Parameters:
        full_predicted_mask (numpy.ndarray): Predicted mask.
        threshold (float): Threshold value.

        Returns:
        numpy.ndarray: Binary mask after thresholding.
        """
        return (full_predicted_mask > threshold).astype(np.int8)

    def connect_trails(self, mask, kernel_size=5):
        """
        Connects disjointed trail segments in a binary mask using morphological operations.

        Parameters:
        mask (numpy.ndarray): Input binary mask.
        kernel_size (int): Size of the kernel used for morphological closing.

        Returns:
        numpy.ndarray: Mask with connected trail segments.
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def safe_pixel_to_world(self, wcs, x, y):
        """
        Converts pixel coordinates to world coordinates safely.

        Parameters:
        wcs (astropy.wcs.WCS): World Coordinate System object.
        x (int): X-coordinate in pixel space.
        y (int): Y-coordinate in pixel space.

        Returns:
        tuple: World coordinates (RA, DEC) or (nan, nan) if conversion fails.
        """
        try:
            ra_dec = wcs.pixel_to_world(x, y)
            if isinstance(ra_dec, SkyCoord):
                return ra_dec.ra.degree, ra_dec.dec.degree
            else:
                print(f"Unexpected type for ra_dec: {type(ra_dec)}")
                return np.nan, np.nan
        except Exception as e:
            print(f"Error converting pixel to world coordinates: {e}")
            return np.nan, np.nan

    def extract_local_background(self, trail_mask, image_data, dilation_size=21):
        """
        Extracts local background pixels around the trail using dilation.

        Parameters:
        trail_mask (numpy.ndarray): Binary mask of the trail (1 for trail, 0 for background)
        image_data (numpy.ndarray): Full image data
        dilation_size (int): Size of dilation for finding local background

        Returns:
        numpy.ndarray: Local background pixel values
        """
        # Dilate the trail mask to get the surrounding region
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        dilated_mask = cv2.dilate(trail_mask.astype(np.uint8), kernel,
                                  iterations=1)
        local_background_mask = (dilated_mask > 0) & (trail_mask == 0)
        local_background_pixels = image_data[local_background_mask]
        return local_background_pixels

    def estimate_width_from_contour_fitting(self, contour):
        """
        Estimates the width of an object represented by a contour by fitting a rotated rectangle
        (minimum-area bounding box) around the contour.

        Parameters:
        ----------
        contour : numpy.ndarray
            The contour of the object, typically obtained from functions like `cv2.findContours`.
            It represents the boundary of the object in pixel coordinates.

        Returns:
        -------
        width : float
            The estimated width of the object, defined as the smaller of the two dimensions (width or height)
            of the fitted bounding box around the contour.
        """

        rect = cv2.minAreaRect(contour)
        width = min(rect[1])  # The smaller dimension is the width
        return width

    def process_image(self, full_field_image, header, area_threshold=3000, unet_threshold=0.58, min_size=500, patch_size=528, save_predicted_mask=False, time_processing=False):
        """
        Processes a FITS file to detect trails using a pre-trained model and various image processing techniques.

        Parameters:
        full_field_image (numpy.ndarray): Full field image data.
        header (astropy.io.fits.Header): FITS header.
        area_threshold (int): Minimum area of contours to keep.
        unet_threshold (float): Threshold for the predicted mask.
        min_size (int): Minimum size of objects to keep.
        patch_size (int): Size of the patches for processing.
        save_predicted_mask (bool): Whether to save the predicted mask.
        time_processing (bool): Whether to time the processing steps.

        Returns:
        tuple: Binary mask, DataFrame with results, and optionally the full predicted mask.
        """
        times = {}

        if time_processing:
            start_time = time.time()

        wcs = WCS(header)
        image_name = header.get('MASKFILE', np.nan)
        fieldID = header.get('OBJECT', np.nan)
        date = header.get('DATE-OBS', np.nan)
        ra = header.get('RA-CNTR', np.nan)
        dec = header.get('DEC-CNTR', np.nan)
        image_flag = header.get('QC-FLAG', np.nan)

        if time_processing:
            start_preprocessing_time = time.time()

        image_mean = np.mean(full_field_image)
        image_std = np.std(full_field_image)
        full_field_image_standardized = (full_field_image - image_mean) / image_std


        # Initialize the full predicted mask
        full_predicted_mask = np.zeros_like(full_field_image)

        # Accumulate patches in a batch
        batch_patches = []
        batch_indices = []
        for i in range(0, full_field_image_standardized.shape[0], patch_size):
            for j in range(0, full_field_image_standardized.shape[1],
                           patch_size):
                image_patch = full_field_image_standardized[i:i + patch_size, j:j + patch_size]
                batch_patches.append(image_patch)
                batch_indices.append((i, j))

        # Convert list to numpy array
        batch_patches = np.array(batch_patches)

        if time_processing:
            times["preprocessing"] = time.time() - start_preprocessing_time

        if time_processing:
            start_prediction_time = time.time()


        # Predict on the whole batch
        batch_predictions = self.predict_on_batch(batch_patches)

        # Place predicted patches back into the full predicted mask
        for idx, (i, j) in enumerate(batch_indices):
            full_predicted_mask[i:i + patch_size, j:j + patch_size] = batch_predictions[idx]

        if time_processing:
            times["prediction"] = time.time() - start_prediction_time

        full_predicted_mask_threshold = self.apply_threshold(full_predicted_mask,
                                                             unet_threshold)
        binary_image = full_predicted_mask_threshold.astype(np.uint8)

        if time_processing:
            start_hough_time = time.time()

        hough_params = {'rho': 1, 'theta': np.pi / 180, 'threshold': 50, 'minLineLength': 100, 'maxLineGap': 250}

        # Hough Line detection with provided parameters
        lines = cv2.HoughLinesP(binary_image, hough_params['rho'], hough_params['theta'],
                                threshold=hough_params['threshold'],
                                minLineLength=hough_params['minLineLength'],
                                maxLineGap=hough_params['maxLineGap'])

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(binary_image, (x1, y1), (x2, y2), 255, thickness=1)

        if time_processing:
            times["hough_transform"] = time.time() - start_hough_time

        if time_processing:
            start_contour_time = time.time()

        binary_image = self.connect_trails(binary_image.astype(np.uint8),
                                           kernel_size=3)
        binary_image = self.remove_small_objects(binary_image,
                                                 min_size=min_size)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        results = {
            "Image": [], "FieldID": [], "ImageRA": [], "ImageDEC": [], "ImageFlag": [], "Date": [],
            "StartX": [], "StartY": [], "EndX": [], "EndY": [], "StartRA": [], "StartDEC": [],
            "EndRA": [], "EndDEC": [], "Quantile25": [], "Quantile50": [], "Quantile75": [],
            "SNR": [], "Length": [], "Width": []
        }

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < area_threshold:
                cv2.drawContours(binary_image, [contour], -1, 0,
                                 thickness=cv2.FILLED)
                continue

            mask = np.zeros_like(binary_image, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
            lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=50,
                                    minLineLength=100, maxLineGap=10)

            if lines is not None:
                lines = lines[:, 0, :]
                angles = np.array([self.calculate_line_angle(line) for line in lines])
                clustering = DBSCAN(eps=5, min_samples=1).fit(angles.reshape(-1, 1))
                labels = clustering.labels_
                unique_labels = set(labels)

                if len(unique_labels) >= 5:
                    cv2.drawContours(binary_image, [contour], -1, 0, thickness=cv2.FILLED)
                    continue

                for label in unique_labels:
                    if label == -1:
                        continue

                    cluster_lines = lines[labels == label]
                    cluster_mask = np.zeros_like(mask, dtype=np.uint8)
                    for line in cluster_lines:
                        x1, y1, x2, y2 = line
                        cv2.line(cluster_mask, (x1, y1), (x2, y2), 1,
                                 thickness=1)

                    cluster_contours, _ = cv2.findContours(cluster_mask,
                                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cl_contour in cluster_contours:
                        cl_area = cv2.contourArea(cl_contour)
                        if cl_area < area_threshold:
                            cv2.drawContours(binary_image, [cl_contour], -1, 0, thickness=cv2.FILLED)
                        else:
                            contour_line_mask = np.zeros_like(mask)
                            cv2.drawContours(contour_line_mask, [cl_contour], -1, 1, thickness=cv2.FILLED)
                            filtered_lines = [line for line in cluster_lines if np.any(
                                contour_line_mask[line[1]][line[0]] == 1 and contour_line_mask[line[3]][line[2]] == 1)]

                            if not filtered_lines:
                                continue

                            filtered_lines_array = np.array(filtered_lines)
                            start_point, end_point = self.find_trail_extremities(filtered_lines_array)
                            trail_pixels = full_field_image[contour_line_mask == 1]
                            quantiles = np.quantile(trail_pixels, [0.25, 0.5, 0.75])
                            start_ra, start_dec = self.safe_pixel_to_world(wcs, start_point[0], start_point[1])
                            end_ra, end_dec = self.safe_pixel_to_world(wcs, end_point[0], end_point[1])

                            # SNR Calculation for the trail
                            local_background_pixels = self.extract_local_background(contour_line_mask, full_field_image,
                                                                                    dilation_size=21)
                            # Filter trail pixels to exclude extreme outliers
                            low_percentile, high_percentile = np.percentile(trail_pixels, [1.5, 98.5])
                            filtered_trail_pixels = trail_pixels[
                                (trail_pixels >= low_percentile) & (trail_pixels <= high_percentile)]

                            # Filter background pixels in the same way
                            low_back_percentile, high_back_percentile = np.percentile(local_background_pixels, [1.5, 98.5])
                            filtered_back_pixels = local_background_pixels[
                                (local_background_pixels >= low_back_percentile) &
                                (local_background_pixels <= high_back_percentile)]

                            # Calculate total flux and SNR using filtered pixels
                            tot_sat_flux = np.sum(filtered_trail_pixels)
                            N_sat_pix = len(filtered_trail_pixels)
                            std_backg_flux = np.std(filtered_back_pixels)

                            # Length Estimation (Euclidean distance between start and end points)
                            length = np.sqrt(
                                (start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2)

                            # Width Estimation using the contour fitting method
                            width = self.estimate_width_from_contour_fitting(cl_contour)

                            # Calculate unit SNR
                            total_snr = (tot_sat_flux / np.sqrt(tot_sat_flux + N_sat_pix * (std_backg_flux**2))) / length

                            results["Image"].append(image_name)
                            results["FieldID"].append(fieldID)
                            results["Date"].append(date)
                            results["ImageRA"].append(ra)
                            results["ImageDEC"].append(dec)
                            results["ImageFlag"].append(image_flag)
                            results["StartX"].append(start_point[0])
                            results["StartY"].append(start_point[1])
                            results["EndX"].append(end_point[0])
                            results["EndY"].append(end_point[1])
                            results["StartRA"].append(start_ra)
                            results["StartDEC"].append(start_dec)
                            results["EndRA"].append(end_ra)
                            results["EndDEC"].append(end_dec)
                            results["Quantile25"].append(quantiles[0])
                            results["Quantile50"].append(quantiles[1])
                            results["Quantile75"].append(quantiles[2])
                            results["SNR"].append(total_snr)
                            results["Length"].append(length)
                            results["Width"].append(width)

        results_df = pd.DataFrame.from_dict(results)
        binary_image = self.remove_small_objects(binary_image,
                                                 min_size=min_size)

        if time_processing:
            times["contour_analysis"] = time.time() - start_contour_time
            times["total"] = time.time() - start_time

        if save_predicted_mask:
            return binary_image, results_df, full_predicted_mask, times if time_processing else None

        return binary_image, results_df, None, times if time_processing else None

    def save_results(self, results_df, base_filename, csv_output_dir, image_output_dir, save_mask=False,
                     mask_image=None, save_predicted_mask=False, predicted_mask=None):
        """
        Saves the processing results, including DataFrame and mask images, to specified directories.

        Parameters:
        results_df (pandas.DataFrame): DataFrame containing processing results.
        base_filename (str): Base filename for saving results.
        csv_output_dir (str): Directory to save the results CSV file.
        image_output_dir (str): Directory to save mask images.
        save_mask (bool): Whether to save the mask image.
        mask_image (numpy.ndarray): Mask image to be saved.
        save_predicted_mask (bool): Whether to save the predicted mask image.
        predicted_mask (numpy.ndarray): Predicted mask image to be saved.
        """
        os.makedirs(csv_output_dir, exist_ok=True)
        os.makedirs(image_output_dir, exist_ok=True)

        if base_filename.endswith('.fits.fz'):
            base_filename = base_filename[:-8]
        elif base_filename.endswith('.fits'):
            base_filename = base_filename[:-5]

        csv_filename = os.path.join(csv_output_dir,
                                    f"{base_filename}_results.csv")
        results_df.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")

        if save_mask and mask_image is not None:
            mask_filename = os.path.join(image_output_dir,
                                         f"{base_filename}_mask.png")
            mask_image_uint8 = (mask_image * 255).astype(np.uint8)
            cv2.imwrite(mask_filename, mask_image_uint8)
            print(f"Mask image saved to {mask_filename}")

        if save_predicted_mask and predicted_mask is not None:
            predicted_mask_filename = os.path.join(image_output_dir,
                                                   f"{base_filename}_predicted_mask.png")
            predicted_mask_uint8 = (predicted_mask * 255).astype(np.uint8)
            cv2.imwrite(predicted_mask_filename, predicted_mask_uint8)
            print(f"Predicted mask image saved to {predicted_mask_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an astronomical image to detect trails.")
    parser.add_argument("model_path", type=str, help="Path to the trained model file.")
    parser.add_argument("fits_file_path", type=str, help="Path to the FITS file to process.")
    parser.add_argument("--unet_threshold", type=float, default=0.58, help="Threshold for the predicted mask.")
    parser.add_argument("--save", action="store_true", help="Save the results DataFrame to a CSV file.")
    parser.add_argument("--save_mask", action="store_true", help="Save the mask image as a PNG file.")
    parser.add_argument("--save_predicted_mask", action="store_true", help="Save the predicted mask image as a PNG file.")
    parser.add_argument("--csv_output_dir", type=str, default=".", help="Directory to save the results CSV file.")
    parser.add_argument("--image_output_dir", type=str, default=".", help="Directory to save the mask image.")
    parser.add_argument("--time_processing", action="store_true", help="Time the processing steps.")

    args = parser.parse_args()

    processor = ASTA(args.model_path)

    if os.path.exists(args.fits_file_path):
        with fits.open(args.fits_file_path) as hdul:
            header = hdul[-1].header
            full_field_image = hdul[-1].data

        binary_img, df, predicted_mask, times = processor.process_image(
            full_field_image, header, unet_threshold=args.unet_threshold,
            save_predicted_mask=args.save_predicted_mask, time_processing=args.time_processing
        )

        if args.time_processing:
            print("Processing times:", times)

        if args.save:
            base_filename = os.path.splitext(os.path.basename(args.fits_file_path))[0]
            processor.save_results(
                df, base_filename, args.csv_output_dir, args.image_output_dir,
                save_mask=args.save_mask, mask_image=binary_img,
                save_predicted_mask=args.save_predicted_mask,
                predicted_mask=predicted_mask
            )
    else:
        print(f"FITS file {args.fits_file_path} does not exist.")
