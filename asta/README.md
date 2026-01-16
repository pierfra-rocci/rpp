![ASTA Title Image](https://see.fontimg.com/api/renderfont4/KpAp/eyJyIjoiZnMiLCJoIjoxMTksInciOjEwMDAsImZzIjoxMTksImZnYyI6IiM4QTAyMDIiLCJiZ2MiOiIjRkZGRkZGIiwidCI6MX0/QVNUQQ/kg-second-chances-sketch.png)

# Automated Satellite Tracking for Astronomy


ASTA is a tool designed for detecting and analyzing astronomical trails in images using deep learning and image processing techniques.

## Features

- Detects trails in astronomical images
- Uses a pre-trained model for image segmentation
- Converts pixel coordinates to world coordinates (RA, DEC)
- Outputs results as CSV and mask images

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/FiorenSt/asta.git
    cd asta
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Run the ASTA tool from the command line to process astronomical images:

```sh
python asta.py <model_path> <fits_file_path> [options]
```

### Options:

- `--unet_threshold <float>`: Set the threshold for the U-Net predicted mask (default: 0.5).
- `--save`: Save the results DataFrame to a CSV file.
- `--save_mask`: Save the binary mask image as a PNG file.
- `--save_predicted_mask`: Save the predicted mask image as a PNG file.
- `--csv_output_dir <dir>`: Directory to save the results CSV file (default: current directory).
- `--image_output_dir <dir>`: Directory to save mask images (default: current directory).
- `--time_processing`: Output processing time for each step.


### Example

```sh
python asta.py model.h5 sample.fits --unet_threshold 0.75 --save --save_mask --save_predicted_mask --csv_output_dir results/csv --image_output_dir results/images
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License.

## Contact

For any questions or inquiries, please contact [f.stoppa@astro.ru.nl].
