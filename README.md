# Foam Imaging Segmentator: Advanced Bubble and Foam Analysis Toolkit

![Python](https://img.shields.io/badge/python-3.8%2B-blue)  
![MIT License](https://img.shields.io/badge/license-MIT-green)

<img align="right" src=https://github.com/fravij99/Foam_Segmentation/blob/master/datasets_and_results/above_sight/schiumaweb.jpg  width="270">

## Introduction

Foam Imaging Segmentator is a Python-based toolkit for advanced image segmentation of foam structures, with applications ranging from the analysis of sparkling wine bubbles to modeling foam dynamics using statistical mechanics.

The project focuses on segmenting and analyzing bubble structures to extract meaningful metrics such as:
- **Average bubble size** with hyperspectral camera acquiring images from above the glass. 
- **Temporal evolution of foam** with hyperspectral camera acquiring images from the side of the glass.
- **Foam relaxation time**, modeled using fractal dimension and heigth progression of the foam. 

This work bridges the gap between image processing and physical modeling, offering a versatile tool for both researchers and industry professionals.

---

## Table of Contents
- [Features](#features)
- [Requirements](#Requirements)
- [Installation](#installation)
- [Code Structure](#code-structure)
- [Usage](#Usage)
- [Results](#results)
- [Performance](#performance)
- [License](#license)
- [Contact me](#contact-me)

---

## Features

- **Multi-approach segmentation**: Gradient-based and fractal-based analysis.
- **Foam dynamic analysis**: Measure foam height and temporal evolution with automated tools.
- **Scalable processing**: Handles high-resolution images and large datasets efficiently.
- **Customizable**: Easily adaptable to various foam analysis scenarios.

---

## Requirements
To use this library, ensure you have the required Python packages installed:

```pip install numpy opencv-python scikit-image matplotlib scipy os tqdm```

---

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/fravij99/Foam_Segmentation.git
cd Foam_Segmentation
pip install numpy opencv-python scikit-image matplotlib scipy os tqdm
```

---

## Code Structure
### classic_segmentator
- `detecting_glass(self)`: Apply the first processing of the image to reduce it to the ROI intrested for the analysis. In my case I used circle recogniton to center and full comprehend the glass containing liquid. 
- `image_segmentation(self, img, filter, threshold)`: Perform image segmentation.
- `filtering_counting_bubbles(self, minimum, props)`: Filter and count bubbles by size.
- `plotting_circles(self, img, props, diameters, min_diameter)`: Plot circles around detected bubbles.
- `computing_fractal_dimension(self, binary_image, min_box_size, max_box_size)`: Calculate fractal dimension using the box-counting method.
- `fractal_dimension_fit(self, fractal_dimension, box_sizes, box_counts)`: Visualize the fit of the fractal dimension calculation.
- `calculate_scaling_exponent(self, diameters, output_excel_path)`: calculates the Scaling Exponent of a log-log fit on the bubbble diameters distribution. It writes a .xlsx file containing the exponent computed for every frame.

### fractal_segmentator
- `fractal_dimension(self, image_segment)`: Calculate the fractal dimension of a segment.
- `refine_segmentation_with_fractal(self, image, threshold)`: Refine segmentation based on fractal dimension analysis.

### binarizer
- `median_filter(self, contrast)`: Apply a median filter.
- `gaussian_filter(self, contrast)`: Apply a Gaussian filter.
- `threshold_otsu(self, smoothed)`: Apply Otsu's thresholding.
- `threshold_adaptive(self, smoothed)`: Apply adaptive thresholding.
- `process_image(self, image)`: Implements different methods for the image preprocessing as: applying median and clahe filter, Otsu threshold and removing small objects.
- `binarize_folder(self, image)`: Implements `process_image` function and saves the binarized images in a folder.

### heigth_measurer
- `select_roi(self, image)`: Shows the last and the first binarized image of the folder and allows the user to select the ROI containing the foam layer by graphic interface. Just use the mouse to select ROI, click and it sves the box vertices.
- `measure_foam_heigth(self, frame, bounding_box)`: Counts the consecutive null rows of a given column. The foam heigth is measured passing the ROI from the user. The method splits the ROI in contigous columns containing 1 and 0, counting the consecutive 0. 
- `foam_progression_plot(self, images, bounding_box)`: Implements the previous function in order to give a guess of the mean of the foam heigth. Secondly it implements an exp/arctan fit of the foam progression in time and gives the main parameters of the function that fits best building a ghraph. 

## Usage
The `bubble.py`, `foam_heigth.py`, `detecting_ROI.py` and `bubble_size.py` scripts contain a brief implementation of the main functionalities of the library. The scripts are implemented in order to make a multiple segmentation of all the images (frames if you have a video utput like me) of all the subfolders comntaioned in a specific main folder. 

The implementation is pretty well authomatized, so the user has to change only few parameters to start the segmentation. The `heigth_measurer` class has a very fast running, analysing a 100 frames (1250x1080 pixels resoluted) folder in less than 20 seconds. Whereas the `classic_segmentator` class methods result very slow, with a 100 frames folder (2500x2160 pixels resolution) processed in almost 20 minutes (it depends also on the device performance).

---

## Results

<img align="right" src=https://github.com/fravij99/Foam_Segmentation/blob/master/datasets_and_results/above_sight/frameAV_077.jpg  width="270">
<img align="right" src=https://github.com/fravij99/Foam_Segmentation/blob/master/datasets_and_results/above_sight/binarization/frameAV_077.jpg  width="270">

<img align="right" src=https://github.com/fravij99/Foam_Segmentation/blob/master/datasets_and_results/above_sight/segmentation/frameAV_077.jpg  width="270">

<img align="right" src=https://github.com/fravij99/Foam_Segmentation/blob/master/datasets_and_results/above_sight/fractal_fit/frameAV_077.jpg  width="270">

<img align="right" src=https://github.com/fravij99/Foam_Segmentation/blob/master/datasets_and_results/above_sight/scaling_fit/frameAV_077.jpg_scaling_fit.png  width="270">

---

## License
This project is licensed under the MIT License.

---

## Contact me

If you have any doubt or issue, please feel free to contact me at fravilla30@gmail.com, francesco.villa@unimi.it. 