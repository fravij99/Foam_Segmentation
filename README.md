# Foam Imaging Segmentator: Advanced Bubble and Foam Analysis Toolkit

![Python](https://img.shields.io/badge/python-3.8%2B-blue)  
![MIT License](https://img.shields.io/badge/license-MIT-green)

<table>
<tr>
    <td align="center">
        <img src="https://raw.githubusercontent.com/fravij99/Foam_Segmentation/master/datasets_and_results/above_sight/schiumaweb.jpg" width="400" alt="Original Foam Image">
        <p><strong>Original Foam Image</strong></p>
    </td>
    <td align="center">
        <img src="https://raw.githubusercontent.com/fravij99/Foam_Segmentation/master/datasets_and_results/above_sight/segmentation/schiumaweb.jpg" width="435" alt="Segmented Foam Image">
        <p><strong>Segmented Foam Image</strong></p>
    </td>
</tr>
</table>


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
- `detecting_glass(self)`: Applies the first processing of the image to reduce it to the ROI intrested for the analysis. In my case I used circle recogniton to center and full comprehend the glass containing liquid. 
- `image_segmentation(self, img, filter, threshold)`: Performs image segmentation.
- `filtering_counting_bubbles(self, minimum, props)`: Filters and count bubbles by size.
- `plotting_circles(self, img, props, diameters, min_diameter)`: Plots circles around detected bubbles.
- `computing_fractal_dimension(self, binary_image, min_box_size, max_box_size)`: Calculates fractal dimension using the box-counting method.
- `fractal_dimension_fit(self, fractal_dimension, box_sizes, box_counts)`: Visualizes the fit of the fractal dimension calculation.
- `calculate_scaling_exponent(self, diameters, output_excel_path)`: Calculates the Scaling Exponent of a log-log fit on the bubbble diameters distribution. It writes a .xlsx file containing the exponent computed for every frame.

### fractal_segmentator
- `fractal_dimension(self, image_segment)`: Calculates the fractal dimension of a segment.
- `refine_segmentation_with_fractal(self, image, threshold)`: Refines segmentation based on fractal dimension analysis.

### binarizer
- `median_filter(self, contrast)`: Applies a median filter.
- `gaussian_filter(self, contrast)`: Applies a Gaussian filter.
- `threshold_otsu(self, smoothed)`: Applies Otsu's thresholding.
- `threshold_adaptive(self, smoothed)`: Applies adaptive thresholding.
- `process_image(self, image)`: Implements different methods for the image preprocessing as: applying median and clahe filter, Otsu threshold and removing small objects.
- `binarize_folder(self, image)`: Implements `process_image` function and saves the binarized images in a folder.

### heigth_measurer
- `select_roi(self, image)`: Shows the last and the first binarized image of the folder and allows the user to select the ROI containing the foam layer by graphic interface. Just use the mouse to select ROI, click and it sves the box vertices.
- `measure_foam_heigth(self, frame, bounding_box)`: Counts the consecutive null rows of a given column. The foam heigth is measured passing the ROI from the user. The method splits the ROI in contigous columns containing 1 and 0, counting the consecutive 0. 
- `foam_progression_plot(self, images, bounding_box)`: Implements the previous function in order to give a guess of the mean of the foam heigth. Secondly it implements an exp/arctan fit of the foam progression in time and gives the main parameters of the function that fits best, building a ghraph. 

## Usage
The `bubble.py`, `foam_heigth.py`, `detecting_ROI.py` and `bubble_size.py` scripts contain a brief implementation of the main functionalities of the library. These scripts are implemented in order to make a multiple segmentation of all the images (frames if you have a video utput like me) of all the subfolders contained in a specific main folder. 

The implementation is pretty well authomatized, so the user has to change only few parameters to start the segmentation. The `heigth_measurer` class has a very fast running, analysing a 100 frames (1250x1080 pixels resoluted) folder in less than 20 seconds. Whereas the `classic_segmentator` class methods result slower, with a 100 frames folder (2500x2160 pixels resolution) processed in almost 15 minutes (it depends also on the device performance).

---

## Results

### Classic Segmentator

The `classic_segmentator` class processes foam images from a top-down perspective, applying various steps such as binarization, segmentation, fractal analysis, and scaling fits. Below are some examples of the results:

| **Original Image** | **Binarized Image** | **Segmented Image** | 
|-------------------|----------------------------------------------|-------------------------------|
![Original](https://raw.githubusercontent.com/fravij99/Foam_Segmentation/master/datasets_and_results/above_sight/frameAV_077.jpg) | ![Binarized](https://raw.githubusercontent.com/fravij99/Foam_Segmentation/master/datasets_and_results/above_sight/binarization/frameAV_077.jpg) | ![Segmented](https://raw.githubusercontent.com/fravij99/Foam_Segmentation/master/datasets_and_results/above_sight/segmentation/frameAV_077.jpg) | 

|**Fractal Fit** | **Scaling Fit** |
|-----------------------------------|-------------------------------------------------------------|
|![Fractal Fit](https://raw.githubusercontent.com/fravij99/Foam_Segmentation/master/datasets_and_results/above_sight/fractal_fit/frameAV_077.jpg) | ![Scaling Fit](https://raw.githubusercontent.com/fravij99/Foam_Segmentation/master/datasets_and_results/above_sight/scaling_fit/frameAV_077.jpg_scaling_fit.png) |

---


### Height Measurer

The `heigth_measurer` class focuses on foam height analysis using side-view images. It performs binarization, foam height measurement, and temporal progression fitting. Below are some examples:

| **Original Image** | **Binarized Image** | **Arctan Fit** |
|---------------------|---------------------|----------------|
| ![Original](https://raw.githubusercontent.com/fravij99/Foam_Segmentation/master/datasets_and_results/side_sight/frameIDS_055.jpg) | ![Binarized](https://raw.githubusercontent.com/fravij99/Foam_Segmentation/master/datasets_and_results/side_sight/binarization/frameIDS_055.jpg) | ![Arctan Fit](https://raw.githubusercontent.com/fravij99/Foam_Segmentation/master/datasets_and_results/side_sight/arctg_fit.png) |

### Explanation of Results

1. **Classic Segmentator**:

The bubbles segmentation presents some points of strength:
   - Robust image centering and binarization.
   - Only analytical methods, no machine learning methods involved.
   - Deterministic methods.
   - No training of the models, less computing time. 
   - Scaling fits provide additional statistical insights into bubble distribution.
   - This tool can be used especially to investigate foam properties and prove statistical mechanics models. 

Unfortunatly it presents also some weaknesses:
   - Large number of the models parameters. 
   - It's a gradient-based model, so its performance is strictly related to image resolution, brightness contrast ecc.. (In facts methods to improve contrast and other features are implemented).
   - Non parallelized code, need of a performing CPU.
   - No implementation for GPU.

2. **Height Measurer**:

The precision of the foam heigth computing is quite impressive and it can be seen in the graph of the *arctan* fit, where the error bars are smaller than dots. This precision can be explained by:
   - Graphic interface for ROI detection, where the human interpretation can avoid the region of the images infected by optical noising phenomenons. 
   - The binarization is robust.
   - The computing of the foam heigth is exact (no numerical simulation or minimization needed), for every pixel column the contugous 0 values are counted.

---

## License
This project is licensed under the MIT License.

---

## Contact me

If you have any doubt or issue, please feel free to contact me at fravilla30@gmail.com, francesco.villa@unimi.it. 