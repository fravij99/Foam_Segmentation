# Foam imaging segmentator

<img align="right" src=https://github.com/fravij99/Foam_Segmentation/blob/master/demo_images_segmented/bubbles_perfect.png  width="270">

Hello everyone! This repository deals with the image segmemtation of different foam structures. The main dataset is coming from hyperspectral acquisitions of sparkling wine. The aim of the wotrk consists in segmenting bubbles from different points of view in order to extract relevant features as average bubble size or the temporal evolution of the foam. Another relevat parameter could be represented by the relaxing time of the foam, where the study of the bubbles dynamics could be modeled by statistical mechanics (probably percolation).

## Code Structure
This work includes:
- Segmentator using gradient analysis
- Segmentator using Fractal analysis
- Dataset
- A numerical simulator of foams images to the model evaluation

## Requirements
To use this library, ensure you have the required Python packages installed:

```pip install numpy opencv-python scikit-image matplotlib scipy```

## Classes and Methods
### classic_segmentator
- `median_filter(self, contrast)`: Apply a median filter.
- `gaussian_filter(self, contrast)`: Apply a Gaussian filter.
- `threshold_otsu(self, smoothed)`: Apply Otsu's thresholding.
- `threshold_adaptive(self, smoothed)`: Apply adaptive thresholding.
- `image_segmentation(self, img, filter, threshold)`: Perform image segmentation.
- `filtering_counting_bubbles(self, minimum, props)`: Filter and count bubbles by size.
- `plotting_circles(self, img, props, diameters, min_diameter)`: Plot circles around detected bubbles.
- `computing_fractal_dimension(self, binary_image, min_box_size, max_box_size)`: Calculate fractal dimension using the box-counting method.
- `fractal_dimension_fit(self, fractal_dimension, box_sizes, box_counts)`: Visualize the fit of the fractal dimension calculation.

### fractal_segmentator
- `fractal_dimension(self, image_segment)`: Calculate the fractal dimension of a segment.
- `refine_segmentation_with_fractal(self, image, threshold)`: Refine segmentation based on fractal dimension analysis.

## Segmemtations

<img align="right" src=https://github.com/fravij99/Foam_Segmentation/blob/master/demo_images_segmented/schiumaweb.jpg  width="270">
<img align="right" src=https://github.com/fravij99/Foam_Segmentation/blob/master/demo_images_segmented/segmentation1.png  width="270">
<img align="right" src=https://github.com/fravij99/Foam_Segmentation/blob/master/demo_images_segmented/fractal_fit.png  width="270">
<img align="right" src=https://github.com/fravij99/Foam_Segmentation/blob/master/demo_images_segmented/bubbles_perfect.png  width="270">


## License
This project is licensed under the MIT License.