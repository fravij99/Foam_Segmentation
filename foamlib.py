import numpy as np
import cv2
import os
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial import distance_matrix
import random
from scipy.optimize import curve_fit
from tqdm import tqdm
from matplotlib.widgets import RectangleSelector

class classic_segmentator:

    def __init__(self, origin_folder, root, filename):
        self.root = root
        self.origin_folder = origin_folder
        complete_path = os.path.join(origin_folder, root, filename)
        self.img = cv2.imread(complete_path)
        self.filename = filename

    def detecting_glass(self):

        gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped_image = self.img[y:y+h, x:x+w]
            self.img = cropped_image
            '''plt.imshow(self.img)
            plt.show()'''
            
        else:
            print("No contour detected")
            

    def median_filter(self, contrast):
        return cv2.medianBlur(contrast, 7)
    

    def gaussian_filter(self, contrast):
        return cv2.GaussianBlur(contrast, (5, 5), 0)
    

    def threshold_otsu(self, smoothed):
        return cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] #Using OTSU iterative thresholdindg method
    

    def threshold_adaptive(self, smoothed):
        return cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 2)


    def image_segmentation(self, filter, threshold):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # Contrast and noise 
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        smoothed = filter(contrast)
        # thresholding
        thresh_val = threshold(smoothed) #Using OTSU iterative thresholdindg method
        binary = cv2.bitwise_not(thresh_val)  
        binary = morphology.remove_small_objects(binary.astype(bool), min_size=50).astype(np.uint8)
        output_path = os.path.join(self.origin_folder, self.root, 'binarization')
        os.makedirs(output_path, exist_ok=True)
        plt.imsave(f'{self.root}/binarization/{self.filename}', cmap='gray', arr=binary, dpi=300)
        # Masking
        labels = measure.label(binary, connectivity=2)
        props = measure.regionprops(labels)   
        
        return binary, labels, props  


    
    """Computing fractal dimension using box-countig method
        
        Parameters:
            binary_image (ndarray)
            min_box_size (int)
            max_box_size (int)
            
        Returns:
            (float, list, list): The fractal dimension of a frame and the data for the plot."""
    def computing_fractal_dimension(self, binary_image, min_box_size=2, max_box_size=1000):
        box_sizes = []
        box_counts = []
        
        # Box counting
        for box_size in range(min_box_size, max_box_size, 2):
            # Dividing image into a grid (box_size X box_size)
            count = 0
            for y in range(0, binary_image.shape[0], box_size):
                for x in range(0, binary_image.shape[1], box_size):
                    # If a pixel is white, it's a bubble
                    if np.any(binary_image[y:y + box_size, x:x + box_size]):
                        count += 1
            box_sizes.append(box_size)
            box_counts.append(count)
        
        # Fractal dimension with least square method
        coeffs = np.polyfit(np.log(box_sizes), np.log(box_counts), 1)
        fractal_dimension = -coeffs[0]

        return fractal_dimension, box_sizes, box_counts


    def calculate_correlation_with_neighbors(self, props):
        # centroids and diametres of the bubbles
        centroids = np.array([prop.centroid for prop in props])
        diameters = np.array([prop.equivalent_diameter for prop in props])
        
        if len(diameters) == 0:
            print("No bubbles detected.")
            return None

        dist_matrix = distance_matrix(centroids, centroids)
        correlations = []
        
        for i, D in enumerate(diameters):
            # Indices of the nearest neighbours 
            neighbors = np.where((dist_matrix[i] <= 5 * D) & (dist_matrix[i] > 0))[0]  

            if len(neighbors) > 0:
                neighbor_diameters = diameters[neighbors]
                # correlation between diameters and the nearest neighbours
                correlation = np.corrcoef(np.full(len(neighbor_diameters), D), neighbor_diameters)[0, 1]

                correlations.append(correlation)

        # Mean of the correlations
        mean_correlation = np.nanmean(correlations)  # NAN filtering
        return mean_correlation
    

    def saving_statistical_data(self, fractal_dimension, min_diameter, max_diameter, props, excel_path):

         # Filtering bubbles based on diameter and eccentricity
        diameters = [prop.equivalent_diameter for prop in props 
                    if prop.equivalent_diameter > min_diameter 
                    and prop.equivalent_diameter < max_diameter]

        if len(diameters) == 0:
            print("No bubbles detected.")
            return []

        mean_diameter = np.mean(diameters)
        min_diam = min(diameters)
        max_diam = max(diameters)
        median_diam = np.median(diameters)
        std_diam = np.std(diameters)

        # excel data
        data = {
            'Filename': [self.filename],
            'Bubbles number': [len(diameters)],
            'Fractal Dimension': [fractal_dimension],
            'Minimum Diameter (pixels)': [min_diam],
            'Maximum Diameter (pixels)': [max_diam],
            'Median Diameter (pixels)': [median_diam],
            'Average Diameter (pixels)': [mean_diameter],
            'Variance (pixels)': [std_diam], 
            'Radius (cm)': 8, 
            'Resolution (pixels)': 2500, 
            'Radius bubbles (cm)': np.array([mean_diameter])/2500 * 8, 
            'Average correlation': 0 #self.calculate_correlation_with_neighbors(props)
        }

        if excel_path:
            
            df = pd.DataFrame(data)
            try:
                with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                    df.to_excel(writer, sheet_name='Bubble Data', index=False, header=writer.sheets.get('Bubble Data') is None, startrow=writer.sheets['Bubble Data'].max_row if 'Bubble Data' in writer.sheets else 0)
            except FileNotFoundError:
                
                df.to_excel(excel_path, sheet_name='Bubble Data', index=False)
        return diameters


    def fractal_dimension_fit(self, fractal_dimension, box_sizes, box_counts):
        output_path = os.path.join(self.origin_folder, self.root, 'fractal_fit')
        os.makedirs(output_path, exist_ok=True)
        sns.set_style('darkgrid')

        # Fractal dimension fit
        plt.figure(figsize=(8, 6))
        plt.plot(np.log(box_sizes), np.log(box_counts), 'bo-', label='Box-counting data')
        plt.plot(np.log(box_sizes), np.polyval(np.polyfit(np.log(box_sizes), np.log(box_counts), 1), np.log(box_sizes)), 'r--', label=f'Fit: Slope = {fractal_dimension:.2f}')
        plt.title('Box-counting & frantal dimension')
        plt.xlabel('Log(Box size)')
        plt.ylabel('Log(Box count)')
        plt.legend()
        #plt.savefig(f'{self.root}/fractal_fit/{self.filename}')
        plt.close()


    def plotting_circles(self, props, diameters, min_diameter, max_diameter):
        fig, ax = plt.subplots()
        ax.imshow(self.img)
        ax.set_title(f'Average bubble diameter: {np.mean(diameters):.2f} pixels')
        # Drawing circles
        output_path = os.path.join(self.origin_folder, self.root, 'segmentation')
        os.makedirs(output_path, exist_ok=True)
        for bubble in props:
            if bubble.equivalent_diameter > min_diameter and bubble.equivalent_diameter < max_diameter:
                y, x = bubble.centroid
                radius = bubble.equivalent_diameter / 2
                circ = plt.Circle((x, y), radius, color='r', fill=False, linewidth=0.7)
                ax.add_patch(circ)
        plt.savefig(f'{self.root}/segmentation/{self.filename}', dpi=300)
        plt.close(fig)



class fractal_segmentator:
    def fractal_dimension(self, image_segment):
        """
        Fractal dimension using box counting
        """
        def box_count(img, box_size):
            S = np.add.reduceat(
                np.add.reduceat(img, np.arange(0, img.shape[0], box_size), axis=0),
                                np.arange(0, img.shape[1], box_size), axis=1)
            return len(np.where((S > 0) & (S < box_size**2))[0])

        sizes = np.logspace(1, np.log2(min(image_segment.shape)), num=10, base=2, dtype=int)
        counts = []

        for size in sizes:
            count = box_count(image_segment, size)
            counts.append(count if count > 0 else 1)  # Avoiding zeros

        log_sizes = np.log(sizes)
        log_counts = np.log(counts)

        coeffs = np.polyfit(log_sizes, log_counts, 1)
        return -coeffs[0]

    def refine_segmentation_with_fractal(self, image, threshold=0.08):
        """
        IMproves segmentation using fractal dimension tecnique
        """
        thresh_val = threshold_otsu(image)
        binary_segment = image > thresh_val
        
        binary_segment = morphology.remove_small_objects(binary_segment, 30)
        binary_segment = binary_fill_holes(binary_segment)
        
        labeled_segments = measure.label(binary_segment)
        
        refined_mask = np.zeros_like(image, dtype=bool)
    
        print("Fractal dimension for the segment:")
        
        for region in measure.regionprops(labeled_segments):
            if region.area >= 50:
                minr, minc, maxr, maxc = region.bbox
                segment = binary_segment[minr:maxr, minc:maxc]
                segment_outline = morphology.binary_dilation(segment) ^ segment
                
                fractal_dim = self.fractal_dimension(segment_outline)
                print(f"Frantal dimension of the segment ({minr}, {minc}) - ({maxr}, {maxc}): {fractal_dim}")
                
                # filtering for fractal dimenasion
                if fractal_dim > threshold:
                    refined_mask[minr:maxr, minc:maxc] |= segment

        return refined_mask
    

class heigth_measurer:

    def __init__(self, root):
        self.root = root
        self.roi_coords = []  # Saving ROI coords

    def select_roi(self, image):

        def onselect(eclick, erelease):
            """
            Callback for the ROI selection.
            Saves the coordinates of the ROI
            """
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            self.roi_coords = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            print(f"ROI selected: {self.roi_coords}")
            plt.close()  

        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.set_title("Select the ROI")

        toggle_selector = RectangleSelector(
            ax, onselect, drawtype='box', useblit=True,
            button=[1],  # Left click
            minspanx=5, minspany=5,  # minimum costrain on the area
            spancoords='pixels', interactive=True
        )
        plt.show()
        return self.roi_coords
        
    """
    Measures the heigth of the foam of a binarized image
    The foam is seen as contiguous null rows 
    """
    def measure_foam_heigth(self, frame, bounding_box):

        x, y, w, h = bounding_box
        roi = frame[y:y+h, x:x+w]
        
        # Counts null raws
        row_sums = np.sum(roi == 0, axis=1)  # Black pixels per column
        foam_rows = np.where(row_sums == w)[0]  # find black rows
        
        if foam_rows.size > 0:
            foam_height = len(foam_rows)  # number of contiguous black raws
        else:
            foam_height = 0 
        
        return foam_height


    def foam_progression_plot(self, images, start_x, end_x, start_y, end_y):
        """
        Calculate and plot the temporal evolution of foam height deterministically.
        Handles outliers in the column heights by weighting them to reduce their impact on the statistics.
        Allows fitting with either an arctangent or an exponential function, choosing the best fit.
        """
        # List to store mean heights and standard deviations for all images
        column_heights = []

        # Loop through all images
        for frame in images:
            # Initialize a list to store the height of each column for the current frame
            heights = []

            # Iterate over all columns in the ROI
            for x in range(start_x, end_x):
                bounding_box = (x, start_y, 1, end_y - start_y)  # 1-pixel wide column
                foam_height = self.measure_foam_heigth(frame, bounding_box)
                heights.append(foam_height)

            # Calculate mean and standard deviation of heights
            mean_height = np.mean(heights)
            std_dev = np.std(heights)

            # Define thresholds for outliers
            upper_threshold = mean_height + 1.5 * std_dev
            lower_threshold = mean_height - 1.5 * std_dev

            # Apply weights to outliers
            weighted_heights = []
            for h in heights:
                if h > upper_threshold:  # Outlier above
                    weighted_heights.append(upper_threshold + (h - upper_threshold) * 0.25)
                elif h < lower_threshold:  # Outlier below
                    weighted_heights.append(lower_threshold + (h - lower_threshold) * 0.25)
                else:  # Inliers
                    weighted_heights.append(h)

            # Calculate weighted mean and standard deviation
            weighted_mean = np.mean(weighted_heights)
            weighted_std_dev = np.std(weighted_heights)

            # Store results
            column_heights.append((weighted_mean, weighted_std_dev))

        # Separate means and standard deviations for plotting
        means = [h[0] for h in column_heights]
        std_devs = [h[1] for h in column_heights]

        # Fit arctan function
        def arc_func(t, a, b, c, d):
            return -a * np.arctan(b * t + c) + d

        # Fit exponential function
        def exp_func(t, a, b, c):
            return a * np.exp(-b * t) + c

        time_indices = np.arange(len(images))

        # Initial parameters for both fits
        arc_initial_params = [1, 0.01, 0, 1]
        exp_initial_params = [1, 0.01, 1]

        # Perform fits
        arc_popt, _ = curve_fit(arc_func, time_indices, means, p0=arc_initial_params, maxfev=10000)
        exp_popt, _ = curve_fit(exp_func, time_indices, means, p0=exp_initial_params, maxfev=10000)

        # Calculate fitted values
        arc_fit_values = arc_func(time_indices, *arc_popt)
        exp_fit_values = exp_func(time_indices, *exp_popt)

        # Calculate RMSE for each fit
        arc_rmse = np.sqrt(np.mean((means - arc_fit_values) ** 2))
        exp_rmse = np.sqrt(np.mean((means - exp_fit_values) ** 2))

        # Choose the best fit
        if arc_rmse < exp_rmse:
            best_fit = 'arc'
            best_fit_values = arc_fit_values
            best_popt = arc_popt
            fit_label = f'Arctan fit: a={arc_popt[0]:.2f}, b={arc_popt[1]:.4f}, c={arc_popt[2]:.2f}, d={arc_popt[3]:.2f}'
        else:
            best_fit = 'exp'
            best_fit_values = exp_fit_values
            best_popt = exp_popt
            fit_label = f'Exponential fit: a={exp_popt[0]:.2f}, b={exp_popt[1]:.4f}, c={exp_popt[2]:.2f}'

        # Plot the results
        plt.figure(figsize=(12, 6))

        # Errorbar plot
        plt.errorbar(
            time_indices,
            means,
            yerr=std_devs,
            fmt='o',
            color='lightblue',
            ecolor='black',
            markeredgecolor='black',
            elinewidth=1,
            capsize=3,
            label='Column average'
        )

        plt.plot(
            time_indices,
            best_fit_values,
            color='red',
            linestyle='--',
            label=fit_label
        )

        plt.xlabel('Frames')
        plt.ylabel('Foam height (pixel)')
        plt.title(f'Foam evolution - Best fit: {best_fit}')
        plt.legend()
        plt.savefig(self.root + f'curve_fit_{best_fit}.png', dpi=300)
        plt.show()


        self.save_foam_data_to_excel(means, std_devs)




    def save_foam_data_to_excel(self, means, std_devs, filename='foam_data.xlsx'):
        """
        Save foam height means and standard deviations to an Excel file.

        Parameters:
            means (list): List of mean foam heights.
            std_devs (list): List of standard deviations of foam heights.
            filename (str): Name of the Excel file to save.
        """
        # Create a DataFrame from the data
        data = {
            'Frame': np.arange(len(means)) + 1,  # Frame numbers start from 1
            'Mean Height (pixels)': means,
            'Standard Deviation (pixels)': std_devs
        }
        df = pd.DataFrame(data)

        # Save to Excel
        output_path = self.root + filename
        df.to_excel(output_path, index=False, sheet_name='Foam Data')
        print(f"Data saved to {output_path}")



class Binarizer:
    def __init__(self, origin_folder, root, filename):
        self.root = root
        self.origin_folder = origin_folder
        complete_path = os.path.join(origin_folder, root, filename)
        self.img = cv2.imread(complete_path)
        self.filename = filename

    @staticmethod
    def apply_clahe(image):
        """
        Applied CLAHE filter to improve contrast
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    @staticmethod
    def median_filter(image):
        """
        Applies median filter
        """
        return cv2.medianBlur(image, 7)

    @staticmethod
    def threshold_otsu(image):
        """
        Otsu treshold
        """
        return cv2.bitwise_not(cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
                  

    @staticmethod
    def clean_binary_image(binary_image):
        """
        Removes small objects
        """
        return morphology.remove_small_objects(binary_image.astype(bool), min_size=50).astype(np.uint8)

    def process_image(self, image):
        """
        Biarizes image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        contrast = self.apply_clahe(gray)
        smoothed = self.median_filter(contrast)
        binary = self.threshold_otsu(smoothed)
        cleaned_binary = self.clean_binary_image(binary)
        
        return cleaned_binary

    def binarize_folder(self):
         
        # Binarizes image
        binary_image = self.process_image(self.img)
                    
        output_path = os.path.join(self.origin_folder, self.root, 'binarization')
        os.makedirs(output_path, exist_ok=True)
        plt.imsave(f'{self.root}/binarization/{self.filename}', arr=binary_image, cmap='gray', dpi=300)
                    
        return binary_image
