import numpy as np
import cv2
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt


class classic_segmentator:


    def median_filter(self, contrast):
        return cv2.medianBlur(contrast, 7)
    
    def gaussian_filter(self, contrast):
        return cv2.GaussianBlur(contrast, (5, 5), 0)
    
    def threshold_otsu(self, smoothed):
        return cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] #Using OTSU iterative thresholdindg method
    
    def threshold_adaptive(self, smoothed):
        return cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 2)


    def image_segmentation(self, img, filter, threshold):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Contrast and noise 
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        smoothed = filter(contrast)
        # thresholding
        thresh_val = threshold(smoothed) #Using OTSU iterative thresholdindg method
        binary = cv2.bitwise_not(thresh_val)  
        binary = morphology.remove_small_objects(binary.astype(bool), min_size=50).astype(np.uint8)
        plt.imshow(binary, cmap='gray')
        plt.show()
        # Masking
        labels = measure.label(binary, connectivity=2)
        props = measure.regionprops(labels)   
        return binary, labels, props  


    def filtering_counting_bubbles(self, minimum, props):
        # Filtering small buubbles
        min_diameter = minimum  
        diameters = [prop.equivalent_diameter for prop in props if prop.equivalent_diameter > min_diameter]
        mean_diameter = np.mean(diameters)
        print(f'Minimum diameter: {min(diameters)} pixels')
        print(f'Maximum diameter: {max(diameters)} pixels')
        print(f'Median diameter: {np.median(diameters)} pixels')
        print(f'Average diameter with variance: {mean_diameter} +- {np.std(diameters)} pixels')
        return diameters

    def plotting_circles(self, img, props, diameters, min_diameter):
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(f'Average bubble diameter: {np.mean(diameters):.2f} pixels')
        # Drawing circles
        for bubble in props:
            if bubble.equivalent_diameter > min_diameter:  
                y, x = bubble.centroid
                radius = bubble.equivalent_diameter / 2
                circ = plt.Circle((x, y), radius, color='r', fill=False, linewidth=0.7)
                ax.add_patch(circ)
        plt.show()

    """Calcola la dimensione frattale usando il metodo del box-counting e ritorna i dati
        per il grafico.
        
        Parameters:
            binary_image (ndarray): Immagine binaria con le bolle segmentate.
            min_box_size (int): Dimensione minima del box.
            max_box_size (int): Dimensione massima del box.
            
        Returns:
            (float, list, list): La dimensione frattale e i dati per il grafico."""
    def computing_fractal_dimension(self, binary_image, min_box_size=2, max_box_size=100):

        box_sizes = []
        box_counts = []
        # Box counting
        for box_size in range(min_box_size, max_box_size, 2):
            # Divinìding image into a grid (box_size X box_size)
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


    def fractal_dimension_fit(self, fractal_dimension, box_sizes, box_counts):
        print(f'Dimensione frattale delle bolle: {fractal_dimension:.2f}')

        # Fractal dimension fit
        plt.figure(figsize=(8, 6))
        plt.plot(np.log(box_sizes), np.log(box_counts), 'bo-', label='Box-counting data')
        plt.plot(np.log(box_sizes), np.polyval(np.polyfit(np.log(box_sizes), np.log(box_counts), 1), np.log(box_sizes)), 'r--', label=f'Fit: Slope = {fractal_dimension:.2f}')
        plt.title('Box-counting e Dimensione Frattale')
        plt.xlabel('Log(Box size)')
        plt.ylabel('Log(Box count)')
        plt.legend()
        plt.show()



class fractal_segmentator:
    def fractal_dimension(self, image_segment):
        """
        Calcola la dimensione frattale di un segmento usando il metodo box-counting.
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
            counts.append(count if count > 0 else 1)  # Evitare zeri

        log_sizes = np.log(sizes)
        log_counts = np.log(counts)

        coeffs = np.polyfit(log_sizes, log_counts, 1)
        return -coeffs[0]

    def refine_segmentation_with_fractal(self, image, threshold=0.08):
        """
        Raffina la segmentazione esistente basandosi sulla dimensione frattale.
        """
        thresh_val = threshold_otsu(image)
        binary_segment = image > thresh_val
        
        binary_segment = morphology.remove_small_objects(binary_segment, 30)
        binary_segment = binary_fill_holes(binary_segment)
        
        labeled_segments = measure.label(binary_segment)
        
        refined_mask = np.zeros_like(image, dtype=bool)
    
        print("Valori di dimensione frattale per ogni segmento:")
        
        for region in measure.regionprops(labeled_segments):
            if region.area >= 50:
                minr, minc, maxr, maxc = region.bbox
                segment = binary_segment[minr:maxr, minc:maxc]
                segment_outline = morphology.binary_dilation(segment) ^ segment
                
                # Calcolo della dimensione frattale del contorno
                fractal_dim = self.fractal_dimension(segment_outline)
                print(f"Dimensione frattale del segmento ({minr}, {minc}) - ({maxr}, {maxc}): {fractal_dim}")
                
                # Filtrare per dimensione frattale
                if fractal_dim > threshold:
                    refined_mask[minr:maxr, minc:maxc] |= segment

        return refined_mask
    
