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
            print("Nessun contorno rilevato!")
            

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
        plt.imshow(binary, cmap='gray')
        plt.savefig(f'{self.root}/binarization/{self.filename}')
        # Masking
        labels = measure.label(binary, connectivity=2)
        props = measure.regionprops(labels)   
        
        return binary, labels, props  


        


    """Calcola la dimensione frattale usando il metodo del box-counting e ritorna i dati
        per il grafico.
        
        Parameters:
            binary_image (ndarray): Immagine binaria con le bolle segmentate.
            min_box_size (int): Dimensione minima del box.
            max_box_size (int): Dimensione massima del box.
            
        Returns:
            (float, list, list): La dimensione frattale e i dati per il grafico."""
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
        # Estrarre i centri e i diametri delle bolle
        centroids = np.array([prop.centroid for prop in props])
        diameters = np.array([prop.equivalent_diameter for prop in props])
        
        if len(diameters) == 0:
            print("No bubbles detected.")
            return None

        # Calcolare la matrice delle distanze tra i centri delle bolle
        dist_matrix = distance_matrix(centroids, centroids)

        correlations = []
        
        for i, D in enumerate(diameters):
            # Trovare gli indici dei vicini il cui centro è entro una distanza <= 2D
            neighbors = np.where((dist_matrix[i] <= 5 * D) & (dist_matrix[i] > 0))[0]  # Ignora la bolla stessa

            if len(neighbors) > 0:
                neighbor_diameters = diameters[neighbors]
                # Calcolare la correlazione tra il diametro della bolla in esame e i vicini
                correlation = np.corrcoef(np.full(len(neighbor_diameters), D), neighbor_diameters)[0, 1]

                correlations.append(correlation)

        # Restituire la media delle correlazioni o un'altra misura caratteristica
        mean_correlation = np.nanmean(correlations)  # Gestisce i NaN se non ci sono vicini
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

        # Dati da salvare su Excel
        data = {
            'Filename': [self.filename],
            'Fractal Dimension': [fractal_dimension],
            'Minimum Diameter (pixels)': [min_diam],
            'Maximum Diameter (pixels)': [max_diam],
            'Median Diameter (pixels)': [median_diam],
            'Average Diameter (pixels)': [mean_diameter],
            'Variance (pixels)': [std_diam], 
            'Radius (cm)': 4.79, 
            'Resolution (pixels)': 1024, 
            'Radius bubbles (cm)': np.array([mean_diameter])/1024 * 4.79, 
            'Average correlation': self.calculate_correlation_with_neighbors(props)
        }
        # Se un percorso Excel è stato fornito, salviamo la dimensione frattale
        if excel_path:
            # Riapriamo il file Excel e aggiungiamo la dimensione frattale
            
            df = pd.DataFrame(data)
            try:
                with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                    df.to_excel(writer, sheet_name='Bubble Data', index=False, header=writer.sheets.get('Bubble Data') is None, startrow=writer.sheets['Bubble Data'].max_row if 'Bubble Data' in writer.sheets else 0)
            except FileNotFoundError:
                # Se il file non esiste, lo crea
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
        plt.savefig(f'{self.root}/fractal_fit/{self.filename}')
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
        plt.savefig(f'{self.root}/segmentation/{self.filename}')
        plt.close(fig)
        



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
    
