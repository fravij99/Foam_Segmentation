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
            
    @staticmethod
    def median_filter(self, contrast):
        return cv2.medianBlur(contrast, 7)
    
    @staticmethod
    def gaussian_filter(self, contrast):
        return cv2.GaussianBlur(contrast, (5, 5), 0)
    
    @staticmethod
    def threshold_otsu(self, smoothed):
        return cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] #Using OTSU iterative thresholdindg method
    
    @staticmethod
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
        plt.savefig(f'{self.root}/binarization/{self.filename}', dpi=300)
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
            'Radius (cm)': 8, 
            'Resolution (pixels)': 1024, 
            'Radius bubbles (cm)': np.array([mean_diameter])/1024 * 4.79, 
            'Average correlation': 0 #self.calculate_correlation_with_neighbors(props)
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
        plt.savefig(f'{self.root}/segmentation/{self.filename}', dpi=300)
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
    

class heigth_measurer:

    def misura_altezza_schiuma(self, frame, bounding_box):
        """
        Calcola l'altezza della schiuma in un'immagine binarizzata.
        La schiuma è rappresentata da righe completamente nere.
        """
        x, y, w, h = bounding_box
        
        # Ritaglia la regione di interesse (ROI) dall'immagine
        roi = frame[y:y+h, x:x+w]
        
        # Conta le righe completamente nere
        row_sums = np.sum(roi == 0, axis=1)  # Somma dei pixel neri per riga
        foam_rows = np.where(row_sums == w)[0]  # Trova righe completamente nere
        
        if foam_rows.size > 0:
            foam_height = len(foam_rows)  # Numero di righe completamente nere consecutive
        else:
            foam_height = 0  # Nessuna schiuma rilevata
        
        return foam_height


    # Funzione per calcolare e tracciare l'evoluzione temporale della schiuma
    def foam_progression_plot(self, immagini, start_x, end_x, col_width, num_colonne, y, h):
        """
        Calcola e traccia l'evoluzione temporale dell'altezza della schiuma.
        """
        # Genera posizioni casuali uniformi per le colonne
        colonne_x = sorted([random.randint(start_x, end_x - col_width) for _ in range(num_colonne)])
        
        # Calcolo dell'altezza della schiuma per ciascuna immagine
        altezze_colonne = {x: [] for x in colonne_x}  # Dizionario per memorizzare le altezze per ciascuna colonna
        
        for frame in immagini:
            for x in colonne_x:
                bounding_box = (x, y, col_width, h)  # Bounding box per la colonna
                altezza_schiuma = self.misura_altezza_schiuma(frame, bounding_box)
                altezze_colonne[x].append(altezza_schiuma)
        
        # Calcola la media e la deviazione standard delle altezze per ciascun passo temporale
        num_immagini = len(immagini)
        altezze_medie = []
        deviazioni_standard = []
        
        for i in range(num_immagini):
            altezze_step = [altezze_colonne[x][i] for x in colonne_x]
            altezze_medie.append(np.mean(altezze_step))
            deviazioni_standard.append(np.std(altezze_step))
        
        # Fit esponenziale
        def funzione_esponenziale(t, a, b, c):
            return a * np.exp(b * t) + c

        time_indices = np.arange(num_immagini)
        parametri_iniziali = [1, 0.01, 1]  # Valori iniziali per il fit
        popt, _ = curve_fit(funzione_esponenziale, time_indices, altezze_medie, p0=parametri_iniziali, maxfev=5000)

        # Parametri ottimizzati
        a, b, c = popt

        # Calcola i valori del fit
        fit_values = funzione_esponenziale(time_indices, a, b, c)
        
        # Plot delle altezze con barre di errore e fit esponenziale
        plt.figure(figsize=(12, 6))
        
        # Dati originali con barre di errore
        plt.errorbar(
            time_indices,
            altezze_medie,
            yerr=deviazioni_standard,
            fmt='o',
            color='lightblue',
            ecolor='black',
            markeredgecolor='black',
            elinewidth=1,
            capsize=3,
            label='Media delle colonne con deviazione standard'
        )
        
        # Fit esponenziale
        plt.plot(
            time_indices,
            fit_values,
            color='red',
            linestyle='--',
            label=f'Fit esponenziale: a={a:.2f}, b={b:.4f}, c={c:.2f}'
        )
        
        plt.xlabel('Indice temporale')
        plt.ylabel('Altezza della schiuma (pixel)')
        plt.title('foam evolution for IDS Rocca4')
        plt.legend()
        plt.show()


class Binarizer:
    def __init__(self, origin_folder, output_folder):
        """
        Inizializza la classe Binarizer con la directory di origine e la directory di output.
        """
        self.origin_folder = origin_folder
        self.output_folder = output_folder

    @staticmethod
    def apply_clahe(image):
        """
        Applica il filtro CLAHE per migliorare il contrasto dell'immagine.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    @staticmethod
    def median_filter(image):
        """
        Applica il filtro mediano all'immagine.
        """
        return cv2.medianBlur(image, 7)

    @staticmethod
    def threshold_otsu(image):
        """
        Applica la soglia di Otsu per binarizzare l'immagine.
        """
        return cv2.bitwise_not(cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
                  

    @staticmethod
    def clean_binary_image(binary_image):
        """
        Rimuove piccoli oggetti dal risultato binarizzato.
        """
        return morphology.remove_small_objects(binary_image.astype(bool), min_size=50).astype(np.uint8)

    def process_image(self, image):
        """
        Esegue la binarizzazione completa su un'immagine.
        """
        # Converti in scala di grigi se necessario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Miglioramento del contrasto
        contrast = self.apply_clahe(gray)
        
        # Filtro per ridurre il rumore
        smoothed = self.median_filter(contrast)
        
        # Binarizzazione con Otsu
        binary = self.threshold_otsu(smoothed)
        
        # Rimuovi piccoli oggetti
        cleaned_binary = self.clean_binary_image(binary)
        
        return cleaned_binary

    def binarize_folder(self):
        """
        Esegue la binarizzazione su tutte le immagini nella cartella di origine e salva
        le immagini binarizzate nella cartella di output mantenendo la struttura.
        """
        for root, _, files in os.walk(self.origin_folder):
            # Determina il percorso relativo
            relative_path = os.path.relpath(root, self.origin_folder)
            output_path = os.path.join(self.output_folder, relative_path)
            os.makedirs(output_path, exist_ok=True)
            
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    # Percorso completo dei file
                    input_file = os.path.join(root, file)
                    output_file = os.path.join(output_path, file)
                    
                    # Carica l'immagine
                    image = cv2.imread(input_file)
                    if image is None:
                        print(f"Errore nel caricamento dell'immagine: {input_file}")
                        continue
                    
                    # Binarizza l'immagine
                    binary_image = self.process_image(image)
                    
                    # Salva l'immagine binarizzata
                    plt.imsave(output_file, binary_image, cmap='gray')
                    print(f"Immagine binarizzata salvata: {output_file}")
        return output_path
