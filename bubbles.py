import foamlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm


cartella = "C:/Users/Francesco/Downloads/OneDrive_2024-11-07/IDS 1/"
for nome_file in tqdm(os.listdir(cartella), desc='Analizing frames...'):

    seg=foamlib.classic_segmentator(cartella+nome_file, nome_file)
    seg.detecting_glass()
    binary, labels, props = seg.image_segmentation(seg.median_filter, seg.threshold_otsu)
    diameters = seg.filtering_counting_bubbles(5, 120, props)
    seg.plotting_circles(props, diameters, 5, 120)
    fractal_dimension, box_sizes, box_counts = seg.computing_fractal_dimension(binary, min_box_size=2, max_box_size=100)
    seg.fractal_dimension_fit(fractal_dimension, box_sizes, box_counts)




"""# Carica l'immagine in scala di grigi
image = cv2.imread("demo_images_segmented/full_glass.jpg", cv2.IMREAD_GRAYSCALE)

# Usa il filtro per ridurre il rumore
blurred_image = cv2.GaussianBlur(image, (9, 9), 2)

# Applica la trasformata di Hough per rilevare i cerchi
circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                           param1=100, param2=30, minRadius=100, maxRadius=300)

# Filtra i cerchi per trovare quello con il raggio massimo e centrato nell'immagine
if circles is not None:
    max_circle = max(circles, key=lambda c: c[2])  # Seleziona il cerchio con il raggio più grande
    x, y, r = max_circle
    
    # Disegna solo il cerchio più grande
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)
    
    # Visualizza il cerchio più grande rilevato
    plt.imshow(output_image, cmap='gray')
    plt.title(f"Cerchio centrale rilevato: Centro=({x}, {y}), Raggio={r}")
    plt.axis('off')
    plt.show()
else:
    print("Nessun cerchio rilevato dopo il filtraggio.")
"""


"""# Carica l'immagine
image_path = "demo_images_segmented/schiumaweb2.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Applicazione della funzione di rifinitura segmentazione
refined_segmentation = refine_segmentation_with_fractal(image, threshold=0.1)  # Inizia con un valore più basso

# Visualizzazione del risultato
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Immagine Originale')
ax[1].imshow(refined_segmentation, cmap='gray')
ax[1].set_title('Segmentazione Rifinita')
plt.show()"""
