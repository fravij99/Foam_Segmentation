import foamlib
import cv2
import matplotlib.pyplot as plt

seg=foamlib.classic_segmentator()



img = cv2.imread("demo_images_segmented/schiumaweb.jpg")

binary, labels, props = seg.image_segmentation(img, seg.median_filter, seg.threshold_otsu)
diameters = seg.filtering_counting_bubbles(4, props)
seg.plotting_circles(img, props, diameters, 4)


fractal_dimension, box_sizes, box_counts = seg.computing_fractal_dimension(binary, min_box_size=2, max_box_size=100)
seg.fractal_dimension_fit(fractal_dimension, box_sizes, box_counts)





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
