import foamlib
import os
from tqdm import tqdm


main_folder = "C:/Users/Francesco/Downloads/OneDrive_2024-11-07"
# Specifica la cartella principale

# Itera su tutte le sotto-cartelle e i file in esse contenuti
for root, dirs, files in os.walk(main_folder):
    for nome_file in tqdm(files, desc=f'Analizing frames in {root}...'):
        complete_path = os.path.join(main_folder, root, nome_file)

        # Esegui il tuo codice solo se il file è effettivamente valido per l'analisi
        if os.path.isfile(complete_path):  
            seg = foamlib.classic_segmentator(main_folder, root, nome_file)
            seg.detecting_glass()
            binary, labels, props = seg.image_segmentation(seg.median_filter, seg.threshold_otsu)
            fractal_dimension, box_sizes, box_counts = seg.computing_fractal_dimension(binary, min_box_size=2, max_box_size=100)
            seg.fractal_dimension_fit(fractal_dimension, box_sizes, box_counts)
            diameters = seg.saving_statistical_data(fractal_dimension, 5, 300, props, root+f'_statistical_bubbles.xlsx')
            seg.plotting_circles(props, diameters, 5, 300)

