import foamlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import seaborn as sns

sns.set_style('darkgrid')
# Percorso principale
main_folder = "C:/Users/Francesco/Downloads/IDSRocca"


for root, dirs, files in os.walk(main_folder):
    binary_images = []
    if not files:  # Salta directory senza file
        print(f"Nessun file trovato in {root}, passando alla prossima directory.")
        continue

    for nome_file in tqdm(files, desc=f'Analizzando frames in {root}...'):
        binarizer = foamlib.Binarizer(main_folder, root, nome_file)
        binary = binarizer.binarize_folder()
        if binary is not None:  # Aggiungi solo immagini valide
            binary_images.append(binary)
        else:
            print(f"Errore nella binarizzazione del file {nome_file}")

    if not binary_images:  # Salta visualizzazione se lista vuota
        print(f"Nessuna immagine binarizzata in {root}, continuando.")
        continue

    # Usa HeightMeasurer per selezionare la ROI
    heigth = foamlib.heigth_measurer(root)
    plt.imshow(binary_images[len(binary_images)-1], cmap='gray')
    plt.show()
    roi_coords = heigth.select_roi(binary_images[0])  # Seleziona ROI dalla prima immagine

    if not roi_coords:
        print("Nessuna ROI selezionata, continuando.")
        continue

    # Usa le coordinate della ROI selezionata
    start_x, start_y, end_x, end_y = roi_coords[0], roi_coords[1], roi_coords[2], roi_coords[3]

    # Traccia l'evoluzione temporale della schiuma
    heigth.foam_progression_plot(binary_images, start_x, end_x, start_y, end_y)
