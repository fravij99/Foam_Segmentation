import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit
import seaborn as sns
import foamlib
sns.set_style('darkgrid')

# Percorso alla cartella di origine contenente tutte le immagini
path = "C:/Users/Francesco/Downloads/IDSRocca4/IDSRocca4"
# Percorso alla cartella di output dove salvare le immagini binarizzate
output_folder = "C:/Users/Francesco/Downloads/IDSRocca4/IDSRocca4/binarization"
# Crea un'istanza della classe Binarizer
binarizer = foamlib.Binarizer(path, output_folder)
# Esegui la binarizzazione
output_folder=binarizer.binarize_folder()

heigth=foamlib.heigth_measurer()

# Percorso alla cartella con le immagini binarizzate

# Carica tutte le immagini binarizzate
immagini = [cv2.imread(os.path.join(output_folder, f), cv2.IMREAD_GRAYSCALE) for f in sorted(os.listdir(output_folder)) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
plt.imshow(immagini[0])
# Verifica che siano state caricate immagini
if not immagini:
    print("Nessuna immagine trovata nella directory.")
else:
    # Parametri della regione di campionamento
    start_x = 550  # Inizio della regione
    end_x = 660    # Fine della regione
    col_width = 5  # Larghezza di ciascuna colonna
    num_colonne = 100  # Numero di colonne da campionare
    y, h = 400, 700  # Coordinate e altezza della ROI

    # Traccia l'evoluzione temporale della schiuma
    heigth.foam_progression_plot(immagini, start_x, end_x, col_width, num_colonne, y, h)