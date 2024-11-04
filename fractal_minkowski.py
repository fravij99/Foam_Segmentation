import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

# Funzione per l'analisi frattale di Minkowski
def minkowski_fractal_analysis(image, max_iterations=10):
    if len(np.array(image).shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

    boundary_areas = []
    for i in range(1, max_iterations + 1):
        dilated = cv2.dilate(binary_image, None, iterations=i)
        eroded = cv2.erode(binary_image, None, iterations=i)
        boundary_image = cv2.bitwise_or(cv2.bitwise_not(dilated), eroded)
        boundary_area = np.sum(boundary_image == 255)
        effective_boundary_width = boundary_area / (i * max_iterations)
        boundary_areas.append(effective_boundary_width)

    log_cycles = np.log(np.arange(1, max_iterations + 1))
    log_boundary_areas = np.log(boundary_areas)
    model = HuberRegressor().fit(log_cycles.reshape(-1, 1), log_boundary_areas)
    slope = model.coef_[0]
    DS = 1 - slope

    DS1, DS2, crossing_point = None, None, None
    if not np.isclose(slope, 1, atol=0.05):
        split_index = max_iterations // 2
        model1 = HuberRegressor().fit(log_cycles[:split_index].reshape(-1, 1), log_boundary_areas[:split_index])
        model2 = HuberRegressor().fit(log_cycles[split_index:].reshape(-1, 1), log_boundary_areas[split_index:])
        DS1 = 1 - model1.coef_[0]
        DS2 = 1 - model2.coef_[0]
        crossing_point = (log_boundary_areas[split_index-1] + log_boundary_areas[split_index]) / 2

    plt.figure(figsize=(8, 6))
    plt.plot(log_cycles, log_boundary_areas, 'o-', label="Boundary width data")
    plt.plot(log_cycles, model.predict(log_cycles.reshape(-1, 1)), 'r--', label=f"DS = {DS:.4f}")
    if DS1 and DS2:
        plt.plot(log_cycles[:split_index], model1.predict(log_cycles[:split_index].reshape(-1, 1)), 'g--', label=f"DS1 = {DS1:.4f}")
        plt.plot(log_cycles[split_index:], model2.predict(log_cycles[split_index:].reshape(-1, 1)), 'b--', label=f"DS2 = {DS2:.4f}")
    plt.xlabel("log(Cycles)")
    plt.ylabel("log(Boundary width)")
    plt.legend()
    plt.title("Minkowski Fractal Dimension Analysis")
    plt.show()

    return DS, DS1, DS2, crossing_point

# Funzione per segmentare le bolle e calcolare i diametri
def segment_bubbles(image_path, min_size=100, max_iterations=10):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Esegui l'analisi frattale per ottenere DS
    DS, DS1, DS2, crossing_point = minkowski_fractal_analysis(image, max_iterations)
    print("Dimensione frattale (DS):", DS)
    if DS1 and DS2:
        print("Dimensione frattale testurale (DS1):", DS1)
        print("Dimensione frattale strutturale (DS2):", DS2)
        print("Punto di incrocio (Crossing Point):", crossing_point)
    
    # Converti l'immagine in binario e inverte per ottenere bolle bianche su sfondo nero
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Rimuove piccoli rumori e migliora il contrasto
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # Etichetta le regioni con connettività a 4
    labeled_array, num_features = ndimage.label(cleaned_image)
    diameters = []

    # Crea una maschera vuota per le bolle
    mask = np.zeros_like(cleaned_image)

    # Calcola il diametro di ciascuna bolla identificata
    for i in range(1, num_features + 1):
        bubble = (labeled_array == i)
        area = np.sum(bubble)
        
        # Considera solo le bolle più grandi della dimensione minima
        if area >= min_size:
            # Trova contorni e calcola il diametro equivalente
            contours, _ = cv2.findContours(bubble.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = contours[0]
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                diameter = 2 * radius
                diameters.append(diameter)
                
                # Aggiungi la bolla alla maschera
                cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)

    # Visualizza i risultati
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Maschera delle Bolle Segmentate")
    plt.imshow(mask, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Distribuzione dei Diametri delle Bolle")
    plt.hist(diameters, bins=20, color='blue', edgecolor='black')
    plt.xlabel("Diametro")
    plt.ylabel("Conteggio")
    plt.show()

    return mask, diameters

# Esegui il codice di esempio (specifica il percorso della tua immagine)
image_path = 'demo_images_segmented/schiuma.jpg'
mask, diameters = segment_bubbles(image_path)
print("Diametri delle bolle:", diameters)
