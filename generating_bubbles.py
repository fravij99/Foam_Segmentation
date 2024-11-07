import numpy as np
import matplotlib.pyplot as plt

def generate_bubbles(image_size, bubble_count, small_bubble_radius_range, large_bubble_radius_range):
    """Genera un'immagine con bolle di schiuma rappresentate come circonferenze."""
    # Crea un'immagine con rumore gaussiano
    image = np.random.normal(0.5, 0.1, (image_size, image_size))  # Background con rumore gaussiano
    image = np.clip(image, 0, 1)  # Mantiene i valori tra 0 e 1
    bubbles = []  # Lista per memorizzare le posizioni e i raggi delle bolle

    for _ in range(bubble_count):
        # Scegli il tipo di bolla (piccola o grande)
        if np.random.rand() < 0.5:  # 50% probabilità per ciascun tipo
            radius = np.random.randint(*small_bubble_radius_range)
        else:
            radius = np.random.randint(*large_bubble_radius_range)

        # Genera coordinate casuali per il centro della bolla
        x = np.random.randint(radius, image_size - radius)
        y = np.random.randint(radius, image_size - radius)

        # Controlla se la bolla si sovrappone ad altre bolle
        overlap = False
        for bx, by, br in bubbles:
            if (x - bx) ** 2 + (y - by) ** 2 < (radius + br + 2) ** 2:
                overlap = True
                break

        # Aggiungi la bolla se non si sovrappone
        if not overlap:
            bubbles.append((x, y, radius))
            for i in range(-radius, radius):
                for j in range(-radius, radius):
                    distance = np.sqrt(i**2 + j**2)
                    # Disegna solo i punti che rientrano nella circonferenza
                    if radius - 1 <= distance <= radius:
                        xi = np.clip(x + i, 0, image_size - 1)
                        yj = np.clip(y + j, 0, image_size - 1)
                        image[xi, yj] = np.clip(image[xi, yj] + 0.5, 0, 1)  # Colore per la circonferenza della bolla

    return image

# Parametri
image_size = 256
bubble_count = 500
small_bubble_radius_range = (5, 10)
large_bubble_radius_range = (10, 15)

# Genera un'immagine
image = generate_bubbles(image_size, bubble_count, small_bubble_radius_range, large_bubble_radius_range)

# Visualizza l'immagine
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
