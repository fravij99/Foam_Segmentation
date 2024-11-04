import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt


img = cv2.imread('demo_images_segmented\sint_schiuma.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrast = clahe.apply(gray)

contrast = cv2.equalizeHist(gray)

smooth = cv2.GaussianBlur(contrast, (5, 5), 0)


thresh = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 2)

edges = cv2.Canny(smooth, 100, 200)

plt.imshow(gray>=thresh, cmap='gray')
plt.show()

labels = measure.label(thresh, connectivity=2)
props = measure.regionprops(labels)


diameters = [prop.equivalent_diameter for prop in props if prop.equivalent_diameter > 0]
mean_diameter = np.mean(diameters)
print(f'Minimum diameter: {min(diameters)} pixels')
print(f'Maximum diameter: {max(diameters)} pixels')
print(f'Median: {np.median(diameters)} pixels')
print(f'Average with variance: {mean_diameter} +- {np.std(diameters)} pixels')



fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax.set_title(f'Average bubble diameter: {mean_diameter:.2f} pixels')


for bubble in props:
    y, x = bubble.centroid
    radius = bubble.equivalent_diameter / 2
    circ = plt.Circle((x, y), radius, color='r', fill=False, linewidth=0.5)
    ax.add_patch(circ)

plt.show()
