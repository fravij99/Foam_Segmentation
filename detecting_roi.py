import foamlib
import matplotlib.pyplot as plt



main_folder="C:/Users/Francesco/Downloads/GINGER_CAM"
root="C:/Users/Francesco/Downloads/GINGER_CAM/AVginger2"
nome_file='frameAV_031.jpg'
seg = foamlib.classic_segmentator(main_folder, root, nome_file)
plt.imshow(seg.img)
plt.show()
seg.detecting_glass()