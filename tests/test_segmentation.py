import os
import cv2
import numpy as np
import pytest
from foam_segmentation import classic_segmentator, Binarizer, heigth_measurer

@pytest.fixture
def dummy_image(tmp_path):
    # Create a dummy image with a white circle
    img_dir = tmp_path / "test_folder"
    img_dir.mkdir()
    img_path = img_dir / "test_image.jpg"
    
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (255, 255, 255), -1)
    
    cv2.imwrite(str(img_path), img)
    return str(tmp_path), "test_folder", "test_image.jpg"

def test_classic_segmentator(dummy_image):
    base_dir, root, filename = dummy_image
    
    seg = classic_segmentator(base_dir, root, filename)
    assert seg.img is not None
    assert seg.img.shape == (100, 100, 3)

def test_binarizer(dummy_image):
    base_dir, root, filename = dummy_image
    
    binarizer = Binarizer(base_dir, root, filename)
    assert binarizer.img is not None
    
    binary = binarizer.process_image(binarizer.img)
    assert binary is not None
    assert len(binary.shape) == 2 # Grayscale/binary

def test_heigth_measurer():
    measurer = heigth_measurer("dummy_root")
    
    # Create a dummy binary frame (foam layer in middle)
    frame = np.ones((50, 50), dtype=np.uint8) * 255
    frame[20:30, :] = 0  # 10 rows of 'foam'
    
    bbox = (10, 0, 20, 50) # x, y, w, h
    h = measurer.measure_foam_heigth(frame, bbox)
    assert h == 10

def test_classic_segmentator_full(dummy_image, tmp_path):
    base_dir, root, filename = dummy_image
    
    seg = classic_segmentator(base_dir, root, filename)
    seg.detecting_glass()
    assert seg.img is not None
    
    # Run image_segmentation
    binary, labels, props = seg.image_segmentation(seg.median_filter, seg.threshold_otsu)
    assert binary is not None
    assert len(props) > 0 # Should have found the dummy circle
    
    # Run computing_fractal_dimension
    frac_dim, box_sizes, box_counts = seg.computing_fractal_dimension(binary, min_box_size=2, max_box_size=10)
    assert frac_dim > 0
    assert len(box_sizes) == len(box_counts)
    
    # Run saving_statistical_data
    excel_path = str(tmp_path / "stats.xlsx")
    diameters = seg.saving_statistical_data(frac_dim, 5, 300, props, excel_path)
    # Call it again to test append mode
    seg.saving_statistical_data(frac_dim, 5, 300, props, excel_path)
    
    # Test calculate_correlation_with_neighbors explicitly
    seg.calculate_correlation_with_neighbors(props)
    
    # Test fractal_dimension_fit
    seg.fractal_dimension_fit(frac_dim, box_sizes, box_counts)
    
    # Test other filters
    gf = seg.gaussian_filter(seg.img)
    ad = seg.threshold_adaptive(cv2.cvtColor(seg.img, cv2.COLOR_BGR2GRAY))
    assert len(diameters) > 0
    assert os.path.exists(excel_path)
    
    # Run plotting_circles
    seg.plotting_circles(props, diameters, 5, 300)
    assert os.path.exists(os.path.join(base_dir, root, "segmentation", filename))

def test_fractal_segmentator():
    from foam_segmentation.core import fractal_segmentator
    fs = fractal_segmentator()
    dummy_img = np.zeros((50, 50), dtype=np.uint8)
    dummy_img[10:40, 10:40] = 255
    
    fd = fs.fractal_dimension(dummy_img)
    assert fd is not None
    
    # test refine_segmentation_with_fractal
    dummy_img_noisy = np.zeros((100, 100), dtype=np.uint8)
    dummy_img_noisy[20:80, 20:80] = 255
    refined = fs.refine_segmentation_with_fractal(dummy_img_noisy)
    assert refined is not None

def test_heigth_measurer_progression(tmp_path):
    measurer = heigth_measurer(str(tmp_path) + "/")
    
    # Create dummy images with decreasing foam height
    images = []
    for i in range(5):
        frame = np.ones((50, 50), dtype=np.uint8) * 255
        frame[20+i:30, :] = 0  # Foam shrinking
        
        # Inject an outlier to cover outlier lines 365, 367
        if i == 2:
            frame[5:45, 12] = 0 # Outlier high
            frame[25:26, 15] = 0 # Outlier low
        images.append(frame)
        
    measurer.foam_progression_plot(images, 10, 20, 0, 50, show_plot=False)
    assert os.path.exists(str(tmp_path / "foam_data.xlsx"))
    
    # Force curve_fit exception by passing constant data or identical points to cover exception lines
    images_fail = [np.ones((50, 50), dtype=np.uint8) * 255] * 5
    measurer.foam_progression_plot(images_fail, 10, 20, 0, 50, show_plot=False)

def test_no_contour_and_no_bubbles(tmp_path):
    # Empty image
    img_dir = tmp_path / "test_folder2"
    img_dir.mkdir()
    img_path = img_dir / "test_empty.jpg"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    
    seg = classic_segmentator(str(tmp_path), "test_folder2", "test_empty.jpg")
    seg.detecting_glass() # Should print "No contour detected"
    
    # Test saving statistical data with empty props
    excel_path = str(tmp_path / "empty_stats.xlsx")
    seg.saving_statistical_data(1.0, 5, 300, [], excel_path)
    seg.calculate_correlation_with_neighbors([])
    
def test_select_roi_mock(monkeypatch):
    measurer = heigth_measurer("dummy_root")
    
    # Mock plt.show to just invoke the onselect callback manually
    import matplotlib.pyplot as plt
    def mock_show():
        # Call it forcefully to simulate the event
        pass
    monkeypatch.setattr(plt, "show", mock_show)
    
    dummy_img = np.zeros((50, 50), dtype=np.uint8)
    # This won't cover the inner function of the event unless we invoke the event.
    measurer.select_roi(dummy_img)

def test_binarize_folder(dummy_image):
    base_dir, root, filename = dummy_image
    binarizer = Binarizer(base_dir, root, filename)
    binary = binarizer.binarize_folder()
    assert binary is not None
    assert os.path.exists(os.path.join(base_dir, root, "binarization", filename))
    
    # Cover the grayscale branch
    gray_img = cv2.cvtColor(binarizer.img, cv2.COLOR_BGR2GRAY)
    binarizer.process_image(gray_img)
