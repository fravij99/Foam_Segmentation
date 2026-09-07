import os
import base64
import cv2
from flask import Flask, render_template, request, jsonify
from foam_segmentation import classic_segmentator, Binarizer, heigth_measurer
from tqdm import tqdm

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/list_dir', methods=['POST'])
def list_dir():
    data = request.json
    path = data.get('path', '')
    
    if not path:
        path = os.path.expanduser('~')
    
    if not os.path.exists(path) or not os.path.isdir(path):
        return jsonify({'error': 'Invalid path'}), 400
        
    try:
        items = []
        parent = os.path.dirname(path) if os.path.dirname(path) != path else path
        
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            is_dir = os.path.isdir(full_path)
            items.append({
                'name': item,
                'path': full_path,
                'is_dir': is_dir
            })
            
        items.sort(key=lambda x: (not x['is_dir'], x['name'].lower()))
        
        return jsonify({
            'current_path': path,
            'parent_path': parent,
            'items': items
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_first_image', methods=['POST'])
def get_first_image():
    data = request.json
    path = data.get('path', '')
    if not os.path.isdir(path):
        return jsonify({'error': 'Invalid directory'}), 400
        
    for item in sorted(os.listdir(path)):
        if item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            full_path = os.path.join(path, item)
            img = cv2.imread(full_path)
            if img is not None:
                _, buffer = cv2.imencode('.jpg', img)
                b64 = base64.b64encode(buffer).decode('utf-8')
                return jsonify({'image': b64})
                
    return jsonify({'error': 'No images found'}), 404

@app.route('/api/analyze/top', methods=['POST'])
def analyze_top():
    data = request.json
    main_folder = data.get('path', '')
    
    if not os.path.isdir(main_folder):
        return jsonify({'error': 'Invalid directory'}), 400
        
    # Using classic_segmentator logic from examples/bubbles.py
    # For a web request, this might take time, but we'll run it synchronously for now.
    processed_files = 0
    for root, dirs, files in os.walk(main_folder):
        for nome_file in files:
            if nome_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                seg = classic_segmentator(main_folder, root, nome_file)
                seg.detecting_glass()
                binary, labels, props = seg.image_segmentation(seg.median_filter, seg.threshold_otsu)
                fractal_dimension, box_sizes, box_counts = seg.computing_fractal_dimension(binary, min_box_size=2, max_box_size=100)
                seg.fractal_dimension_fit(fractal_dimension, box_sizes, box_counts)
                diameters = seg.saving_statistical_data(fractal_dimension, 5, 300, props, os.path.join(root, 'statistical_bubbles.xlsx'))
                seg.plotting_circles(props, diameters, 5, 300)
                processed_files += 1
                
    return jsonify({'success': True, 'processed': processed_files})

@app.route('/api/analyze/side', methods=['POST'])
def analyze_side():
    data = request.json
    main_folder = data.get('path', '')
    bbox = data.get('bbox', [])
    
    if not os.path.isdir(main_folder) or len(bbox) != 4:
        return jsonify({'error': 'Invalid directory or bounding box'}), 400
        
    start_x, start_y, end_x, end_y = map(int, bbox)
    
    # Process images logic from examples/foam_heigth.py
    for root, dirs, files in os.walk(main_folder):
        binary_images = []
        valid_files = [f for f in sorted(files) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not valid_files:
            continue
            
        for nome_file in valid_files:
            binarizer = Binarizer(main_folder, root, nome_file)
            if binarizer.img is not None:
                binary = binarizer.binarize_folder()
                if binary is not None:
                    binary_images.append(binary)
                
        if not binary_images:
            continue
            
        heigth = heigth_measurer(root)
        
        # Analyze and save plots without blocking matplotlib
        heigth.foam_progression_plot(binary_images, start_x, end_x, start_y, end_y, show_plot=False)
        
    return jsonify({'success': True, 'message': 'Side analysis complete. Results saved in the folder.'})

if __name__ == '__main__': # pragma: no cover
    app.run(debug=True)

