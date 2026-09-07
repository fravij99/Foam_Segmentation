import os
import cv2
import numpy as np
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def dummy_dir(tmp_path):
    # Create a directory with a mix of files and subdirs
    dir_path = tmp_path / "app_test_dir"
    dir_path.mkdir()
    
    # Subdir
    (dir_path / "subdir").mkdir()
    
    # Dummy images
    for i in range(5):
        img_path = dir_path / f"test_{i}.jpg"
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[20+i:30, :] = 255  # Just something to binarize
        cv2.imwrite(str(img_path), img)
    
    # Non-image file
    (dir_path / "test.txt").write_text("hello")
    
    return str(dir_path)

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200

def test_list_dir(client, dummy_dir):
    # Test valid dir
    response = client.post('/api/list_dir', json={'path': dummy_dir})
    assert response.status_code == 200
    data = response.get_json()
    assert data['current_path'] == dummy_dir
    assert len(data['items']) == 7  # subdir, 5 test images, test.txt
    
    # Test empty path (defaults to home dir)
    response = client.post('/api/list_dir', json={'path': ''})
    assert response.status_code == 200
    
    # Test invalid dir
    response = client.post('/api/list_dir', json={'path': '/invalid/fake/path/12345'})
    assert response.status_code == 400
    
def test_list_dir_exception(client, monkeypatch):
    def mock_listdir(path):
        raise Exception("Mocked exception")
    monkeypatch.setattr(os, "listdir", mock_listdir)
    response = client.post('/api/list_dir', json={'path': os.path.expanduser('~')})
    assert response.status_code == 500

def test_get_first_image(client, dummy_dir, tmp_path):
    # Test valid dir with image
    response = client.post('/api/get_first_image', json={'path': dummy_dir})
    assert response.status_code == 200
    assert 'image' in response.get_json()
    
    # Test valid dir without image
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    response = client.post('/api/get_first_image', json={'path': str(empty_dir)})
    assert response.status_code == 404
    
    # Test invalid dir
    response = client.post('/api/get_first_image', json={'path': '/invalid/fake/path/123'})
    assert response.status_code == 400
    
def test_analyze_top(client, dummy_dir):
    # Valid
    response = client.post('/api/analyze/top', json={'path': dummy_dir})
    assert response.status_code == 200
    assert response.get_json()['processed'] == 5
    
    # Invalid
    response = client.post('/api/analyze/top', json={'path': '/invalid/fake'})
    assert response.status_code == 400

def test_analyze_side(client, dummy_dir, tmp_path):
    # Valid
    response = client.post('/api/analyze/side', json={
        'path': dummy_dir,
        'bbox': [10, 10, 20, 20]
    })
    assert response.status_code == 200
    assert response.get_json()['success'] == True
    
    # Invalid dir
    response = client.post('/api/analyze/side', json={'path': '/invalid'})
    assert response.status_code == 400
    
    # Invalid bbox
    response = client.post('/api/analyze/side', json={'path': dummy_dir, 'bbox': [10]})
    assert response.status_code == 400
    
    # Empty dir (no valid files inside)
    empty_dir = tmp_path / "empty_dir2"
    empty_dir.mkdir()
    response = client.post('/api/analyze/side', json={
        'path': str(empty_dir),
        'bbox': [10, 10, 20, 20]
    })
    assert response.status_code == 200 # It should just skip and return success

def test_analyze_side_no_binary(client, tmp_path):
    bad_dir = tmp_path / "bad_dir"
    bad_dir.mkdir()
    (bad_dir / "bad.jpg").write_text("not an image")
    response = client.post('/api/analyze/side', json={
        'path': str(bad_dir),
        'bbox': [10, 10, 20, 20]
    })
    assert response.status_code == 200
