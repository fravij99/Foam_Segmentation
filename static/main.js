document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const folderPathInput = document.getElementById('folder-path');
    const browseBtn = document.getElementById('browse-btn');
    const folderModal = document.getElementById('folder-modal');
    const closeModal = document.querySelector('.close-modal');
    const upDirBtn = document.getElementById('up-dir-btn');
    const currentDirText = document.getElementById('current-dir-text');
    const dirList = document.getElementById('dir-list');
    const selectDirBtn = document.getElementById('select-dir-btn');
    
    const analyzeTopBtn = document.getElementById('analyze-top-btn');
    const analyzeSideBtn = document.getElementById('analyze-side-btn');
    
    const roiModal = document.getElementById('roi-modal');
    const closeRoiModal = document.querySelector('.close-roi-modal');
    const confirmRoiBtn = document.getElementById('confirm-roi-btn');
    const roiCanvas = document.getElementById('roi-canvas');
    const ctx = roiCanvas.getContext('2d');
    
    let currentPath = '';
    let selectedPath = '';

    // Folder Browser Logic
    browseBtn.addEventListener('click', () => {
        loadDirectory(currentPath);
        folderModal.classList.remove('hidden');
    });

    closeModal.addEventListener('click', () => {
        folderModal.classList.add('hidden');
    });

    async function loadDirectory(path) {
        try {
            const res = await fetch('/api/list_dir', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path })
            });
            const data = await res.json();
            
            if (res.ok) {
                currentPath = data.current_path;
                currentDirText.textContent = currentPath;
                upDirBtn.onclick = () => loadDirectory(data.parent_path);
                
                dirList.innerHTML = '';
                data.items.forEach(item => {
                    const li = document.createElement('li');
                    li.innerHTML = `<span>${item.is_dir ? '📁' : '📄'}</span> ${item.name}`;
                    if (item.is_dir) {
                        li.onclick = () => loadDirectory(item.path);
                    }
                    dirList.appendChild(li);
                });
            } else {
                alert('Error: ' + data.error);
            }
        } catch (e) {
            console.error(e);
        }
    }

    selectDirBtn.addEventListener('click', () => {
        selectedPath = currentPath;
        folderPathInput.value = selectedPath;
        folderModal.classList.add('hidden');
        analyzeTopBtn.disabled = false;
        analyzeSideBtn.disabled = false;
    });

    // ROI Selection Logic
    let isDrawing = false;
    let startX = 0, startY = 0;
    let rectW = 0, rectH = 0;
    let imageObj = new Image();
    let imgScale = 1;

    analyzeSideBtn.addEventListener('click', async () => {
        // Fetch the first image to draw on canvas
        try {
            const res = await fetch('/api/get_first_image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: selectedPath })
            });
            const data = await res.json();
            if (res.ok) {
                imageObj.src = 'data:image/jpeg;base64,' + data.image;
                imageObj.onload = () => {
                    // Set canvas size
                    const containerWidth = document.querySelector('.canvas-container').clientWidth;
                    imgScale = containerWidth / imageObj.width;
                    
                    roiCanvas.width = imageObj.width * imgScale;
                    roiCanvas.height = imageObj.height * imgScale;
                    
                    drawImageAndRect();
                    roiModal.classList.remove('hidden');
                };
            } else {
                alert('Error: ' + data.error);
            }
        } catch (e) {
            console.error(e);
        }
    });

    closeRoiModal.addEventListener('click', () => {
        roiModal.classList.add('hidden');
    });

    roiCanvas.addEventListener('mousedown', (e) => {
        const rect = roiCanvas.getBoundingClientRect();
        startX = e.clientX - rect.left;
        startY = e.clientY - rect.top;
        isDrawing = true;
    });

    roiCanvas.addEventListener('mousemove', (e) => {
        if (!isDrawing) return;
        const rect = roiCanvas.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;
        rectW = currentX - startX;
        rectH = currentY - startY;
        drawImageAndRect();
    });

    roiCanvas.addEventListener('mouseup', () => {
        isDrawing = false;
    });
    
    roiCanvas.addEventListener('mouseleave', () => {
        isDrawing = false;
    });

    function drawImageAndRect() {
        ctx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
        ctx.drawImage(imageObj, 0, 0, roiCanvas.width, roiCanvas.height);
        
        if (rectW !== 0 || rectH !== 0) {
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 2;
            ctx.fillStyle = 'rgba(59, 130, 246, 0.2)';
            ctx.fillRect(startX, startY, rectW, rectH);
            ctx.strokeRect(startX, startY, rectW, rectH);
        }
    }

    confirmRoiBtn.addEventListener('click', () => {
        // Calculate original coordinates
        const origX = Math.min(startX, startX + rectW) / imgScale;
        const origY = Math.min(startY, startY + rectH) / imgScale;
        const origW = Math.abs(rectW) / imgScale;
        const origH = Math.abs(rectH) / imgScale;
        
        const bbox = [origX, origY, origX + origW, origY + origH];
        
        roiModal.classList.add('hidden');
        
        // Start Side analysis
        fetch('/api/analyze/side', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: selectedPath, bbox: bbox })
        })
        .then(res => res.json())
        .then(data => {
            alert('Analysis complete! Check results.');
            console.log(data);
        })
        .catch(e => console.error(e));
    });

    // Top-down analysis
    analyzeTopBtn.addEventListener('click', () => {
        fetch('/api/analyze/top', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: selectedPath })
        })
        .then(res => res.json())
        .then(data => {
            alert('Analysis complete! Check results.');
            console.log(data);
        })
        .catch(e => console.error(e));
    });
});
