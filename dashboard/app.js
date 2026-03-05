let scene, camera, renderer, controls, model;

function init3D() {
    const container = document.getElementById('canvas-container');
    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.z = 5;

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 0.8);
    camera.add(pointLight);
    scene.add(camera);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const mtlLoader = new THREE.MTLLoader();
    const objLoader = new THREE.OBJLoader();

    function loadModelFromMetadata(data) {
        // Defaults if metadata fields are missing
        const objPath = data.obj_path ? `../${data.obj_path}` : '../output/rock_model.obj';
        const mtlPath = data.mtl_path ? `../${data.mtl_path}` : '../output/rock_model.mtl';

        console.log(`Attempting to load model from metadata: ${objPath}`);

        mtlLoader.load(mtlPath, (materials) => {
            materials.preload();
            objLoader.setMaterials(materials);
            objLoader.load(objPath, (obj) => {
                console.log(`Successfully loaded: ${objPath}`);
                model = obj;
                // Enhance materials for normal mapping
                model.traverse((child) => {
                    if (child.isMesh) {
                        child.material.side = THREE.DoubleSide;
                        if (child.material.map) child.material.map.anisotropy = 16;
                        if (child.material.normalMap) {
                            child.material.normalScale.set(1.5, 1.5); // Crystal roughness
                        }
                    }
                });
                // Center model
                const box = new THREE.Box3().setFromObject(model);
                const center = box.getCenter(new THREE.Vector3());
                model.position.sub(center);
                scene.add(model);
            }, undefined, (error) => {
                console.error(`Failed to load OBJ: ${objPath}`, error);
            });
        }, undefined, (error) => {
            console.warn(`Failed to load MTL: ${mtlPath}. Falling back to plain OBJ.`, error);
            objLoader.load(objPath, (obj) => {
                model = obj;
                const box = new THREE.Box3().setFromObject(model);
                const center = box.getCenter(new THREE.Vector3());
                model.position.sub(center);
                scene.add(model);
            }, undefined, () => console.error("Final model load failed."));
        });
    }

    // This will be called once metadata is loaded
    window.update3DModel = (data) => {
        if (model) {
            scene.remove(model);
        }
        loadModelFromMetadata(data);
    };

    window.addEventListener('resize', onWindowResize, false);
    animate();
}

function onWindowResize() {
    const container = document.getElementById('canvas-container');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
    requestAnimationFrame(animate);
    if (model) {
        // Subtle idle rotation
        // model.rotation.y += 0.005;
    }
    controls.update();
    renderer.render(scene, camera);
}

function showSheet(sheetId) {
    console.log(`Switching to sheet: ${sheetId}`);
    // Buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-sheet') === sheetId) {
            btn.classList.add('active');
        }
    });

    // Sheets
    document.querySelectorAll('.sheet').forEach(sheet => {
        sheet.classList.remove('active');
    });
    const targetSheet = document.getElementById(`sheet-${sheetId}`);
    if (targetSheet) {
        targetSheet.classList.add('active');
    } else {
        console.error(`Sheet not found: sheet-${sheetId}`);
    }

    // Special handling for 3D visibility
    if (sheetId === '3d') {
        onWindowResize();
    }
}

async function loadMetadata() {
    console.log("Loading geological metadata from ../output/result.json...");
    try {
        const response = await fetch('../output/result.json');
        if (!response.ok) throw new Error(`Fetch failed: ${response.status} ${response.statusText}`);

        const data = await response.json();
        console.log("Metadata payload received:", data);

        // Sidebar Update
        const content = document.getElementById('meta-content');
        if (content) {
            content.innerHTML = `
                <p><strong>Classification:</strong> ${data.rock_type}</p>
                <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                <hr style="margin: 10px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.1);">
                <p>${data.geology ? data.geology.description : 'No detailed description available.'}</p>
            `;
        }

        // Update 3D model using the newly loaded metadata paths
        if (window.update3DModel) {
            window.update3DModel(data);
        }

        // Gallery Population
        const gallery = document.getElementById('gallery-container');
        if (!gallery) {
            console.error("Gallery container not found in DOM!");
            return;
        }
        gallery.innerHTML = '';
        console.log("Populating gallery items...");

        // 1. Source Images
        if (data.images && Array.isArray(data.images)) {
            data.images.forEach((img, idx) => {
                const item = document.createElement('div');
                item.className = 'gallery-item';
                const isBest = idx === data.best_image_index;
                item.innerHTML = `
                    <img src="../${img}" alt="Source Image" onerror="console.warn('Failed to load gallery image: ${img}')">
                    <div class="label">Source Image ${idx + 1} ${isBest ? '(Best View)' : ''}</div>
                `;
                gallery.appendChild(item);
                console.log(`  Added source image: ${img}`);
            });
        }

        // 2. 3D Preview (Dynamic path if available)
        const previewPath = data.preview_path ? `../${data.preview_path}` : '../output/rock_model_preview.png';
        const previewItem = document.createElement('div');
        previewItem.className = 'gallery-item';
        previewItem.innerHTML = `
            <img src="${previewPath}" alt="3D Preview" onerror="this.parentElement.style.display='none'">
            <div class="label">3D Reconstruction (Snapshot)</div>
        `;
        gallery.appendChild(previewItem);

        // 3. Texture Map (Dynamic path if available)
        const texturePath = data.texture_path ? `../${data.texture_path}` : '../output/rock_model.png';
        const textureItem = document.createElement('div');
        textureItem.className = 'gallery-item';
        textureItem.innerHTML = `
            <img src="${texturePath}" alt="UV Texture" onerror="this.parentElement.style.display='none'">
            <div class="label">UV Texture Map</div>
        `;
        gallery.appendChild(textureItem);

        console.log("Gallery population complete.");

    } catch (e) {
        console.error("Geological metadata loading failed:", e);
        const content = document.getElementById('meta-content');
        if (content) {
            content.innerHTML = `
                <p style="color: #ff4d4d;"><strong>Data Sync Error</strong></p>
                <p style="font-size: 0.75rem;">Could not sync with analysis results. If running locally, ensure you are using a web server (Python -m http.server).</p>
            `;
        }
    }
}

// Start
init3D();
loadMetadata();
