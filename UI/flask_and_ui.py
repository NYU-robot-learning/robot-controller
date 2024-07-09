from flask import Flask, Response, render_template_string, send_from_directory, jsonify, url_for, request
import os
import zmq
import threading

app = Flask(__name__)

IMAGE_DIR = 'path_'
PCD_DIR = 'path'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
if not os.path.exists(PCD_DIR):
    os.makedirs(PCD_DIR)

live_feed_context = zmq.Context()
live_feed_receiver = live_feed_context.socket(zmq.PAIR)
live_feed_receiver.bind("tcp://*:5555")

context = zmq.Context()
image_receiver = context.socket(zmq.PULL)
image_receiver.bind("tcp://0.0.0.0:5558")

point_cloud_context = zmq.Context()
point_cloud_receiver = point_cloud_context.socket(zmq.PULL)
point_cloud_receiver.bind("tcp://0.0.0.0:5560")

point_cloud_data_path = os.path.join(PCD_DIR, '3d_point_cloud.pcd')
images_received = False

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap" rel="stylesheet">
    <title>Live Feed and Received Images</title>
    <style>
        body {
            background-color: #121212;
            color: #ffffff;
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .container {
            display: flex;
            height: 100vh;
        }
        .lhs, .rhs {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        .lhs {
        }
        .lhs-upper, .lhs-lower {
            flex: 1;
            position: relative;
            padding: 20px;
        }
        #live-feed {
            width: 100%;
            padding-top: 100%;
            position: relative;
        }
        #live-feed img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 80%;
            object-fit: cover;
        }
        #point-cloud-visualization {
            width: 100%;
            height: 100%;
        }
        .rhs-upper {
            flex: 1;
            position: relative;
            padding: 20px;
            overflow: hidden;
        }
        .rhs-lower {
            flex: 1;
            display: flex;
        }
        .image-card {
            margin: 10px;
            flex: 1;
            overflow: hidden;
        }
        .image-card img {
            max-width: 100%;
            height: auto;
            transition: transform 0.5s ease;
        }
        .image-card img:hover {
            transform: scale(2);
        }
        .card {
            background-color: #1f1f1f;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .btn-danger {
            background-color: #bb2d3b;
            border-color: #bb2d3b;
        }
        .btn-danger:hover {
            background-color: #a12431;
            border-color: #a12431;
        }
        .vertical-bar {
            position: absolute;
            left: 20px;
            width: 10px;
            height: 100%;
            top: 0;
            background-color: rgba(255, 0, 0, 0.3);
            border-radius: 5px;
            overflow: hidden;
            display: none;
        }
        .fill-bar {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 0;
            background-color: red;
            border-radius: 5px;
            transition: height 0.25s ease-out;
        }
        .vertical-bar-place {
            position: absolute;
            left: 20px;
            width: 10px;
            height: 100%;
            top: 0;
            background-color: rgba(0, 0, 255, 0.3);
            border-radius: 5px;
            overflow: hidden;
            display: none;
        }
        .fill-bar-place {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 0;
            background-color: blue;
            border-radius: 5px;
            transition: height 0.25s ease-out;
        }
        .step-container {
            position: relative;
            width: 100%;
            height: 100%;
            padding-left: 40px;
            display: none;
        }
        .step-container-place {
            position: relative;
            width: 100%;
            height: 100%;
            padding-left: 40px;
            display: none;
        }
        .step-list, .step-list-place {
            list-style-type: none;
            padding: 0;
            margin: 0;
        }
        .step-list li, .step-list-place li {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            opacity: 0;
            animation: fadeIn 0.25s ease-out forwards, slideUp 0.25s ease-out forwards;
        }
        .step-list li::before, .step-list-place li::before {
            content: '';
            width: 20px;
            height: 20px;
            background-color: white;
            border-radius: 50%;
            margin-right: 10px;
            flex-shrink: 0;
        }
        @keyframes slideUp {
            from {
                transform: translateY(20px);
            }
            to {
                transform: translateY(0);
            }
        }
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        .reset-button {
            position: absolute;
            top: 20px;
            right: 20px;
        }
        .centered-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 6em;
            color: yellow;
            text-align: center;
            opacity: 0;
            transition: opacity 0.5s ease-in-out;
            font-family: 'Orbitron', sans-serif;
        }
        .centered-text-blue {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 6em;
            color: blue;
            text-align: center;
            opacity: 0;
            transition: opacity 0.5s ease-in-out;
            font-family: 'Orbitron', sans-serif;
        }
        .zoom-in {
            animation: zoomIn 2s linear forwards;
        }
        @keyframes zoomIn {
            0% { 
                opacity: 0; 
                transform: translate(-50%, -50%) scale(0.5); 
            }
            50% { 
                opacity: 1; 
                transform: translate(-50%, -50%) scale(1.1); 
            }
            100% { 
                opacity: 1; 
                transform: translate(-50%, -50%) scale(1); 
            }
        }
        @keyframes thunder {
            0% {
                text-shadow: 0 0 10px rgba(255, 255, 255, 0.5), 0 0 20px rgba(255, 255, 255, 0.5),
                             0 0 30px rgba(255, 255, 255, 0.5), 0 0 40px rgba(255, 255, 255, 0.5),
                             0 0 50px rgba(255, 255, 255, 0.5);
                opacity: 0;
            }
            50% {
                text-shadow: 0 0 20px rgba(255, 255, 255, 1), 0 0 30px rgba(255, 255, 255, 1),
                             0 0 40px rgba(255, 255, 255, 1), 0 0 50px rgba(255, 255, 255, 1),
                             0 0 60px rgba(255, 255, 255, 1);
                opacity: 1;
            }
            100% {
                text-shadow: 0 0 10px rgba(255, 255, 255, 0.5), 0 0 20px rgba(255, 255, 255, 0.5),
                             0 0 30px rgba(255, 255, 255, 0.5), 0 0 40px rgba(255, 255, 255, 0.5),
                             0 0 50px rgba(255, 255, 255, 0.5);
                opacity: 0;
            }
        }
        .thunder {
            animation: thunder 6s infinite;
        }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/PCDLoader.js"></script>
</head>
<body>
    <div class="container">
        <div class="lhs">
            <div class="lhs-upper">
                <div id="live-feed">
                    <img src="{{ url_for('video_feed') }}" alt="Live Feed">
                </div>
            </div>
            <div class="lhs-lower">
                <div id="point-cloud-visualization"></div>
            </div>
        </div>
        <div class="rhs">
            <div class="rhs-upper">
                <div class="vertical-bar" id="vertical-bar">
                    <div class="fill-bar" id="fill-bar"></div>
                </div>
                <div class="step-container" id="step-container">
                    <ul class="step-list" id="step-list">
                    </ul>
                </div>
                <div class="vertical-bar-place" id="vertical-bar-place">
                    <div class="fill-bar-place" id="fill-bar-place"></div>
                </div>
                <div class="step-container-place" id="step-container-place">
                    <ul class="step-list-place" id="step-list-place">
                    </ul>
                </div>
            </div>
            <div class="rhs-lower" id="image-container">
            </div>
        </div>
        <button class="btn btn-danger reset-button" onclick="clearImages()">Reset</button>
    </div>
    <div class="centered-text thunder" id="mode-display"></div>
    <div class="centered-text-blue thunder" id="task-display"></div>
    <script>
        let stepCount = 0;
        const totalSteps = 7;
        const stepQueue = [];
        let isProcessingQueue = false;
        let imagesReceived = false;
        let stepCountPlace = 0;
        const totalStepsPlace = 7;
        const stepQueuePlace = [];
        let isProcessingQueuePlace = false;
        function clearImages() {
            if (confirm('Are you sure you want to clear all images?')) {
                fetch('/clear_images', { method: 'POST' })
                    .then(response => response.text())
                    .then(data => {
                        alert(data);
                        fetch('/clear_task', { method: 'POST' });
                        fetch('/clear_mode', { method: 'POST' });
                        location.reload();
                    });
            }
        }
        function updateImages() {
            fetch('/update_images')
                .then(response => response.json())
                .then(data => {
                    const imageContainer = document.getElementById('image-container');
                    imageContainer.innerHTML = '';
                    data.images.forEach(image => {
                        const imageCard = document.createElement('div');
                        imageCard.className = 'image-card';
                        imageCard.innerHTML = `
                            <div class="card">
                                <img src="${image.url}" class="card-img-top" alt="Received Image">
                                <div class="card-body">
                                    <p class="card-text">${image.name}</p>
                                </div>
                            </div>
                        `;
                        imageContainer.appendChild(imageCard);
                    });
                    imagesReceived = true;
                })
                .catch(error => console.error("Error updating images:", error));
        }
        function enqueueSteps(newSteps) {
            stepQueue.push(...newSteps);
            if (!isProcessingQueue) {
                processQueue();
            }
        }
        function enqueueStepsPlace(newSteps) {
            stepQueuePlace.push(...newSteps);
            if (!isProcessingQueuePlace) {
                processQueuePlace();
            }
        }
        function processQueue() {
            if (stepQueue.length > 0) {
                isProcessingQueue = true;
                const step = stepQueue.shift();
                displayStep(step);
                setTimeout(processQueue, 500);
            } else {
                isProcessingQueue = false;
            }
        }
        function processQueuePlace() {
            if (stepQueuePlace.length > 0) {
                isProcessingQueuePlace = true;
                const step = stepQueuePlace.shift();
                displayStepPlace(step);
                setTimeout(processQueuePlace, 500);
            } else {
                isProcessingQueuePlace = false;
            }
        }
        function displayStep(step) {
            const stepList = document.getElementById('step-list');
            const fillBar = document.getElementById('fill-bar');
            const verticalBar = document.getElementById('vertical-bar');
            const stepItem = document.createElement('li');
            stepItem.innerHTML = `
                <div class="step-text">${step.name}</div>
            `;
            stepList.appendChild(stepItem);
            stepCount += 1;
            const fillHeight = `${(stepCount / totalSteps) * 100}%`;
            fillBar.style.height = fillHeight;
            verticalBar.style.height = '100%';
            setTimeout(() => {
                Array.from(stepList.getElementsByClassName('new-step')).forEach((elem) => {
                    elem.classList.remove('new-step');
                });
            }, 150);
        }
        function displayStepPlace(step) {
            const stepListPlace = document.getElementById('step-list-place');
            const fillBarPlace = document.getElementById('fill-bar-place');
            const verticalBarPlace = document.getElementById('vertical-bar-place');
            const stepItem = document.createElement('li');
            stepItem.innerHTML = `
                <div class="step-text">${step.name}</div>
            `;
            stepListPlace.appendChild(stepItem);
            stepCountPlace += 1;
            const fillHeight = `${(stepCountPlace / totalStepsPlace) * 100}%`;
            fillBarPlace.style.height = fillHeight;
            verticalBarPlace.style.height = '100%';
            setTimeout(() => {
                Array.from(stepListPlace.getElementsByClassName('new-step')).forEach((elem) => {
                    elem.classList.remove('new-step');
                });
            }, 150);
        }
        function updateSteps() {
            fetch('/update_steps')
                .then(response => response.json())
                .then(data => {
                    const newSteps = data.steps.slice(stepCount);
                    enqueueSteps(newSteps);
                })
                .catch(error => console.error("Error updating steps:", error));
        }
        function updatePlaceSteps() {
            fetch('/update_place_steps')
                .then(response => response.json())
                .then(data => {
                    const newSteps = data.steps.slice(stepCountPlace);
                    enqueueStepsPlace(newSteps);
                })
                .catch(error => console.error("Error updating place steps:", error));
        }
        function fetchPointCloudData() {
            if (imagesReceived) {
                fetch('/point_cloud_data')
                    .then(response => response.blob())
                    .then(blob => blob.arrayBuffer())
                    .then(arrayBuffer => {
                        console.log('Point cloud data received:', arrayBuffer.byteLength);
                        const data = new Uint8Array(arrayBuffer);
                        visualizePointCloud(data);
                    })
                    .catch(error => console.error("Error fetching point cloud data:", error));
            }
        }
        function visualizePointCloud(data) {
            const container = document.getElementById('point-cloud-visualization');
            container.innerHTML = '';
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(0, 0, -2);
            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.25;
            controls.screenSpacePanning = false;
            controls.minDistance = 0.1;
            controls.maxDistance = 10;
            controls.maxPolarAngle = Math.PI / 2;
            controls.target.set(0, 0, 0);
            controls.update();
            const loader = new THREE.PCDLoader();
            const blob = new Blob([data.buffer], { type: 'application/octet-stream' });
            const url = URL.createObjectURL(blob);
            loader.load(url, points => {
                scene.add(points);
                function animate() {
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }
                animate();
            }, xhr => {
                console.log((xhr.loaded / xhr.total * 100) + '% loaded');
            }, error => {
                console.error('An error occurred loading the PCD file', error);
            });
            window.addEventListener('resize', onWindowResize, false);
            function onWindowResize() {
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }
        }
        function showText(displayId, text, color='yellow') {
            const displayElement = document.getElementById(displayId);
            displayElement.innerText = text;
            displayElement.style.opacity = 1;
            displayElement.style.color = color;
            displayElement.classList.add('thunder');
        }
        function clearText(displayId) {
            const displayElement = document.getElementById(displayId);
            displayElement.style.opacity = 0;
            displayElement.classList.remove('thunder');
        }
        function updateModeDisplay() {
            fetch('/get_mode')
                .then(response => response.json())
                .then(data => {
                    clearText('mode-display');
                    showText('mode-display', data.mode);
                    if (data.mode.includes('Exploration Mode Activated') || data.mode.includes('Navigation Mode Activated') || data.mode.includes('Stopped')) {
                        clearText('mode-display');
                    }
                })
                .catch(error => console.error("Error updating mode display:", error));
        }
        function updateTaskDisplay() {
            fetch('/get_task')
                .then(response => response.json())
                .then(data => {
                    clearText('task-display');
                    if (data.task.includes('Pickup Mode Activated')) {
                        document.getElementById('vertical-bar').style.display = 'block';
                        document.getElementById('step-container').style.display = 'block';
                        showText('task-display', data.task, 'red');
                        clearText('task-display');
                    }
                    if (data.task.includes('Place Mode Activated')) {
                        document.getElementById('vertical-bar-place').style.display = 'block';
                        document.getElementById('step-container-place').style.display = 'block';
                        showText('task-display', data.task, 'blue');
                        clearText('task-display');
                    }
                })
                .catch(error => console.error("Error updating task display:", error));
        }
        setInterval(updateImages, 5000);
        setInterval(updateSteps, 5000);
        setInterval(updatePlaceSteps, 5000);
        setInterval(fetchPointCloudData, 5000);
        setInterval(updateModeDisplay, 1000);
        setInterval(updateTaskDisplay, 1000);
        window.onload = function() {
            updateImages();
            updateSteps();
            updatePlaceSteps();
        }
    </script>
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
"""

steps = []
place_steps = []

@app.route('/')
def index():
    images = filter_images()
    return render_template_string(HTML_TEMPLATE, images=images)

@app.route('/images/<filename>')
def image_file(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route('/clear_images', methods=['POST'])
def clear_images():
    for filename in os.listdir(IMAGE_DIR):
        file_path = os.path.join(IMAGE_DIR, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    if os.path.exists(point_cloud_data_path):
        os.remove(point_cloud_data_path)
    return "All images and point cloud data have been cleared."

@app.route('/update_images')
def update_images():
    images = filter_images()
    image_data = [{'url': url_for('image_file', filename=image), 'name': name} for image, name in images]
    print("Returning updated images:", image_data)
    return jsonify({'images': image_data})

@app.route('/update_step', methods=['POST'])
def update_step():
    step = request.json
    if step not in steps:
        steps.append(step)
    return jsonify({'status': 'success'})

@app.route('/update_place_step', methods=['POST'])
def update_place_step():
    step = request.json
    if step not in place_steps:
        place_steps.append(step)
    return jsonify({'status': 'success'})

@app.route('/update_steps')
def update_steps():
    return jsonify({'steps': steps})

@app.route('/update_place_steps')
def update_place_steps():
    return jsonify({'steps': place_steps})

@app.route('/point_cloud_data')
def point_cloud_data():
    try:
        with open(point_cloud_data_path, 'rb') as f:
            data = f.read()
        return Response(data, mimetype='application/octet-stream')
    except Exception as e:
        print(f"Error reading PCD data: {e}")
        return "", 500

@app.route('/update_mode', methods=['POST'])
def update_mode():
    mode = request.json['mode']
    with open('mode.txt', 'w') as f:
        f.write(mode)
    return jsonify({'status': 'success'})

@app.route('/get_mode')
def get_mode():
    try:
        with open('mode.txt', 'r') as f:
            mode = f.read()
    except FileNotFoundError:
        mode = 'Select Mode'
    return jsonify({'mode': mode})

@app.route('/update_task', methods=['POST'])
def update_task():
    task = request.json['task']
    with open('task.txt', 'w') as f:
        f.write(task)
    return jsonify({'status': 'success'})

@app.route('/get_task')
def get_task():
    try:
        with open('task.txt', 'r') as f:
            task = f.read()
    except FileNotFoundError:
        task = ''
    return jsonify({'task': task})

@app.route('/clear_task', methods=['POST'])
def clear_task():
    with open('task.txt', 'w') as f:
        f.write('')
    return jsonify({'status': 'success'})

@app.route('/clear_mode', methods=['POST'])
def clear_mode():
    with open('mode.txt', 'w') as f:
        f.write('')
    return jsonify({'status': 'success'})

@app.route('/clear_mode_display', methods=['POST'])
def clear_mode_display():
    return jsonify({'status': 'success'})

@app.route('/clear_task_display', methods=['POST'])
def clear_task_display():
    return jsonify({'status': 'success'})

def filter_images():
    images = []
    for filename in os.listdir(IMAGE_DIR):
        if "object_detection" in filename:
            image_type = "Object Detection"
            images.append((filename, image_type))
        elif "semantic_segmentation" in filename:
            image_type = "Semantic Segmentation"
            images.append((filename, image_type))
    return images

def save_image(image_data, filename):
    global images_received
    image_type = None
    if "object_detection" in filename:
        image_type = "object_detection"
    elif "semantic_segmentation" in filename:
        image_type = "semantic_segmentation"
    if image_type:
        for existing_filename in os.listdir(IMAGE_DIR):
            if image_type in existing_filename:
                os.remove(os.path.join(IMAGE_DIR, existing_filename))
                break
    image_path = os.path.join(IMAGE_DIR, filename)
    with open(image_path, 'wb') as f:
        f.write(image_data)
    print(f"Saved {image_type} image: {filename}")
    images_received = True

def receive_images():
    while True:
        try:
            filename = image_receiver.recv_string()
            image_data = image_receiver.recv()
            if "object_detection" in filename or "semantic_segmentation" in filename:
                save_image(image_data, filename)
                print(f"Received and saved {filename}")
        except zmq.ZMQError as e:
            print(f"ZMQ Error: {e}, retrying...")

def receive_point_cloud_data():
    global point_cloud_data_path
    while True:
        try:
            if images_received:
                data = point_cloud_receiver.recv()
                with open(point_cloud_data_path, 'wb') as f:
                    f.write(data)
                print("Received and saved point cloud data")
        except zmq.ZMQError as e:
            print(f"ZMQ Error: {e}, retrying...")

def generate_live_feed():
    while True:
        frame = live_feed_receiver.recv()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_live_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    from threading import Thread
    Thread(target=receive_images, daemon=True).start()
    Thread(target=receive_point_cloud_data, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
