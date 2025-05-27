let isStreaming = false;

async function populateCameras() {
    const cameraSelect = document.getElementById('cameraSelect');
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');

    cameraSelect.innerHTML = '';
    videoDevices.forEach((device, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.text = device.label || `Cámara ${index + 1}`;
        cameraSelect.appendChild(option);
    });
}

function startStream() {
    const cameraIndex = document.getElementById('cameraSelect').value;
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');

    if (isStreaming) return;

    fetch('/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_index: parseInt(cameraIndex) })
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                document.getElementById('video-feed').src = `/video?${Date.now()}`;
                isStreaming = true;
                startBtn.disabled = true;
                stopBtn.disabled = false;
            } else {
                alert(`Error: ${data.error}`);
            }
        })
        .catch(() => {
            alert('No se pudo iniciar la transmisión');
        });
}

function stopStream() {
    if (!isStreaming) return;

    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');

    fetch('/stop', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                isStreaming = false;
                document.getElementById('video-feed').src = `/static/placeholder.jpg?t=${Date.now()}`;
                stopBtn.disabled = true;
                startBtn.disabled = false;
            } else {
                alert(`Error: ${data.error}`);
            }
        })
        .catch(() => {
            alert('No se pudo detener la transmisión');
        });
}

window.addEventListener('load', async () => {
    await populateCameras();

    fetch('/status')
        .then(res => res.json())
        .then(data => {
            if (data.active) {
                document.getElementById('video-feed').src = `/video?${Date.now()}`;
                isStreaming = true;
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
            }
        });
});