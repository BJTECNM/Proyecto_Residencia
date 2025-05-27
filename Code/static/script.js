let isStreaming = false;

function startStream() {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');

    if (isStreaming) return;

    fetch('/start', { method: 'POST' })
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
            stopBtn.disabled = true;
            startBtn.disabled = false;
        });
}

window.addEventListener('load', () => {
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