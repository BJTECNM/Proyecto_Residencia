let isStreaming = false;

function startStream() {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');

    if (isStreaming) return;

    fetch('/start', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const img = document.getElementById('video-feed');
                img.src = `/video?${new Date().getTime()}`;
                isStreaming = true;
                startBtn.disabled = true;
                stopBtn.disabled = false;
            } else {
                alert(`Error: ${data.error}`);
                startBtn.disabled = false;
            }
        })
        .catch(() => {
            alert('No se pudo iniciar la transmisión');
            startBtn.disabled = false;
        });
}

function stopStream() {
    if (!isStreaming) return;

    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');

    fetch('/stop', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                isStreaming = false;
                // Volver a mostrar imagen de fondo
                document.getElementById('video-feed').src = "{{ url_for('static', filename='placeholder.jpg') }}?t=" + new Date().getTime();
                stopBtn.disabled = true;
                startBtn.disabled = false;
            } else {
                alert(`Error: ${data.error}`);
            }
        })
        .catch(() => {
            alert('No se pudo detener la transmisión');
            stopStream();
        });
}

// Estado inicial
window.addEventListener('load', () => {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            if (data.active) {
                document.getElementById('video-feed').src = `/video?${new Date().getTime()}`;
                isStreaming = true;
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
            }
        });
});
