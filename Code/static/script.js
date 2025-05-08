let isStreaming = false;

function startStream() {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');

    startBtn.disabled = true;
    stopBtn.disabled = false;

    fetch('/start', {method: 'POST'})
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('video-feed').src = 
                    `/video?${new Date().getTime()}`;
                isStreaming = true;
            } else {
                alert(`Error: ${data.error}`);
                startBtn.disabled = false;
            }
        })
        .catch(err => {
            console.error(err);
            alert('Error en el servidor');
            startBtn.disabled = false;
        });
}

function stopStream() {
    if (!isStreaming) return;

    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');

    stopBtn.disabled = true;
    startBtn.disabled = false;

    fetch('/stop', {method: 'POST'})
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('video-feed').src = '';
                isStreaming = false;
            } else {
                alert(`Error: ${data.error}`);
                stopBtn.disabled = false;
            }
        })
        .catch(err => {
            console.error(err);
            alert('Error en el servidor');
            stopBtn.disabled = false;
        });
}

// Estado inicial al cargar
window.addEventListener('load', () => {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            if (data.active) {
                document.getElementById('video-feed').src = `/video?${new Date().getTime()}`;
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                isStreaming = true;
            }
        });
});
