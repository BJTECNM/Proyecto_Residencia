let isStreaming = false;
let ejercicioActual = "flexion_codo";  // valor por defecto
let repeticiones = 0; // valor por defecto
let feedbackTexto = "Esperando indicaciones..."; // valor por defecto
let progreso = 0;
let esperando = false;
let barraInterval = null;


document.querySelectorAll('.btn-ejercicio').forEach(btn => {
    btn.addEventListener('click', async () => {
        document.querySelectorAll('.btn-ejercicio').forEach(b => b.classList.remove('activo'));
        btn.classList.add('activo');
        ejercicioActual = btn.dataset.ejercicio;

        // Detener la transmisión si está activa
        if (isStreaming) {
            await stopStream();
        }

        // Reiniciar contador y feedback en pantalla
        document.getElementById('reps').textContent = "0";
        document.getElementById('feedback').textContent = "Esperando indicaciones...";
    });
});

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
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const img = document.getElementById('video-feed');

    if (isStreaming) return;

    const cameraIndex = document.getElementById('cameraSelect').value;

    fetch('/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            camara_index: parseInt(cameraIndex),
            ejercicio: ejercicioActual
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                img.classList.add('fade-out');
                setTimeout(() => {
                    img.src = `/video?${new Date().getTime()}`;
                    img.classList.remove('fade-out');
                }, 300);

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
    const img = document.getElementById('video-feed');

    fetch('/stop', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                isStreaming = false;

                // Fade out antes de mostrar el placeholder
                img.classList.add('fade-out');

                setTimeout(() => {
                    img.src = "/static/placeholder.jpg?t=" + new Date().getTime();
                    img.classList.remove('fade-out');
                }, 300);

                stopBtn.disabled = true;
                startBtn.disabled = false;
            } else {
                alert(`Error: ${data.error}`);
            }
        })
        .catch(() => {
            alert('No se pudo detener la transmisión');
            stopBtn.disabled = false;
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

// Actualiza contador de repeticiones y feedback cada segundo
setInterval(() => {
    if (isStreaming) {
        fetch('/contador')
            .then(res => res.json())
            .then(data => {
                document.getElementById('reps').textContent = data.repeticiones;
            });

        fetch('/feedback')
            .then(res => res.json())
            .then(data => {
                document.getElementById('feedback').textContent = data.mensaje;
            });
    }
}, 1000);

function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    const themeButton = document.getElementById('themeButton');
    const themeIcon = document.getElementById('themeIcon');

    // Animación de rotación
    themeButton.classList.add('animating');

    // Cambiar ícono con un ligero retraso para que coincida con la animación
    setTimeout(() => {
        themeIcon.textContent = newTheme === 'dark' ? '🌙' : '☀️';
        themeButton.classList.remove('animating');
    }, 300);

    // Aplicar tema
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
}

// Aplicar el tema guardado al cargar
window.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    document.getElementById('themeIcon').textContent = savedTheme === 'dark' ? '🌙' : '☀️';
});