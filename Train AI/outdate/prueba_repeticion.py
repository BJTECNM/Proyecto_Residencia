import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Cargar el archivo .npy
data = np.load("data/sentadilla/sentadilla_20250610_181551.npy")

# Asumimos que los primeros 66 valores son coordenadas x, y de 33 puntos
frames = data[:, :66].reshape(-1, 33, 2)

# Crear figura
fig, ax = plt.subplots()
scat = ax.scatter([], [], s=40)

def init():
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    return scat,

def update(frame):
    scat.set_offsets(frame)
    return scat,

ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=50)
plt.show()
