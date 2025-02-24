from tkinter import *
import pose_detection


def main():
    root = Tk()
    root.title("Pose Detection Interface")
    root.geometry("1280x720")  # Tamaño de la ventana

# Configurar la división del frame en tres partes iguales
    button_frame = Frame(root)
    button_frame.grid(row=0, column=0, sticky="nsew", padx=54, pady=54)

    start_button = Button(button_frame, text="Start", font=(
        "Arial", 16), command=lambda: pose_detection.start_detection())
    start_button.pack(padx=12, pady=24)  # Botón para iniciar la detección

    stop_button = Button(button_frame, text="Stop", font=(
        "Arial", 16), command=lambda: pose_detection.stop_detection)
    stop_button.pack(padx=12, pady=24)  # Botón para detener la detección

    # Configurar filas y columnas para centrar los botones
    button_frame.grid_rowconfigure(0, weight=1)  # Espacio entre botones
    button_frame.grid_rowconfigure(1, weight=1)  # Espacio entre botones
    button_frame.grid_columnconfigure(0, weight=1)

    root.mainloop()


if __name__ == "__main__":
    main()
