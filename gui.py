import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import sys, subprocess

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Apri l'immagine
        image = Image.open(file_path)
        
        # Ottieni le dimensioni della finestra
        window_width = root.winfo_width()
        window_height = root.winfo_height()

        # Calcola le dimensioni dell'immagine mantenendo le proporzioni
        image_width, image_height = image.size
        aspect_ratio = image_width / image_height

        if image_width > window_width or image_height > window_height:
            if image_width / window_width > image_height / window_height:
                new_width = window_width
                new_height = int(window_width / aspect_ratio)
            else:
                new_height = window_height
                new_width = int(window_height * aspect_ratio)
        else:
            new_width, new_height = image_width, image_height

        # Ridimensiona l'immagine
        image = image.resize((new_width, new_height), Image.LANCZOS)
        image = ImageTk.PhotoImage(image)

        # Mostra l'immagine nella label
        label_image.config(image=image)
        label_image.image = image  # Mantieni un riferimento all'immagine

        # Avvia il processo del modulo principale in un terminale separato
        if sys.platform == "win32":
            # Per Windows
            command = f'python init.py "{file_path}"'
            subprocess.Popen(['start', 'cmd', '/k', command], shell=True)
        elif sys.platform == "darwin":
            # Per macOS
            command = f'python3 init.py "{file_path}"'
            subprocess.Popen(['osascript', '-e', f'tell app "Terminal" to do script "{command}"'])
        else:
            # Per Linux con GNOME Terminal
            command = f'python3 init.py "{file_path}"'
            subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', f'{command}; exec bash'])

# Crea la finestra principale
root = tk.Tk()
root.title("Carica Immagine")

# Imposta la dimensione della finestra
window_width = 800
window_height = 600

# Ottieni la dimensione dello schermo
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calcola le coordinate per centrare la finestra
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2

# Imposta la geometria della finestra
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Frame principale per il contenuto
frame = tk.Frame(root)
frame.pack(expand=True, fill='both')

# Bottone per caricare il file
button_load = tk.Button(frame, text="Carica Immagine", command=load_image)
button_load.pack(pady=10)

# Label per mostrare l'immagine
label_image = tk.Label(frame)
label_image.pack(expand=True)

# Avvia il loop principale dell'interfaccia grafica
root.mainloop()
