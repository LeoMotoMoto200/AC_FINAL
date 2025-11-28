# app_gui.py (CORREGIDO)

import tkinter as tk
from tkinter import font
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import speech_recognition as sr
import pyttsx3
from thefuzz import fuzz

# ===================================================================
# LÓGICA DE PROCESAMIENTO DE VOZ
# ===================================================================
# NOTA: Ya no inicializamos el 'engine' aquí de forma global.

# CAMBIO 1: La función ahora recibe el motor de voz como parámetro.
def decir_en_ingles(engine, texto):
    """Usa una instancia específica del motor TTS para hablar."""
    print(f"Sistema dice: '{texto}'")
    engine.say(texto)
    engine.runAndWait()

# El resto de las funciones de voz permanecen igual
r = sr.Recognizer()

def escuchar_microfono():
    with sr.Microphone() as source:
        print("Di la palabra ahora...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source)
    try:
        texto_reconocido = r.recognize_google(audio, language="en-US")
        print(f"Tú dijiste: '{texto_reconocido}'")
        return texto_reconocido.lower()
    except: return ""

def comparar_pronunciacion(palabra_correcta, palabra_usuario):
    similitud = fuzz.ratio(palabra_correcta.lower(), palabra_usuario.lower())
    print(f"Similitud: {similitud}%")
    return similitud > 80

# ===================================================================
# CLASE PRINCIPAL DE LA APLICACIÓN GUI
# ===================================================================
class LanguageLearningApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.practicando = False
        self.aciertos = 0
        self.intentos = 0
        
        # --- MODELO YOLO ---
        config_path = "yolov3.cfg"
        weights_path = "yolov3.weights"
        class_names_path = "coco.names"
        
        self.objetos_permitidos = {"person", "dog", "cat", "bottle", "cup", "spoon", "laptop", "mouse", "keyboard", "cell phone", "book", "scissors"}
        with open(class_names_path, 'r') as f: self.CLASES_YOLO = [line.strip() for line in f.readlines()]
        print("[INFO] Cargando modelo YOLO (versión completa)...")
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        
        self.vs = cv2.VideoCapture(0)
        self.vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.ultimo_objeto_detectado = ""
        self.practicando = False

        # --- LAYOUT DE LA INTERFAZ --- (Sin cambios aquí)
        self.video_panel = tk.Label(window)
        self.video_panel.pack(side="left", padx=10, pady=10)
        control_frame = tk.Frame(window)
        control_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        title_font = font.Font(family='Helvetica', size=16, weight='bold')
        title_label = tk.Label(control_frame, text="Aprendizaje Interactivo", font=title_font)
        title_label.pack(pady=10)
        self.objeto_label = tk.Label(control_frame, text="Objeto Detectado: Ninguno", font=('Helvetica', 12))
        self.objeto_label.pack(pady=20)
        self.practice_button = tk.Button(control_frame, text="¡Practicar Pronunciación!", font=('Helvetica', 12, 'bold'), command=self.iniciar_practica_thread, bg="#4CAF50", fg="white")
        self.practice_button.pack(pady=10, ipadx=10, ipady=10)
        self.status_label = tk.Label(control_frame, text="Muestra un objeto a la cámara.", font=('Helvetica', 11), wraplength=250)
        self.status_label.pack(pady=20)
        score_font = font.Font(family='Helvetica', size=12)
        self.score_label = tk.Label(control_frame, text="Puntuación: 0 / 0", font=score_font)
        self.score_label.pack(pady=15)

        self.update_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_frame(self):
        # Esta función no tiene cambios, es solo para el video
        ret, frame = self.vs.read()
        if ret:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
            self.net.setInput(blob)
            layer_outputs = self.net.forward(self.output_layers)
            
            boxes, confidences, class_ids = [], [], []
            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        box = detection[0:4] * np.array([w, h, w, h])
                        (centerX, centerY, width, height) = box.astype("int")
                        x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            objeto_en_frame = ""
            if len(indices) > 0:
                for i in indices.flatten():
                    nombre_obj = self.CLASES_YOLO[class_ids[i]]
                    if nombre_obj in self.objetos_permitidos:
                        objeto_en_frame = nombre_obj
                        (x, y, w_box, h_box) = boxes[i]
                        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                        cv2.putText(frame, nombre_obj, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        break

            if objeto_en_frame:
                self.ultimo_objeto_detectado = objeto_en_frame
                self.objeto_label.config(text=f"Objeto Detectado: {objeto_en_frame.capitalize()}")
            else:
                 self.objeto_label.config(text="Objeto Detectado: Ninguno")
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.video_panel.imgtk = img_tk
            self.video_panel.config(image=img_tk)
        
        self.window.after(15, self.update_frame)
    
    def iniciar_practica_thread(self):
        if not self.practicando:
            self.practicando = True
            thread = threading.Thread(target=self.ciclo_de_voz)
            thread.start()

    def ciclo_de_voz(self):
        """Función de voz que se ejecuta en un hilo separado con su propio motor."""
        # CAMBIO 2: ¡Aquí está la magia! Creamos un motor nuevo y limpio.
        engine = pyttsx3.init()
        
        palabra_objetivo = self.ultimo_objeto_detectado
        if not palabra_objetivo:
            self.status_label.config(text="Por favor, muestra un objeto primero.")
            decir_en_ingles(engine, "Please, show me an object first.") # Le pasamos el motor
            self.practicando = False
            return

        self.status_label.config(text=f"Practicando: {palabra_objetivo.capitalize()}")
        # CAMBIO 3: Usamos el motor local para cada llamada de voz.
        decir_en_ingles(engine, f"Let's practice the word: {palabra_objetivo}")
        
        self.status_label.config(text="Escuchando...")
        tu_pronunciacion = escuchar_microfono()
        
        if tu_pronunciacion:
            self.intentos += 1 # Aumentar intentos solo si el usuario habla
            es_correcta = comparar_pronunciacion(palabra_objetivo, tu_pronunciacion)
            if es_correcta:
                self.aciertos += 1
                self.status_label.config(text="✅ ¡Correcto! ¡Muy bien!", fg="green")
                decir_en_ingles(engine, "Excellent! Correct pronunciation.")
                self.score_label.config(text=f"Puntuación: {self.aciertos} / {self.intentos}")
                self.window.configure(bg='#d4edda') # Un verde claro
                self.window.after(1000, lambda: self.window.configure(bg='SystemButtonFace')) # Volver al color original
            else:
                self.status_label.config(text="❌ ¡Casi! Intenta de nuevo.", fg="red")
                decir_en_ingles(engine, "That's close. Let's try again.")
                self.score_label.config(text=f"Puntuación: {self.aciertos} / {self.intentos}")
                self.window.configure(bg='#f8d7da') # Un rojo claro
                self.window.after(1000, lambda: self.window.configure(bg='SystemButtonFace')) # Volver al color original
        else:
            self.status_label.config(text="No pude entender. Inténtalo de nuevo.", fg="orange")
            
        
        time.sleep(2)
        self.status_label.config(text="Muestra un objeto a la cámara.", fg="black")
        self.practicando = False
        
    def on_closing(self):
        print("Cerrando aplicación...")
        self.vs.release()
        self.window.destroy()

# ===================================================================
# INICIAR LA APLICACIÓN
# ===================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = LanguageLearningApp(root, "Sistema Interactivo de Aprendizaje de Idiomas")
    root.mainloop()