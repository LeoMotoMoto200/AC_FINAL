# app_gui.py (VERSIÓN FINAL PULIDA CON SONIDO Y MEJOR UI)

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
import os
import pygame ### MEJORA: Importamos pygame para el sonido ###

# ===================================================================
# LÓGICA DE PROCESAMIENTO DE VOZ (La que sí funciona siempre)
# ===================================================================
def decir_en_ingles(engine, texto):
    """Usa una instancia específica del motor TTS para hablar."""
    print(f"Sistema dice: '{texto}'")
    engine.say(texto)
    engine.runAndWait()

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
    except: return None

def comparar_pronunciacion(palabra_correcta, palabra_usuario):
    if not palabra_usuario: return False
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
        self.window.configure(bg='#f0f0f0')

        ### MEJORA: Inicializamos el mezclador de sonido de pygame ###
        pygame.mixer.init()
        try:
            self.congrats_sound = pygame.mixer.Sound("congrats.wav")
        except pygame.error:
            print("ADVERTENCIA: No se encontró el archivo 'congrats.wav'. El sonido de victoria no funcionará.")
            self.congrats_sound = None

        self.mode = 'menu'
        self.practicando = False
        
        # ... El resto del __init__ no cambia ...
        self.lesson_data = []
        self.current_lesson_index = 0
        self.cargar_datos_desafio()
        self.aciertos = 0
        self.intentos = 0
        config_path = "yolov3.cfg"
        weights_path = "yolov3.weights"
        class_names_path = "coco.names"
        self.objetos_permitidos = {"dog", "cat", "bottle", "cup", "spoon", "laptop", "mouse", "keyboard", "cell phone", "book", "scissors"}
        with open(class_names_path, 'r') as f: self.CLASES_YOLO = [line.strip() for line in f.readlines()]
        print("[INFO] Cargando modelo YOLO...")
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        self.vs = cv2.VideoCapture(0)
        self.vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ultimo_objeto_detectado = ""
        self.crear_widgets()
        self.mostrar_menu_principal()
        self.update_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def cargar_datos_desafio(self):
        path = 'lesson_images'
        if not os.path.exists(path): return
        for filename in os.listdir(path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                word = os.path.splitext(filename)[0]
                self.lesson_data.append({'word': word, 'image_path': os.path.join(path, filename)})
        print(f"Se cargaron {len(self.lesson_data)} desafíos.")

    def crear_widgets(self):
        # ... (sin cambios en la estructura básica)
        self.main_frame = tk.Frame(self.window, bg='#f0f0f0')
        self.main_frame.pack(fill="both", expand=True)
        self.video_panel = tk.Label(self.main_frame, bg='black')
        self.video_panel.pack(side="left", padx=10, pady=10)
        self.control_frame = tk.Frame(self.main_frame, bg='#ffffff', padx=20, pady=20)
        self.control_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        self.title_font = font.Font(family='Comic Sans MS', size=24, weight='bold')
        self.body_font = font.Font(family='Comic Sans MS', size=14)
        self.button_font = font.Font(family='Comic Sans MS', size=16, weight='bold')
        self.menu_title = tk.Label(self.control_frame, text="¡Aprende Jugando!", font=self.title_font, bg='white', fg='#FF6347')
        self.explore_button = tk.Button(self.control_frame, text="Modo Explorar", font=self.button_font, bg='#1E90FF', fg='white', command=self.iniciar_modo_explorar, relief=tk.FLAT)
        self.challenge_button = tk.Button(self.control_frame, text="Modo Desafío", font=self.button_font, bg='#32CD32', fg='white', command=self.iniciar_modo_desafio, relief=tk.FLAT)
        self.game_title = tk.Label(self.control_frame, text="", font=self.title_font, bg='white', fg='#1E90FF')
        self.instruction_label = tk.Label(self.control_frame, text="", font=self.body_font, bg='white', wraplength=300)
        self.image_panel = tk.Label(self.control_frame, bg='white')
        self.practice_button = tk.Button(self.control_frame, text="¡Practicar!", font=self.button_font, bg="#FFD700", command=self.iniciar_practica_thread, relief=tk.FLAT)
        self.score_label = tk.Label(self.control_frame, text="", font=self.body_font, bg='white')
        self.back_button = tk.Button(self.control_frame, text="<-- Volver al Menú", font=('Comic Sans MS', 12), command=self.mostrar_menu_principal, relief=tk.FLAT)
        
        ### MEJORA: Indicador de progreso para el desafío ###
        self.level_label = tk.Label(self.control_frame, text="", font=('Comic Sans MS', 12, 'bold'), bg='white', fg='#555555')

        ### MEJORA: Efecto Hover para los botones del menú ###
        self.explore_button.bind("<Enter>", lambda e: self.on_button_enter(e, '#58ACFA'))
        self.explore_button.bind("<Leave>", lambda e: self.on_button_leave(e, '#1E90FF'))
        self.challenge_button.bind("<Enter>", lambda e: self.on_button_enter(e, '#76E476'))
        self.challenge_button.bind("<Leave>", lambda e: self.on_button_leave(e, '#32CD32'))
    
    def on_button_enter(self, e, color): e.widget['background'] = color
    def on_button_leave(self, e, color): e.widget['background'] = color

    def limpiar_control_frame(self):
        for widget in self.control_frame.winfo_children(): widget.pack_forget()

    def mostrar_menu_principal(self):
        self.limpiar_control_frame()
        self.mode = 'menu'
        self.menu_title.pack(pady=(40, 20))
        self.explore_button.pack(pady=20, ipadx=30, ipady=15)
        self.challenge_button.pack(pady=20, ipadx=30, ipady=15)

    def iniciar_modo_explorar(self):
        self.limpiar_control_frame()
        self.mode = 'explore'
        self.game_title.config(text="Explorer Mode", fg='#1E90FF')
        self.game_title.pack(pady=(10,20))
        self.instruction_label.config(text="Show an object to the camera and press 'Practice'.")
        self.instruction_label.pack(pady=10)
        self.practice_button.pack(pady=20, ipadx=10, ipady=10)
        self.score_label.pack(pady=20)
        self.back_button.pack(side="bottom", pady=20)
        self.aciertos = 0; self.intentos = 0
        self.score_label.config(text=f"Score: {self.aciertos} / {self.intentos}")

    def iniciar_modo_desafio(self):
        if not self.lesson_data: return
        self.limpiar_control_frame()
        self.mode = 'challenge'
        self.current_lesson_index = 0
        self.game_title.config(text="Challenge Mode", fg='#32CD32')
        self.game_title.pack(pady=10)
        self.level_label.pack(pady=5)
        self.image_panel.pack(pady=15)
        self.instruction_label.pack(pady=10)
        self.back_button.pack(side="bottom", pady=20)
        self.mostrar_desafio_actual()

    def mostrar_desafio_actual(self):
        if self.current_lesson_index >= len(self.lesson_data):
            self.desafio_completado()
            return
        
        ### MEJORA: Actualiza el contador de nivel ###
        total_levels = len(self.lesson_data)
        self.level_label.config(text=f"Level {self.current_lesson_index + 1} of {total_levels}")
        
        challenge = self.lesson_data[self.current_lesson_index]
        palabra = challenge['word']
        self.instruction_label.config(text=f"What object is this?")
        img = Image.open(challenge['image_path']); img.thumbnail((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        self.image_panel.config(image=img_tk); self.image_panel.image = img_tk
        self.iniciar_practica_thread()

    def desafio_completado(self):
        self.limpiar_control_frame()
        self.game_title.config(text="CONGRATULATIONS!", fg='#FFD700', font=('Comic Sans MS', 28, 'bold'))
        self.game_title.pack(pady=40)
        self.instruction_label.config(text="You completed all the challenges!\nYou are a champion!", font=self.body_font)
        self.instruction_label.pack(pady=20)
        self.back_button.pack(side="bottom", pady=20)
        
        ### MEJORA: Reproduce el sonido de victoria ###
        if self.congrats_sound:
            self.congrats_sound.play()

    def update_frame(self):
        # ... (sin cambios aquí)
        ret, frame = self.vs.read()
        if not ret: self.window.after(15, self.update_frame); return
        if self.mode == 'explore':
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (320, 320)), 1/255.0, (320, 320), swapRB=True, crop=False)
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
            self.ultimo_objeto_detectado = ""
            if len(indices) > 0:
                for i in indices.flatten():
                    nombre_obj = self.CLASES_YOLO[class_ids[i]]
                    if nombre_obj in self.objetos_permitidos:
                        self.ultimo_objeto_detectado = nombre_obj
                        (x, y, w_box, h_box) = boxes[i]
                        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
                        cv2.putText(frame, nombre_obj, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_panel.config(image=img_tk); self.video_panel.image = img_tk
        self.window.after(15, self.update_frame)
    
    def iniciar_practica_thread(self):
        if not self.practicando:
            palabra_objetivo = ""
            if self.mode == 'explore': palabra_objetivo = self.ultimo_objeto_detectado
            elif self.mode == 'challenge':
                if self.current_lesson_index < len(self.lesson_data):
                    palabra_objetivo = self.lesson_data[self.current_lesson_index]['word']
            if palabra_objetivo:
                self.practicando = True
                thread = threading.Thread(target=self.ciclo_de_logica, args=(palabra_objetivo,))
                thread.start()
            else:
                self.practicando = False

    def ciclo_de_logica(self, palabra_objetivo):
        engine = pyttsx3.init()
        self.instruction_label.config(text=f"Say the word '{palabra_objetivo}'")
        decir_en_ingles(engine, f"Say the word: {palabra_objetivo}")
        self.instruction_label.config(text="I'm listening...")
        tu_pronunciacion = escuchar_microfono()
        es_correcta = comparar_pronunciacion(palabra_objetivo, tu_pronunciacion)
        self.window.after(0, self.procesar_resultado, es_correcta)

    def procesar_resultado(self, es_correcta):
        engine = pyttsx3.init()
        if es_correcta:
            self.instruction_label.config(text="✅ GREAT JOB!", fg="green")
            self.window.configure(bg='#d4edda')
            decir_en_ingles(engine, "Great job!")
        else:
            self.instruction_label.config(text="❌ That's close. Try again!", fg="red")
            self.window.configure(bg='#f8d7da')
            decir_en_ingles(engine, "Let's try again!")
        
        self.window.after(1500, lambda: self.window.configure(bg='#f0f0f0'))

        if self.mode == 'explore':
            self.intentos += 1
            if es_correcta: self.aciertos += 1
            self.score_label.config(text=f"Score: {self.aciertos} / {self.intentos}")

        if self.mode == 'challenge':
            if es_correcta:
                self.current_lesson_index += 1
            self.window.after(1000, self.mostrar_desafio_actual)
        
        self.practicando = False

    def on_closing(self):
        # ... (sin cambios)
        print("Cerrando aplicación...")
        self.vs.release()
        self.window.destroy()

# ===================================================================
# INICIAR LA APLICACIÓN
# ===================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = LanguageLearningApp(root, "LinguaLens")
    root.mainloop()