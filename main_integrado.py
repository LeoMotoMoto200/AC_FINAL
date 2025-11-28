# main_integrado.py

import cv2
import numpy as np
import speech_recognition as sr
import pyttsx3
from thefuzz import fuzz
import time

# ===================================================================
# PARTE 1: LÓGICA DE PROCESAMIENTO DE VOZ
# ===================================================================

# --- CONFIGURACIÓN DE VOZ ---
engine = pyttsx3.init()
r = sr.Recognizer()

def decir_en_ingles(texto):
    print(f"Sistema dice: '{texto}'")
    engine.say(texto)
    engine.runAndWait()

def escuchar_microfono():
    with sr.Microphone() as source:
        print("Di la palabra ahora...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source)
    try:
        texto_reconocido = r.recognize_google(audio, language="en-US")
        print(f"Tú dijiste: '{texto_reconocido}'")
        return texto_reconocido.lower()
    except sr.UnknownValueError:
        print("Lo siento, no entendí lo que dijiste.")
        return ""
    except sr.RequestError as e:
        print(f"Error con el servicio de reconocimiento de voz; {e}")
        return ""

def comparar_pronunciacion(palabra_correcta, palabra_usuario):
    similitud = fuzz.ratio(palabra_correcta.lower(), palabra_usuario.lower())
    print(f"Similitud: {similitud}%")
    return similitud > 80 # Umbral de 80% de similitud

def iniciar_practica_de_voz(palabra_objetivo):
    """Función que encapsula todo el ciclo de práctica de voz."""
    if not palabra_objetivo:
        print("No hay ningún objeto detectado para practicar.")
        decir_en_ingles("Please, show me an object first.")
        return

    print("\n--- INICIO DE LA PRÁCTICA DE PRONUNCIACIÓN ---")
    decir_en_ingles(f"Let's practice the word: {palabra_objetivo}")
    
    tu_pronunciacion = escuchar_microfono()
    
    if tu_pronunciacion:
        es_correcta = comparar_pronunciacion(palabra_objetivo, tu_pronunciacion)
        if es_correcta:
            print("✅ ¡MUY BIEN! Pronunciación correcta.")
            decir_en_ingles("Excellent! Correct pronunciation.")
        else:
            print("❌ ¡CASI! Inténtalo de nuevo.")
            decir_en_ingles("That's close. Let's try again.")
    time.sleep(1)

# ===================================================================
# PARTE 2: LÓGICA DE VISIÓN ARTIFICIAL
# ===================================================================

# --- CONFIGURACIÓN DEL MODELO YOLO ---
### CAMBIO CLAVE: Usamos los archivos que ya tienes ###
config_path = "yolov3.cfg"
weights_path = "yolov3.weights"
class_names_path = "coco.names"
confianza_minima = 0.5
nms_threshold = 0.4

objetos_permitidos = {
    "person", "dog", "cat", "bottle", "cup", "spoon", "laptop", "mouse",
    "keyboard", "cell phone", "book", "scissors"
}

with open(class_names_path, 'r') as f:
    CLASES_YOLO = [line.strip() for line in f.readlines()]

print("[INFO] Cargando modelo YOLO (versión completa)...")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

print("[INFO] Iniciando webcam...")
vs = cv2.VideoCapture(0)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ultimo_objeto_detectado = ""

# ===================================================================
# PARTE 3: BUCLE PRINCIPAL DE LA APLICACIÓN
# ===================================================================

while True:
    ret, frame = vs.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    # Usamos 320x320 para que sea más rápido
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confianza_minima:
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confianza_minima, nms_threshold)
    
    objeto_en_frame = ""

    if len(indices) > 0:
        for i in indices.flatten():
            nombre_objeto_ingles = CLASES_YOLO[class_ids[i]]
            if nombre_objeto_ingles in objetos_permitidos:
                objeto_en_frame = nombre_objeto_ingles
                (x, y, w_box, h_box) = boxes[i]
                label = f"Objeto: {nombre_objeto_ingles.capitalize()}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                break 

    if objeto_en_frame:
        ultimo_objeto_detectado = objeto_en_frame

    cv2.putText(frame, "Presiona 'p' para practicar", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Presiona 'q' para salir", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Sistema de Aprendizaje Interactivo", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    
    if key == ord('p'):
        iniciar_practica_de_voz(ultimo_objeto_detectado)

vs.release()
cv2.destroyAllWindows()