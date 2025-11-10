import cv2
import numpy as np

# --- NUESTRA "BASE DE DATOS" DE PALABRAS ---
# Mapea el nombre del objeto reconocido en inglés a su traducción o información.
# Agreguen aquí los objetos que quieren que su sistema reconozca.
diccionario_palabras = {
    "bottle": "Botella",
    "chair": "Silla",
    "cat": "Gato",
    "dog": "Perro",
    "person": "Persona",
    "tvmonitor": "Monitor de TV", # El modelo a veces lo reconoce así
    "laptop": "Laptop",
    "mouse": "Ratón",
    "keyboard": "Teclado",
    "cellphone": "Celular",
    "book": "Libro",
    "cup": "Taza"
}

# --- CONFIGURACIÓN DEL MODELO DE VISIÓN ARTIFICIAL ---
# Nombres de los archivos que descargaste
prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"
confianza_minima = 0.4 # Umbral de confianza para mostrar la detección

# Lista de clases que el modelo puede reconocer
CLASES = ["background", "aeroplane", "bicycle", "bird", "boat",
          "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
          "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
          "sofa", "train", "tvmonitor"]

# Cargamos el modelo desde los archivos
print("[INFO] Cargando modelo...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Iniciamos la captura de video desde la webcam
print("[INFO] Iniciando webcam...")
vs = cv2.VideoCapture(0) # El 0 indica la webcam por defecto

# --- BUCLE PRINCIPAL ---
while True:
    # Leemos un frame (una imagen) de la webcam
    ret, frame = vs.read()
    if not ret:
        break

    # Obtenemos las dimensiones del frame
    (h, w) = frame.shape[:2]
    
    # Pre-procesamos la imagen para que el modelo la entienda
    # Cambiamos su tamaño a 300x300 y normalizamos
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pasamos la imagen procesada a la red neuronal
    net.setInput(blob)
    detections = net.forward()

    # Iteramos sobre las detecciones encontradas
    for i in np.arange(0, detections.shape[2]):
        # Extraemos la confianza (probabilidad) de la detección
        confidence = detections[0, 0, i, 2]

        # Si la confianza es mayor que nuestro mínimo...
        if confidence > confianza_minima:
            # Obtenemos el ID de la clase y el nombre del objeto
            idx = int(detections[0, 0, i, 1])
            nombre_objeto = CLASES[idx]

            # Verificamos si el objeto está en nuestra "base de datos"
            if nombre_objeto in diccionario_palabras:
                # Calculamos las coordenadas del rectángulo para dibujar sobre el objeto
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Preparamos el texto a mostrar
                label = f"Objeto: {nombre_objeto}"

                # Dibujamos el rectángulo y el texto en la pantalla
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Mostramos el resultado en una ventana
    cv2.imshow("Reconocimiento de Objetos - Avance", frame)
    
    # Si se presiona la tecla 'q', salimos del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos la cámara y cerramos las ventanas
vs.release()
cv2.destroyAllWindows()