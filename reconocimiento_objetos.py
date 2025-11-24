import cv2
import numpy as np

# --- CONFIGURACIÓN DEL MODELO YOLO ---
config_path = "yolov3.cfg"
weights_path = "yolov3.weights"
class_names_path = "coco.names"
confianza_minima = 0.5
nms_threshold = 0.4

# --- CAMBIO 1: NUESTRA LISTA DE 20 OBJETOS PERMITIDOS ---
# Solo mostraremos las detecciones de estos objetos. Deben estar en inglés
# tal como aparecen en el archivo 'coco.names'.
# Usamos un "set" porque es más rápido para verificar si un elemento existe.
objetos_permitidos = {
    "person", "dog", "cat",
    "bottle", "cup", "spoon",
    "laptop", "mouse",
    "keyboard", "cell phone", "book", "scissors"
}
print(f"[INFO] Se buscarán {len(objetos_permitidos)} tipos de objetos.")


# Cargamos los nombres de TODAS las clases que YOLO puede detectar desde el archivo
with open(class_names_path, 'r') as f:
    CLASES_YOLO = [line.strip() for line in f.readlines()]

# Cargamos el modelo YOLO desde los archivos
print("[INFO] Cargando modelo YOLO...")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Obtenemos los nombres de las capas de salida
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Iniciamos la captura de video desde la webcam
print("[INFO] Iniciando webcam... Muestra un objeto a la cámara!")
vs = cv2.VideoCapture(0)

vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- BUCLE PRINCIPAL ---
while True:
    ret, frame = vs.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Pre-procesamos la imagen para YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)

    # Pasamos la imagen a la red neuronal
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    # Listas para guardar las detecciones
    boxes = []
    confidences = []
    class_ids = []

    # Iteramos sobre cada una de las capas de salida
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confianza_minima:
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicamos "Non-Maximum Suppression" para eliminar cuadros duplicados
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confianza_minima, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            
            # Obtenemos el nombre del objeto en inglés desde la lista de YOLO
            nombre_objeto_ingles = CLASES_YOLO[class_ids[i]]

            # --- CAMBIO 2: ¡AQUÍ ESTÁ EL FILTRO! ---
            # Verificamos si el objeto detectado está en nuestra lista permitida
            if nombre_objeto_ingles in objetos_permitidos:
                
                # Si está en la lista, procedemos a dibujar
                (x, y) = (boxes[i][0], boxes[i][1])
                (w_box, h_box) = (boxes[i][2], boxes[i][3])
                
                # --- CAMBIO 3: LA ETIQUETA PARA ENSEÑAR ---
                # Preparamos la etiqueta para mostrarla en pantalla.
                # Usamos .capitalize() para que la primera letra sea mayúscula.
                label = f"English: {nombre_objeto_ingles.capitalize()}"
                
                # Dibujamos el rectángulo y el texto
                color = (255, 165, 0) # Un color anaranjado/azul claro
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Mostramos el resultado en una ventana
    cv2.imshow("Aprende Inglés con Visión Artificial", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()