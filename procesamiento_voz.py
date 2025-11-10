import speech_recognition as sr
import pyttsx3
from thefuzz import fuzz

# --- CONFIGURACIÓN DE VOZ ---
# Inicializamos el motor de Texto a Voz (TTS)
engine = pyttsx3.init()
# Opcional: Cambiar la voz si tienes varias instaladas
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id) # Elige una voz en inglés

# Inicializamos el reconocedor de voz (STT)
r = sr.Recognizer()

# --- FUNCIONES PRINCIPALES ---

def decir_en_ingles(texto):
    """Usa el motor de TTS para decir un texto en voz alta."""
    print(f"Sistema dice: '{texto}'")
    engine.say(texto)
    engine.runAndWait()

def escuchar_microfono():
    """Captura audio del micrófono y lo convierte a texto."""
    with sr.Microphone() as source:
        print("Di la palabra ahora...")
        # Ajustamos el reconocedor al ruido ambiental
        r.adjust_for_ambient_noise(source)
        # Escuchamos el audio
        audio = r.listen(source)

    try:
        # Intentamos reconocer el audio usando la API de Google
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
    """Compara dos strings y devuelve si la pronunciación es 'correcta'."""
    # Usamos fuzz.ratio para obtener un porcentaje de similitud (0 a 100)
    similitud = fuzz.ratio(palabra_correcta.lower(), palabra_usuario.lower())
    print(f"Similitud: {similitud}%")
    
    # Definimos un umbral. Si la similitud es > 85%, lo damos por bueno.
    if similitud > 85:
        return True
    else:
        return False

# --- EJEMPLO DE USO / PRUEBA ---
if __name__ == "__main__":
    # La palabra que el sistema ha reconocido con la cámara (ejemplo)
    palabra_objetivo = "bottle"

    print("--- INICIO DE LA PRUEBA DE PRONUNCIACIÓN ---")
    
    # 1. El sistema dice la palabra
    decir_en_ingles(f"Please, say the word: {palabra_objetivo}")
    
    # 2. El sistema escucha al usuario
    tu_pronunciacion = escuchar_microfono()
    
    # 3. Comparamos y damos el resultado
    if tu_pronunciacion:
        es_correcta = comparar_pronunciacion(palabra_objetivo, tu_pronunciacion)
        
        if es_correcta:
            print("\n✅ ¡MUY BIEN! Pronunciación correcta.")
            decir_en_ingles("Correct!")
            # Aquí iría el código para encender el LED verde del Arduino
        else:
            print("\n❌ ¡CASI! Inténtalo de nuevo.")
            decir_en_ingles("Try again.")
            # Aquí iría el código para encender el LED rojo del Arduino