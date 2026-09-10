# Intel RealSense: captura de imágenes RGB y mapas de profundidad.
import pyrealsense2 as rs
# NumPy: manejo de matrices y datos de imagen.
import numpy as np
# OpenCV: visualización y procesamiento de imágenes.
import cv2
# PyTorch: operaciones utilizadas por Ultralytics/YOLO.
import torch
# Ultralytics YOLO: detección de objetos en la imagen RGB.
from ultralytics import YOLO
# pyttsx3: conversión de los mensajes de alerta de texto a voz.
import pyttsx3
# threading: permite ejecutar la síntesis de voz sin bloquear el procesamiento de vídeo.
import threading

# Load YOLO model
# Carga el modelo YOLOv8 utilizado para detectar objetos.
model = YOLO('yolov8m.pt')

# Configure RealSense pipeline
# Crea la canalización de RealSense y su configuración de streams.
pipeline = rs.pipeline()
config = rs.config()

# Resolución y frecuencia de captura utilizadas por color y profundidad.
width, height = 640, 480
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

# Inicia la cámara con la configuración seleccionada.
pipeline.start(config)

# Initialize text-to-speech engine
# Inicializa el motor de síntesis de voz.
engine = pyttsx3.init()
engine.setProperty('rate', 180)
engine.setProperty('volume', 1)

# Set a voice (optional)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[11].id)

# Obtiene la escala necesaria para convertir las unidades de profundidad del sensor a metros.
depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()


# Reproduce una advertencia mediante voz en un hilo independiente.
def speak_warning(text):
    def speak():
        engine.say(text)
        engine.runAndWait()

    speech_thread = threading.Thread(target=speak)
    speech_thread.start()


# Procesa continuamente los fotogramas hasta que el usuario presione ESC.
while True:
    # Retrieve frames from the RealSense camera
    # Espera al siguiente conjunto de fotogramas de color y profundidad.
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    # Convert frames to numpy arrays
    # Convierte el mapa de profundidad de RealSense a un arreglo NumPy.
    depth = np.asanyarray(depth_frame.get_data())
    # Convierte la imagen RGB/BGR de RealSense a un arreglo NumPy.
    color = np.asanyarray(color_frame.get_data())

    # Apply colormap to depth frame for visualization
    depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.5), cv2.COLORMAP_JET)

    # Run YOLO detection
    # Ejecuta YOLO sobre la imagen RGB. `conf` establece la confianza mínima de detección.
    result = model(color, show=False, save=False, imgsz=640, conf=0.6)

    for frame in result:
        rendered_frame = frame.plot()
        boxes = frame.boxes

        h, w, _ = rendered_frame.shape
        c_x, c_y = w // 2, h // 2

        # Draw crosshair lines
        cv2.line(rendered_frame, (c_x, 0), (c_x, h), (0, 0, 0), 2)
        cv2.line(rendered_frame, (0, c_y), (w, c_y), (0, 0, 0), 2)

        # Initialize variables to track the closest object
        closest_obj = None
        min_distance = float('inf')  # Start with a very high minimum distance

        # Process each detected box
        if boxes:
            box_coords = torch.tensor(
                [box.xyxy[0].cpu().numpy() for box in boxes],
                dtype=torch.int32,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            classes = [model.names[int(box.cls[0])] for box in boxes]

            for i, (x1, y1, x2, y2) in enumerate(box_coords.tolist()):
                # Extrae la región de profundidad correspondiente al objeto detectado.
                roi_depth = depth[y1:y2, x1:x2]
                # Descarta valores cero, que representan mediciones de profundidad no válidas.
                valid_depth = roi_depth[roi_depth > 0]

                if valid_depth.size > 0:
                    # Calcula la distancia del objeto usando el valor válido más cercano.
                    current_distance = valid_depth.min() * depth_scale
                    if current_distance < min_distance:  # Update the closest object
                        min_distance = current_distance
                        region = []

                        if x1 < c_x and y1 < c_y:
                            region.append('top left region')
                        if x2 >= c_x and y1 < c_y:
                            region.append('top right region')
                        if x1 < c_x and y2 >= c_y:
                            region.append('bottom left region')
                        if x2 >= c_x and y2 >= c_y:
                            region.append('bottom right region')

                        region = list(set(region))

                        if len(region) >= 3:
                            region_text = "it's an imminent crash"
                        elif len(region) == 2:
                            region_text = "on {} and {}".format(region[0], region[1])
                        else:
                            region_text = "on {}".format(region[0])

                        closest_obj = (classes[i], region_text, round(current_distance, 3))

        # Warn about the closest object
        if closest_obj and min_distance <= 1.5:
            text = 'There is a {}, {}, at {} meters.'.format(*closest_obj)
            print(text)
            speak_warning(text)

        # Display the frames
        res = np.hstack((rendered_frame, depth_color))
        cv2.imshow('Detection', res)

    # Exit on pressing 'ESC'
    # ESC finaliza el programa.
    if cv2.waitKey(1) == 27:
        break

# Cleanup
# Libera la cámara y cierra las ventanas de OpenCV.
pipeline.stop()
cv2.destroyAllWindows()

