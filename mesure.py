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

# Carga el modelo YOLOv8 utilizado para detectar objetos.
model = YOLO('yolov8m.pt')

# Crea la canalización de RealSense y su configuración de streams.
pipeline = rs.pipeline()
config = rs.config()

# Resolución y frecuencia de captura utilizadas por color y profundidad.
width, height = 640, 480
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)


# Inicia la cámara con la configuración seleccionada.
pipeline.start(config)


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

    # Espera al siguiente conjunto de fotogramas de color y profundidad.
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    # Convierte el mapa de profundidad de RealSense a un arreglo NumPy.
    depth = np.asanyarray(depth_frame.get_data())
    # Convierte la imagen RGB/BGR de RealSense a un arreglo NumPy.
    color = np.asanyarray(color_frame.get_data())

    depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.5), cv2.COLORMAP_JET)

    # Ejecuta YOLO sobre la imagen RGB. `conf` establece la confianza mínima de detección.
    result = model(color, show=False, save=False, imgsz=640, conf=0.6)


    for frame in result:
        rendered_frame = frame.plot()
        boxes = frame.boxes

        h, w, _ = rendered_frame.shape
        c_x, c_y = w // 2, h // 2

        cv2.line(rendered_frame, (c_x, 0), (c_x, h), (0, 0, 0), 2)
        cv2.line(rendered_frame, (0, c_y), (w, c_y), (0, 0, 0), 2)

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
                    # Usa la profundidad válida mínima como estimación conservadora de la distancia más cercana.
                    min_distance = valid_depth.min() * depth_scale
                    min_dis = round(min_distance, 3)

                    if min_dis <= 1.5:
                        regions = []
                        if x1 < c_x and y1 < c_y:
                            regions.append('top left region')
                        if x2 >= c_x and y1 < c_y:
                            regions.append('top right region')
                        if x1 < c_x and y2 >= c_y:
                            regions.append('bottom left region')
                        if x2 >= c_x and y2 >= c_y:
                            regions.append('bottom right region')

                        regions = list(set(regions))

                        if len(regions) >= 3:
                            region = "it's an imminent crash"
                        elif len(regions) == 2:
                            region = "on {} and {}".format(regions[0], regions[1])
                        else:
                            region = "on {}".format(regions[0])

                        text = 'There is a {}, {}, at {} meters.'.format(classes[i], region, min_dis)
                        print(text)

                        speak_warning(text)

        res = np.hstack((rendered_frame, depth_color))

        cv2.imshow('Detection', res)

    # ESC finaliza el programa.
    if cv2.waitKey(1) == 27:
        break

# Libera la cámara y cierra las ventanas de OpenCV.
pipeline.stop()
cv2.destroyAllWindows()

