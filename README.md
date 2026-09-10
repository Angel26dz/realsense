# Sistema de asistencia visual mediante visión artificial y percepción de profundidad

Sistema experimental de asistencia para personas ciegas o con dificultades visuales, basado en **visión artificial, percepción de profundidad y síntesis de voz**.

El proyecto utiliza una cámara **Intel RealSense** para obtener simultáneamente imágenes RGB y profundidad, un modelo **YOLOv8** para detectar objetos y **pyttsx3** para comunicar mediante voz la presencia y distancia aproximada de los obstáculos detectados.

> **Estado del proyecto:** prototipo académico/experimental. No debe utilizarse como único sistema de orientación o seguridad para una persona en un entorno real.

## 🎯 Objetivo

Desarrollar una prueba de concepto capaz de:

- Detectar objetos presentes frente al usuario mediante inteligencia artificial.
- Obtener una estimación de la distancia de cada objeto utilizando el sensor de profundidad de la RealSense.
- Identificar aproximadamente la zona del campo visual donde se encuentra el obstáculo.
- Priorizar el objeto más cercano para reducir la cantidad de información transmitida al usuario.
- Generar alertas mediante síntesis de voz cuando un objeto se encuentra a una distancia considerada de riesgo.

## 🧠 Funcionamiento

El flujo principal del sistema es:

```text
Intel RealSense
      │
      ├── Imagen RGB ──────► YOLOv8 ──────► Detección de objetos
      │                                      │
      └── Profundidad ───────────────────────┤
                                             ▼
                                  Distancia del objeto
                                             │
                                             ▼
                                    Evaluación de riesgo
                                             │
                                             ▼
                                      Síntesis de voz
                                             │
                                             ▼
                                  Alerta para el usuario
```

Para cada fotograma, el sistema obtiene las imágenes RGB y de profundidad. YOLOv8 genera las detecciones mediante cuadros delimitadores. Posteriormente, se consulta la información de profundidad correspondiente al área detectada y se utiliza el valor válido mínimo como estimación conservadora de la distancia más cercana dentro del objeto.

Cuando la distancia es igual o inferior a **1,5 m**, se genera una advertencia por voz indicando el objeto, su posición aproximada en la imagen y la distancia.

## 📁 Estructura del proyecto

```text
realsense-main/
├── mesure.py
├── test.py
├── say.txt
└── pro_vision/
    ├── mesure.py
    └── test.py
```

### `mesure.py`
Versión inicial del sistema. Detecta todos los objetos que se encuentran dentro del umbral de distancia y genera una advertencia para cada uno.

### `test.py`
Versión experimental que selecciona el **objeto más cercano** entre las detecciones antes de generar la advertencia. Esta estrategia reduce la cantidad de mensajes de voz y puede resultar más apropiada para una interfaz asistiva.

### `pro_vision/`
Contiene versiones de prueba utilizadas durante el desarrollo del proyecto.

## 🛠️ Tecnologías utilizadas

- **Python 3**
- **Intel RealSense SDK (`pyrealsense2`)** — adquisición de imagen RGB y profundidad.
- **OpenCV** — procesamiento y visualización de imágenes.
- **NumPy** — manipulación de matrices de imagen y profundidad.
- **PyTorch** — operaciones asociadas al modelo de detección.
- **Ultralytics YOLOv8** — detección de objetos mediante visión artificial.
- **pyttsx3** — conversión de texto a voz.
- **Threading** — ejecución de la síntesis de voz sin detener completamente el procesamiento de vídeo.

## 📦 Requisitos

Se requiere:

- Cámara **Intel RealSense** compatible con `pyrealsense2`.
- Python 3.
- Un entorno con soporte para las dependencias de PyTorch/Ultralytics.
- Modelo `yolov8m.pt` en el directorio desde el que se ejecuta el programa.
- Salida de audio disponible para `pyttsx3`.

Instalación básica de las dependencias de Python:

```bash
pip install numpy opencv-python torch ultralytics pyttsx3
```

La instalación de `pyrealsense2` puede depender del sistema operativo y de la versión del SDK de Intel RealSense utilizada. Se recomienda instalar el SDK siguiendo la documentación oficial de Intel RealSense.

## ▶️ Ejecución

Con la cámara conectada y el modelo disponible:

```bash
python test.py
```

También puede ejecutarse la versión inicial:

```bash
python mesure.py
```

Para finalizar la aplicación, presione **ESC** en la ventana de OpenCV.

## ⚙️ Parámetros principales

En los scripts se pueden modificar los siguientes parámetros:

```python
width, height = 640, 480
```

Resolución utilizada por los streams de color y profundidad.

```python
conf=0.6
```

Confianza mínima utilizada por YOLO para conservar una detección.

```python
if closest_obj and min_distance <= 1.5:
```

Distancia máxima a partir de la cual se genera una alerta. Actualmente está configurada en **1,5 metros**.

```python
engine.setProperty('rate', 180)
engine.setProperty('volume', 1)
```

Configuración de velocidad y volumen de la síntesis de voz.

## 🔊 Ejemplo de alerta

El sistema puede generar mensajes con una estructura similar a:

```text
There is a person, on top left region, at 0.82 meters.
```

La aplicación imprime el mensaje por consola y simultáneamente intenta reproducirlo mediante síntesis de voz.

## 🔬 Consideraciones técnicas

### Estimación de profundidad

La profundidad entregada por la RealSense está expresada inicialmente en unidades del sensor. El programa obtiene automáticamente el `depth_scale` del dispositivo y lo utiliza para convertir la medición a metros.

### Selección del objeto más cercano

`test.py` compara la distancia de las detecciones y conserva únicamente aquella con menor distancia válida. Esto permite concentrar la salida auditiva en el obstáculo potencialmente más inmediato.

### Ubicación del obstáculo

La imagen se divide conceptualmente utilizando un punto central. Según la posición y extensión del cuadro delimitador, el sistema determina si el objeto se encuentra en la parte superior/inferior e izquierda/derecha del campo visual.

## 🚧 Limitaciones actuales

Este proyecto es un prototipo y presenta varias limitaciones importantes:

- La detección depende del modelo YOLO utilizado y de las clases que este haya aprendido.
- La distancia se estima a partir de los valores de profundidad disponibles dentro del cuadro delimitador, por lo que puede verse afectada por ruido, superficies reflectantes, oclusiones y errores del sensor.
- El sistema utiliza una representación 2D de la posición del objeto y no realiza navegación autónoma.
- Las alertas pueden repetirse rápidamente porque el sistema procesa continuamente los fotogramas.
- La selección de la voz mediante `voices[11]` depende de las voces instaladas en el sistema y puede provocar un error en equipos donde ese índice no exista.
- La síntesis de voz se ejecuta en hilos independientes, por lo que una escena con muchas detecciones puede producir múltiples mensajes simultáneos.
- El sistema no sustituye un bastón, perro guía, dispositivo médico ni otro mecanismo de movilidad asistida certificado.

## 🔮 Posibles mejoras

Entre las siguientes etapas de desarrollo se plantean:

- Implementar control de frecuencia de las alertas para evitar mensajes repetitivos.
- Utilizar una cola de audio para reproducir advertencias de forma ordenada.
- Incorporar seguimiento temporal de objetos para estabilizar las detecciones.
- Mejorar la estimación de profundidad utilizando filtros estadísticos o la mediana de la región en lugar del mínimo absoluto.
- Incorporar zonas de riesgo configurables según la distancia y posición del obstáculo.
- Utilizar un modelo entrenado específicamente para los objetos relevantes para asistencia visual.
- Añadir clasificación de riesgo y diferentes patrones de alerta.
- Optimizar el procesamiento para dispositivos embebidos con GPU, como NVIDIA Jetson.
- Diseñar una interfaz de audio más natural y menos intrusiva.

## 📚 Contexto académico

El proyecto integra conceptos de **visión artificial, inteligencia artificial, procesamiento de imágenes, percepción 3D, sensores de profundidad y sistemas de asistencia**. Su propósito principal es explorar cómo la combinación de detección visual y retroalimentación auditiva puede transformar información del entorno en señales comprensibles para una persona con discapacidad visual.

## ⚠️ Aviso de seguridad

Este repositorio presenta una implementación experimental. Las mediciones, detecciones y alertas no deben considerarse suficientemente confiables para garantizar la seguridad de una persona. Cualquier aplicación destinada a asistencia real debe someterse a pruebas exhaustivas, validación de hardware y software, análisis de riesgos y las certificaciones correspondientes.

## 👤 Autor

Proyecto desarrollado como iniciativa académica y experimental en el área de **electrónica, robótica, visión artificial e inteligencia artificial**.
