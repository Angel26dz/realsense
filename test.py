import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import pyttsx3
import threading

# Load YOLO model
model = YOLO('yolov8m.pt')

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

width, height = 640, 480
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

pipeline.start(config)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 180)
engine.setProperty('volume', 1)

# Set a voice (optional)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[11].id)

# Get depth scale
depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()


def speak_warning(text):
    def speak():
        engine.say(text)
        engine.runAndWait()

    speech_thread = threading.Thread(target=speak)
    speech_thread.start()


while True:
    # Retrieve frames from the RealSense camera
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    # Convert frames to numpy arrays
    depth = np.asanyarray(depth_frame.get_data())
    color = np.asanyarray(color_frame.get_data())

    # Apply colormap to depth frame for visualization
    depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.5), cv2.COLORMAP_JET)

    # Run YOLO detection
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
                roi_depth = depth[y1:y2, x1:x2]
                valid_depth = roi_depth[roi_depth > 0]

                if valid_depth.size > 0:
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
    if cv2.waitKey(1) == 27:
        break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()

