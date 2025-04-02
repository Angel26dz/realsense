import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import pyttsx3
import threading

model = YOLO('yolov8m.pt')

pipeline = rs.pipeline()
config = rs.config()

width, height = 640, 480
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)


pipeline.start(config)


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

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    depth = np.asanyarray(depth_frame.get_data())
    color = np.asanyarray(color_frame.get_data())

    depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.5), cv2.COLORMAP_JET)

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
                roi_depth = depth[y1:y2, x1:x2]
                valid_depth = roi_depth[roi_depth > 0]

                if valid_depth.size > 0:
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

    if cv2.waitKey(1) == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()

