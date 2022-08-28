import streamlit as st
import cv2
from object_detection_utils import label_colors
import torch
import av

class VideoObjectDetection:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    @staticmethod
    def create_video_frames(video_path):
        cap = cv2.VideoCapture(video_path)

        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        if int(major_ver) < 3:
            fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()
        return frames, fps

    def prediction_label(self, image_np):
            results = self.model(image_np)
            df_result = results.pandas().xyxy[0]
            dict_result = df_result.to_dict()
            scores = list(dict_result["confidence"].values())
            labels = list(dict_result["name"].values())

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 1

            list_boxes = list()
            for dict_item in df_result.to_dict('records'):
                list_boxes.append(list(dict_item.values()))
            count = 0

            for xmin, ymin, xmax, ymax, confidence, classes, name in list_boxes:
                if confidence > .45:
                    image_np = cv2.rectangle(image_np, pt1=(int(xmin), int(ymin)), pt2=(int(xmax), int(ymax)), \
                                             color=label_colors[name], thickness=2)
                    cv2.putText(image_np, f"{labels[count]}: {round(scores[count], 2)}", (int(xmin), int(ymin) - 10), font,
                                fontScale, label_colors[name], thickness, cv2.LINE_AA)
                count = count + 1

            return image_np

    def static_vid_obj(self, frames, fps):
        image_container = st.empty()

        # Set video
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter('outputs/annotated_video.mp4', fourcc, fps, (width, height))

        for index, frame in enumerate(frames):
            # Model inference
            annotated_image = self.prediction_label(frame)
            image_container.image(annotated_image, caption=f'Frame number #{index}')
            color_adjusted_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            video.write(color_adjusted_image)

        cv2.destroyAllWindows()
        video.release()



    def callback(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        annotated_image = self.prediction_label(image)

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")