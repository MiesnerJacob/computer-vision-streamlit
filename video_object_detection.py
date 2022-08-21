import streamlit as st
from transformers import YolosFeatureExtractor, YolosForObjectDetection
import cv2
import torch
import numpy as np
from object_detection_utils import *


class VideoObjectDetection:
    def __init__(self):
        self.feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
        self.model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')

    @staticmethod
    def create_video_frames(video_path, skip_frames):
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        finally:
            cap.release()

        frames = frames[::skip_frames]
        return frames

    def object_detection(self, frames, skip_frames):
        image_container = st.empty()
        for index, frame in enumerate(frames):

            # Convert opencv frame to rgb for display
            rgb_frame = frame[..., ::-1]

            # Generate image object from frame numpy array
            image = Image.fromarray(np.uint8(rgb_frame))

            # Only get preds every few frames to reduce runtime
            if index % skip_frames == 0:
                # Model inference
                inputs = self.feature_extractor(images=image, return_tensors="pt")
                outputs = self.model(**inputs)

                # get logits and bounding boxes
                img_size = torch.tensor([tuple(reversed(image.size))])
                processed_outputs = self.feature_extractor.post_process(outputs, img_size)
                output_dict = processed_outputs[0]

                # Visualize prediction
                viz_img, filtered_preds = visualize_prediction(image, output_dict, id2label=self.model.config.id2label)
                image_container.image(viz_img, caption=f'frame number #{index}')