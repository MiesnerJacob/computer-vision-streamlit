from transformers import DetrFeatureExtractor, DetrForObjectDetection
import streamlit as st
import torch
from object_detection_utils import *


class ImageObjectDetection:
    def __init__(self):
        self.feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101')
        self.model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101')

    def classify(self, image):
        image_container = st.empty()
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # get logits and bounding boxes
        img_size = torch.tensor([tuple(reversed(image.size))])
        processed_outputs = self.feature_extractor.post_process(outputs, img_size)
        output_dict = processed_outputs[0]
        viz_img, filtered_preds = visualize_prediction(image, output_dict, id2label=self.model.config.id2label)
        image_container.image(viz_img)

        return viz_img, filtered_preds
