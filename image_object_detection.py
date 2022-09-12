from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from object_detection_utils import *


class ImageObjectDetection:
    """
    Object detection on images.
    """

    def __init__(self):
        """
        The constructor for ImageObjectDetection class.
        Attributes:
            feature_extractor: model for extracting features from image
            model: model for performing object detection, utilizing extracted features
        """

        self.feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
        self.model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

    def classify(self, image):
        """
        Detect objects in image.

        Parameters:
            image (PIL image): image to detect objects in
        Returns:
            viz_img (type): annotated image with predictions
            filtered_preds (type): predictions for object detection
        """

        # Extract features and perform inference
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # get logits and bounding boxes
        img_size = torch.tensor([tuple(reversed(image.size))])
        processed_outputs = self.feature_extractor.post_process(outputs, img_size)

        # Grab output predictions
        output_dict = processed_outputs[0]

        # Draw predictions on raw image
        viz_img, filtered_preds = visualize_prediction(image, output_dict, id2label=self.model.config.id2label)

        return viz_img, filtered_preds
