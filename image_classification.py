from transformers import ViTFeatureExtractor, ViTForImageClassification
import streamlit as st
import pandas as pd
import numpy as np
from torch.nn import Softmax as softmax


class ImageClassification:
    def __init__(self):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    def classify(self, image):
        image_container = st.empty()
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        pred_probs = softmax(dim=1)(outputs.logits)

        predictions = pd.DataFrame(columns=['Class', 'Pred_Prob'])
        for i in range(1, 4):
            predicted_class_idx = np.argsort(np.max(pred_probs.cpu().detach().numpy(), axis=0))[-i]
            predicted_class_pred_prob = float(pred_probs[(0, predicted_class_idx)].detach().numpy())
            predicted_class_name = self.model.config.id2label[predicted_class_idx]
            predictions = predictions.append({'Class': predicted_class_name, 'Pred_Prob': predicted_class_pred_prob}, ignore_index=True)
        image_container.image(image)

        return predictions