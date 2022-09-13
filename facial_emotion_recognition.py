import streamlit as st
import cv2
from fer import FER
import av

class FacialEmotionRecognition:
    """
    Recognize emotions on faces utilizing computer vision.
    """

    def __init__(self):
        """
        The constructor for FacialEmotionRecognition class.
        Attributes:
            fer: class, for accessing facial emotion recognitino model
            label_colors: dict, mapping dictionary for diaplying bounding box and text colors
        """

        self.fer = FER()
        self.label_colors = {
            'Neutral': (245, 114, 66),
            'Happy': (5, 245, 5),
            'Surprise': (18, 255, 215),
            'Sad': (245, 5, 49),
            'Angry': (82, 50, 168),
            'Disgust': (5, 245, 141),
            'Fear': (205, 245, 5)
        }

    def prediction_label(self, image_np):
        """
        Perform inference and annotate image.

        Parameters:
            image_np (PIL image): image to perform classification on
        Returns:
            output_image (type): PIL image, annotated image with bounding box, class, and pred probability
            clean_preds (type): dict, pred probabilities for all emotion classes
        """

        # Copy raw image and perform inference
        output_image = image_np.copy()
        results_raw = self.fer.detect_emotions(image_np)

        # Return nothing if no detections
        if results_raw == []:
            return image_np, None
        # Get detections from raw_results
        else:
            results = results_raw[0]

            # Get bounding box values
            x, y, width, height = results['box']

            # Sort emoitons by pred probability descending
            emotions = sorted(results['emotions'].items(), key=lambda x: x[1], reverse=True)

            # Title case predicted class
            class_pred = emotions[0][0].title()

            # Format pred probability as percentage
            class_prob = "{:.2%}".format(emotions[0][1])

            # Settings for drawing predictions
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 1

            # Draw face bouding box
            output_image= cv2.rectangle(output_image, pt1=(int(x), int(y)), pt2=(int(x + width), int(y + height)), \
                                     color=self.label_colors[class_pred], thickness=2)

            # Draw pred class and pred probability
            cv2.putText(output_image, f"{class_pred}: {str(class_prob)}", (int(x) - 20, int(y) - 20), font,
                        fontScale, self.label_colors[class_pred], thickness, cv2.LINE_AA)

            # Format emotion predictions as dictionary
            clean_preds = dict(emotions)

            return output_image, clean_preds

    def static_vid_fer(self, frames, fps):
        """
        Perform emotion recognition on video file.

        Parameters:
            frames (list): list of images from broken down video
            fps (int): number of frames per second in source video
        """

        # Create empty container for displaying inference in real-time
        image_container = st.empty()

        # Set video
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter('outputs/annotated_video.mp4', fourcc, fps, (width, height))

        # Iterate through video frames and perform inference
        for index, frame in enumerate(frames):
            # Model inference
            annotated_image, raw_preds = self.prediction_label(frame)

            # Display annotated image
            image_container.image(annotated_image, caption=f'Frame number #{index}')
            color_adjusted_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            video.write(color_adjusted_image)

        cv2.destroyAllWindows()
        video.release()

    def callback(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Callback for running emotion recognition through webcam.

        Parameters:
            frame (av.VideoFrame): video frame taken from webcam
        Returns:
            annotated_frame (av.VideoFrame): video frame with annotations included
        """

        image = frame.to_ndarray(format="bgr24")
        annotated_image, raw_preds = self.prediction_label(image)

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")



