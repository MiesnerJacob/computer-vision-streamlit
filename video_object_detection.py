import streamlit as st
import cv2
from object_detection_utils import label_colors
import torch
import av

class VideoObjectDetection:
    """
    Object detection on videos.
    """

    def __init__(self):
        """
        The constructor for VideoObjectDetection class.
        Attributes:
            model: model for detecting objects
        """

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def prediction_label(self, image_np):
        """
        Run object detection on a single frame.

        Parameters:
            image_np (np.array): numpy array of image
        Returns:
            image_np (np.array): annotated image
        """

        # Run inference
        results = self.model(image_np)

        # Extract delevant data from result
        df_result = results.pandas().xyxy[0]
        dict_result = df_result.to_dict()
        scores = list(dict_result["confidence"].values())
        labels = list(dict_result["name"].values())

        # Set variables for drawing on raw image
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 1

        # Create a list of all the bounding boxes for detected objects
        list_boxes = list()
        for dict_item in df_result.to_dict('records'):
            list_boxes.append(list(dict_item.values()))
        count = 0

        # Iterate through identified objects
        for xmin, ymin, xmax, ymax, confidence, classes, name in list_boxes:
            # Title case class name
            name = name.title()

            # draw prediction on image if model pred probability is above .45
            if confidence > .45:
                # Draw bounding box on image
                image_np = cv2.rectangle(image_np, pt1=(int(xmin), int(ymin)), pt2=(int(xmax), int(ymax)), \
                                         color=label_colors[name], thickness=2)
                # Draw class and pred probability on image
                cv2.putText(image_np, f"{str(labels[count]).title()}: {round(scores[count], 2)}", (int(xmin), int(ymin) - 10), font,
                            fontScale, label_colors[name], thickness, cv2.LINE_AA)
            count = count + 1

        return image_np

    def static_vid_obj(self, frames, fps):
        """
        Run object detection on video file.

        Parameters:
            frames (list): list of images from broken down video
            fps (int): number of frames per second in source video
        """

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
        """
        Callback for running object detection through webcam.

        Parameters:
            frame (av.VideoFrame): video frame taken from webcam
        Returns:
            annotated_frame (av.VideoFrame): video frame with annotations included
        """

        image = frame.to_ndarray(format="bgr24")
        annotated_image = self.prediction_label(image)

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")