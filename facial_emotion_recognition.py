import streamlit as st
import cv2
from fer_pytorch.fer import FER
import av

class FacialEmotionRecognition:
    def __init__(self):
        self.fer = FER()
        self.fer.get_pretrained_model("resnet34")
        self.label_colors = {
            'Neutral': (5, 5, 5),
            'Happiness': (5, 245, 5),
            'Surprise': (18, 255, 215),
            'Sadness': (245, 5, 49),
            'Anger': (82, 50, 168),
            'Disgust': (5, 245, 141),
            'Fear': (205, 245, 5)
        }

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
        output_image = image_np.copy()
        results_raw = self.fer.predict_image(image_np)

        if results_raw == []:
            return image_np, None
        else:
            results = results_raw[0]

            xmin, ymin, xmax, ymax = results['box']
            emotions = sorted(results['emotions'].items(), key=lambda x: x[1], reverse=True)
            class_pred = emotions[0][0].title()
            class_prob = "{:.2%}".format(emotions[0][1])

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 1

            output_image= cv2.rectangle(output_image, pt1=(int(xmin), int(ymin)), pt2=(int(xmax), int(ymax)), \
                                     color=self.label_colors[class_pred], thickness=2)
            cv2.putText(output_image, f"{class_pred}: {str(class_prob)}", (int(xmin) - 20, int(ymin) - 20), font,
                        fontScale, self.label_colors[class_pred], thickness, cv2.LINE_AA)

            return output_image, dict(emotions)

    def static_vid_fer(self, frames, fps):
        image_container = st.empty()

        # Set video
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter('outputs/annotated_video.mp4', fourcc, fps, (width, height))

        for index, frame in enumerate(frames):
            # Model inference
            annotated_image, raw_preds = self.prediction_label(frame)
            image_container.image(annotated_image, caption=f'Frame number #{index}')
            color_adjusted_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            video.write(color_adjusted_image)

        cv2.destroyAllWindows()
        video.release()

    def callback(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        annotated_image, raw_preds = self.prediction_label(image)

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")



