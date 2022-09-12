import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import av


class HandGestureClassification:
    """
    Classify hand gestures.
    """

    def __init__(self):
        """
        The constructor for HandGestureClassification class.
        Attributes:
            mpHands: mediapipe hands objects
            hands: mediapipe hands object with arguments set
            mpDraw: mediapipe drawing utils
            model: model for performing classification of hand gestures
            classNames: names of classes that the model classifies
        """

        # initialize mediapipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mpDraw = mp.solutions.drawing_utils

        # Load the gesture recognizer model
        self.model = load_model('resources/mp_hand_gesture/')

        # Load class names
        f = open('resources/mp_hand_gesture/gesture.names', 'r')
        self.classNames = f.read().split('\n')
        f.close()

    def gesture_classification(self, frame):
        """
        Perform hand detection, classification inference, and annotate frame.

        Parameters:
            frame (av.VideoFrame): frame from webcam
        Returns:
            frame (av.VideoFrame): annotated video frame
        """

        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = self.hands.process(framergb)

        className = ''

        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                self.mpDraw.draw_landmarks(frame, handslms, self.mpHands.HAND_CONNECTIONS,
                                      self.mpDraw.DrawingSpec(color=(48, 255, 48), thickness=2, circle_radius=3),
                                      self.mpDraw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3))

                # Predict gesture
                prediction = self.model.predict([landmarks])
                classID = np.argmax(prediction)
                className = self.classNames[classID]

            # Get text cooorinates
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            text_width, text_height = cv2.getTextSize(className, font, fontScale, cv2.LINE_AA)[0]
            CenterCoordinates = (int(frame.shape[1] / 2) - int(text_width / 2)-25, 50)

            # show the prediction on the frame
            cv2.putText(frame, className, CenterCoordinates, cv2.FONT_HERSHEY_SIMPLEX,
                        1, (48, 255, 48), 2, cv2.LINE_AA)
        else:
            text = "Waiting for hand gesture..."

            # show the prediction on the frame
            cv2.putText(frame, text, (75, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 1, cv2.LINE_AA)

        return frame

    def callback(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Callback for hand gesture classification through webcam.

        Parameters:
            frame (av.VideoFrame): video frame taken from webcam
        Returns:
            annotated_frame (av.VideoFrame): video frame with annotations included
        """

        image = frame.to_ndarray(format="bgr24")
        annotated_image = self.gesture_classification(image)

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

