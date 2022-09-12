import streamlit as st
from streamlit_option_menu import option_menu
from video_object_detection import VideoObjectDetection
from image_object_detection import ImageObjectDetection
from facial_emotion_recognition import FacialEmotionRecognition
from hand_gesture_classification import HandGestureClassification
from image_optical_character_recgonition import ImageOpticalCharacterRecognition
from image_classification import ImageClassification
import plotly.express as px
from PIL import Image
from io import BytesIO
import base64
import json
import os
import cv2
import numpy as np
from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    WebRtcStreamerContext,
    webrtc_streamer,
)

# Hide warnings to make it easier to locate
# errors in logs, should they show up
import warnings
warnings.filterwarnings("ignore")

# Hide Streamlit logo
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Make Radio buttons horizontal
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# Functions to load models
@st.cache(allow_output_mutation=True)
def load_video_object_detection():
    return VideoObjectDetection()

@st.cache(allow_output_mutation=True)
def load_image_object_detection():
    return ImageObjectDetection()

@st.cache(allow_output_mutation=True)
def load_image_classifier():
    return ImageClassification()

@st.cache(allow_output_mutation=True)
def load_facial_emotion_classifier():
    return FacialEmotionRecognition()

@st.cache(allow_output_mutation=True)
def load_hand_gesture_classifier():
    return HandGestureClassification()

@st.cache(allow_output_mutation=True)
def load_image_optical_character_recognition():
    return ImageOpticalCharacterRecognition()


# Load models and store in cache
video_object_detection = load_video_object_detection()
image_object_detection = load_image_object_detection()
facial_emotion_classifier = load_facial_emotion_classifier()
hand_gesture_classifier = load_hand_gesture_classifier()
image_optical_character_recognition = load_image_optical_character_recognition()
image_classifier = load_image_classifier()

# Paths for image examples
image_examples = {'Traffic': 'examples/Traffic.jpeg',
                  'Barbeque': 'examples/Barbeque.jpeg',
                  'Home Office': 'examples/Home Office.jpeg',
                  'Car': 'examples/Car.jpeg',
                  'Dog': 'examples/Dog.jpeg',
                  'Tropics': 'examples/Tropics.jpeg',
                  'Quick Brown Dog': 'examples/Quick Brown Dog.png',
                  'Receipt': 'examples/Receipt.png',
                  'Street Sign': 'examples/Street Sign.jpeg',
                  'Kanye': 'examples/Kanye.png',
                  'Shocked': 'examples/Shocked.png',
                  'Yelling': 'examples/Yelling.jpeg'}

# Paths for video examples
video_examples = {'Traffic': 'examples/Traffic.mp4',
                  'Elephant': 'examples/Elephant.mp4',
                  'Airport': 'examples/Airport.mp4',
                  'Kanye': 'examples/Kanye.mp4',
                  'Laughing Guy': 'examples/Laughing Guy.mp4',
                  'Parks and Recreation': 'examples/Parks and Recreation.mp4'}

# Create streamlit sidebar with options for different tasks
with st.sidebar:
    page = option_menu(menu_title='Menu',
                       menu_icon="robot",
                       options=["Welcome!",
                                "Object Detection",
                                "Facial Emotion Recognition",
                                "Hand Gesture Classification",
                                "Optical Character Recognition",
                                "Image Classification"],
                       icons=["house-door",
                              "search",
                              "emoji-smile",
                              "hand-thumbs-up",
                              "eyeglasses",
                              "check-circle"],
                       default_index=0,
                       )

    # Make sidebar slightly larger to accommodate larger names
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 375px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.title('Open-source Computer Vision')

# Load and display local gif file
file_ = open("resources/camera-robot-eye.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

# Page Definitions
if page == "Welcome!":
    st.header('Welcome!')
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )

    st.subheader('Quickstart')
    st.write(
        """
        Flip through the pages in the menu on the left hand side bar to perform CV tasks on-demand!
        
        Run computer vision tasks on:
        
            * Images
                * Examples
                * Upload your own
            * Video
                * Webcam
                * Examples
                * Upload your own
        """
    )

    st.subheader("Introduction")
    st.write("""
        Hello! This application is a continuation of my last space that you can access here: [Open-source NLP tool](https://huggingface.co/spaces/miesnerjacob/Multi-task-NLP). 
        Utilizing a range of open-source Python libraries programmers have the ability to stand up Machine Learning applications 
        easier than ever before. Pre-trained Machine Learning models' mass availability enabled Data Scientists to 
        avoid focusing on building complex Neural Networks from scratch for every project. Now Data Scientists can put more
        attention into solving domain specific problems by fine-tuning these model and applying them to business use-cases. 
        This application shows how easy it can be to implement Computer Vision on-demand within your application.   

        Utilizing this tool you will be able to perform a few Computer Vision Tasks. All you need to do is select your task, select or upload your input, and hit the start button!
        
        * This application has the ability to take both Images and Videos as input 

        * This application currently supports:
            * Image Object Detection
            * Video Object Detection
            * Image Classification
            * Hand Gesture Classification
            * Image Captioning

        More features may be added in the future including additional Computer Vision tasks and demos, depending on community feedback. 
        Please reach out to me at miesner.jacob@gmail.com or at my Linkedin page listed below if you have ideas or suggestions for improvement.

        If you would like to contribute yourself, feel free to fork the Github repository listed below and submit a merge request.
        """
             )
    st.subheader("Notes")
    st.write(
        """
        * If you are interested viewing the source code for this project or any of my other works you can view them here:
        
           [Project Github](https://github.com/MiesnerJacob/computer-vision-streamlit)

           [Jacob Miesner's Github](https://github.com/MiesnerJacob)

           [Jacob Miesner's Linkedin](https://www.linkedin.com/in/jacob-miesner-885050125/)

           [Jacob Miesner's Website](https://www.jacobmiesner.com)  
        """
    )

if page == "Object Detection":

    st.header('Object Detection')
    st.markdown("![Alt Text](https://media.giphy.com/media/vAvWgk3NCFXTa/giphy.gif)")
    st.write("This object detection app uses a pretrained YOLOv5 model which was trained to recognize the labels contained within the COCO dataset. More info [here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) on the classes this app can detect.")

    data_type = st.radio(
        "Select Data Type",
        ('Webcam', 'Video', 'Image'))

    if data_type == 'Webcam':
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_object_detection.callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    elif data_type == 'Video':
        input_type = st.radio(
            "Use example or upload your own?",
            ('Example', 'Upload'))

        if input_type == 'Example':
            option = st.selectbox(
                'Which example would you like to use?',
                (['Traffic',
                  'Elephant',
                  'Airport']))
            uploaded_file = video_examples[option]
            vid = uploaded_file
        else:
            uploaded_file = st.file_uploader("Choose a file", type=['mp4'])

        if st.button('🔥 Run!'):
            if st.button('STOP'):
                pass
            if uploaded_file is None:
                st.error("No file uploaded yet.")
            else:
                if uploaded_file and input_type == 'Upload':
                    vid = uploaded_file.name
                    with open(vid, mode='wb') as f:
                        f.write(uploaded_file.read())

                with st.spinner("Creating video frames..."):
                    frames, fps = video_object_detection.create_video_frames(vid)

                with st.spinner("Running object detection..."):
                    st.subheader("Object Detection Predictions")
                    video_object_detection.static_vid_obj(frames, fps)
                    if input_type == 'Upload':
                        # Delete uploaded video after annotation is complete
                        if vid:
                            os.remove(vid)

                video_file=open('outputs/annotated_video.mp4', 'rb')
                video_bytes = video_file.read()
                st.download_button(
                    label="Download annotated video",
                    data=video_bytes,
                    file_name='annotated_video.mp4',
                    mime='video/mp4'
                )
    elif data_type == 'Image':
        input_type = st.radio(
            "Use example or upload your own?",
            ('Example', 'Upload'))

        if input_type == 'Example':
            option = st.selectbox(
                'Which example would you like to use?',
                ('Home Office', 'Traffic', 'Barbeque'))
            uploaded_file = image_examples[option]
        else:
            uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

        if st.button('🔥 Run!'):
            if uploaded_file is None:
                st.error("No file uploaded yet.")
            else:
                with st.spinner("Running object detection..."):
                    img = Image.open(uploaded_file)
                    labeled_image, detections = image_object_detection.classify(img)

                if labeled_image and detections:
                    # Create image buffer and download
                    buf = BytesIO()
                    labeled_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()

                    st.subheader("Object Detection Predictions")
                    st.image(labeled_image)
                    st.download_button('Download Image', data=byte_im,file_name="image_object_detection.png", mime="image/jpeg")

                    # Create json and download button
                    st.json(detections)
                    st.download_button('Download Predictions', json.dumps(detections), file_name='image_object_detection.json')

elif page == 'Facial Emotion Recognition':

    st.header('Facial Emotion Recognition')
    st.markdown("![Alt Text](https://media.giphy.com/media/bnhtSlVeo7BxC/giphy.gif)")
    st.write('This app can classify seven different emotions including: Neutral, Happiness, Surprise, Sadness, Anger, Disgust, and Fear. Try it out!')

    data_type = st.radio(
        "Select Data Type",
        ('Webcam', 'Video', 'Image'))

    if data_type == 'Webcam':

        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        webrtc_ctx = webrtc_streamer(
            key="facial-emotion-recognition",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=facial_emotion_classifier.callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    elif data_type == 'Video':
        input_type = st.radio(
            "Use example or upload your own?",
            ('Example', 'Upload'))

        if input_type == 'Example':
            option = st.selectbox(
                'Which example would you like to use?',
                (['Kanye',
                  'Laughing Guy',
                  'Parks and Recreation']))
            uploaded_file = video_examples[option]
            vid = uploaded_file
        else:
            uploaded_file = st.file_uploader("Choose a file", type=['mp4'])

        if st.button('🔥 Run!'):
            if st.button('STOP'):
                pass
            if uploaded_file is None:
                st.error("No file uploaded yet.")
            else:
                if uploaded_file and input_type == 'Upload':
                    vid = uploaded_file.name
                    with open(vid, mode='wb') as f:
                        f.write(uploaded_file.read())

                with st.spinner("Creating video frames..."):
                    frames, fps = facial_emotion_classifier.create_video_frames(vid)

                with st.spinner("Running emotion recognition..."):
                    st.subheader("Emotion Recognition Predictions")
                    facial_emotion_classifier.static_vid_fer(frames, fps)
                    if input_type == 'Upload':
                        # Delete uploaded video after annotation is complete
                        if vid:
                            os.remove(vid)

                video_file=open('outputs/annotated_video.mp4', 'rb')
                video_bytes = video_file.read()
                st.download_button(
                    label="Download annotated video",
                    data=video_bytes,
                    file_name='annotated_video.mp4',
                    mime='video/mp4'
                )
    elif data_type == 'Image':
        input_type = st.radio(
            "Use example or upload your own?",
            ('Example', 'Upload'))

        if input_type == 'Example':
            option = st.selectbox(
                'Which example would you like to use?',
                ('Kanye', 'Shocked', 'Yelling'))
            uploaded_file = image_examples[option]
        else:
            uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

        if st.button('🔥 Run!'):
            if uploaded_file is None:
                st.error("No file uploaded yet.")
            else:
                with st.spinner("Running emotion recognition..."):
                    img = cv2.imread(uploaded_file)

                    labeled_image, detections = facial_emotion_classifier.prediction_label(img)
                    labeled_image = labeled_image[..., ::-1]
                    labeled_image = Image.fromarray(np.uint8(labeled_image))

                if labeled_image is not None and detections is not None:
                    # Create image buffer and download
                    buf = BytesIO()
                    labeled_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()

                    st.subheader("Emotion Recognition Predictions")
                    st.image(labeled_image)
                    st.download_button('Download Image', data=byte_im,file_name="image_emotion_recognition.png", mime="image/jpeg")

                    # Create json and download button
                    st.json(detections)
                    st.download_button('Download Predictions', json.dumps(str(detections)), file_name='image_emotion_recognition.json')
                else:
                    st.image(img)
                    st.warning('No faces recognized in this image...')

elif page == 'Hand Gesture Classification':

    st.header('Hand Gesture Classification')
    st.markdown("![Alt Text](https://media.giphy.com/media/tIeCLkB8geYtW/giphy.gif)")
    st.write('This app can classify ten different hand gestures including: Okay, Peace, Thumbs Up, Thumbs Down, Hang Loose, Stop, Rock On, Star Trek, Fist, Smile Sign. Try it out!')
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    webrtc_ctx = webrtc_streamer(
        key="hand-gesture-classification",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=hand_gesture_classifier.callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif page == 'Optical Character Recognition':

    st.header('Image Optical Character Recognition')
    st.markdown("![Alt Text](https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif)")

    input_type = st.radio(
        "Use example or upload your own?",
        ('Example', 'Upload'))

    if input_type == 'Example':
        option = st.selectbox(
            'Which example would you like to use?',
            ('Quick Brown Dog', 'Receipt', 'Street Sign'))
        uploaded_file = image_examples[option]
    else:
        uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

    if st.button('🔥 Run!'):
        with st.spinner("Running optical character recognition..."):
            annotated_image, text = image_optical_character_recognition.image_ocr(uploaded_file)

        # Create image buffer and download
        buf = BytesIO()
        annotated_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.subheader("Captioning Prediction")
        st.image(annotated_image)
        if text == '':
            st.wite("No text in this image...")
        else:
            st.write(text)

            st.download_button('Download Text', data=text, file_name='outputs/ocr_pred.txt')

elif page == 'Image Classification':

    st.header('Image Classification')
    st.markdown("![Alt Text](https://media.giphy.com/media/Zvgb12U8GNjvq/giphy.gif)")

    input_type = st.radio(
        "Use example or upload your own?",
        ('Example', 'Upload'))

    if input_type == 'Example':
        option = st.selectbox(
            'Which example would you like to use?',
            ('Car', 'Dog', 'Tropics'))
        uploaded_file = image_examples[option]
    else:
        uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

    if st.button('🔥 Run!'):
        if uploaded_file is None:
            st.error("No file uploaded yet.")
        else:
            with st.spinner("Running classification..."):
                img = Image.open(uploaded_file)
                preds = image_classifier.classify(img)

            st.write("")
            st.subheader("Classification Predictions")
            st.image(img)
            fig = px.bar(preds.sort_values("Pred_Prob", ascending=True), x='Pred_Prob', y='Class', orientation='h')
            st.write(fig)

            st.write("")
            csv = preds.to_csv(index=False).encode('utf-8')
            st.download_button('Download Predictions',csv,
                               file_name='classification_predictions.csv')