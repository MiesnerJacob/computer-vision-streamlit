import streamlit as st
from streamlit_option_menu import option_menu
from video_object_detection import VideoObjectDetection
from image_object_detection import ImageObjectDetection
from image_classification import ImageClassification
from image_segmentation import ImageSegmentation
from image_captioning import ImageCaptioning
from image_question_answering import ImageQuestionAnswering
import plotly.express as px
from PIL import Image
from io import BytesIO
import base64
import json
from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    WebRtcStreamerContext,
    webrtc_streamer,
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_video_object_detection():
    return VideoObjectDetection()

@st.cache(allow_output_mutation=True)
def load_image_object_detection():
    return ImageObjectDetection()

@st.cache(allow_output_mutation=True)
def load_image_classifier():
    return ImageClassification()

# @st.cache(allow_output_mutation=True)
# def load_image_segmentation():
#     return ImageSegmentation()

@st.cache(allow_output_mutation=True)
def load_image_captioning():
    return ImageCaptioning()


video_object_detection = load_video_object_detection()
image_object_detection = load_image_object_detection()
image_classifier = load_image_classifier()
# image_segmentation = load_image_segmentation()
image_captioning = load_image_captioning()

image_examples = {'Traffic': 'examples/Traffic.jpeg',
                  'Barbeque': 'examples/Barbeque.jpeg',
                  'Home Office': 'examples/Home Office.jpeg',
                  'Car': 'examples/Car.jpeg',
                  'Dog': 'examples/Dog.jpeg',
                  'Tropics': 'examples/Tropics.jpeg'}
video_examples = {'Traffic': 'examples/Traffic.mp4'}

with st.sidebar:
    page = option_menu(menu_title='Menu',
                       menu_icon="robot",
                       options=["Welcome!",
                                "Object Detection",
                                "Classification",
                                "Semantic Segmentation",
                                "Captioning"],
                       icons=["house-door",
                              "search",
                              "check-circle",
                              "cone-striped",
                              "body-text"],
                       default_index=0,
                       )

st.title('Open-source Computer Vision')

# Make Radio button horizontal
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# Load and display local gif file
file_ = open("resources/camera-robot-eye.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

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

        Utilizing this tool you will be able to perform a multitude of Computer Vision Tasks on a range of
        different tasks. All you need to do is select or upload your input, select your task, and hit the start button!
        
        * This application has the ability to take both Images and Videos as input 

        * This application currently supports:
            * Object Detection
            * Classification
            * Semantic Segmentation
            * Captioning
            * Question Answering

        More features may be added in the future including additional Computer Vision tasks, depending on community feedback. 
        Please reach out to me at miesner.jacob@gmail.com or at my Linkedin page listed below if you have ideas or suggestions for improvement.

        If you would like to contribute yourself, feel free to fork the Github repository listed below and submit a merge request.
        """
             )
    st.subheader("Notes")
    st.write(
        """
        * If you are interested viewing the source code for this project or any of my other works you can view them here:
        
           [Project Github](https://github.com/MiesnerJacob/Multi-task-NLP-dashboard)

           [Jacob Miesner's Github](https://github.com/MiesnerJacob)

           [Jacob Miesner's Linkedin](https://www.linkedin.com/in/jacob-miesner-885050125/)

           [Jacob Miesner's Website](https://www.jacobmiesner.com)  
        """
    )

if page == "Object Detection":
    st.header('Object Detection')
    st.markdown("![Alt Text](https://media.giphy.com/media/vAvWgk3NCFXTa/giphy.gif)")

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
                (['Traffic']))
            uploaded_file = video_examples[option]
            vid = uploaded_file
        else:
            uploaded_file = st.file_uploader("Choose a file", type=['mp4'])

        if uploaded_file and input_type == 'Upload':
            vid = uploaded_file.name
            with open(vid, mode='wb') as f:
                f.write(uploaded_file.read())

        if st.button('🔥 Run!'):
            if st.button('STOP'):
                pass
            if uploaded_file is None:
                st.error("No file uploaded yet.")
            else:
                with st.spinner("Creating video frames..."):
                    frames, fps = video_object_detection.create_video_frames(vid)

                with st.spinner("Running object detection..."):
                    st.subheader("Object Detection Predictions")
                    video_object_detection.static_vid_obj(frames, fps)

                video_file=open('outputs/annotated_video.mp4', 'rb')
                video_bytes = video_file.read()
                st.download_button(
                    label="Download annotated video",
                    data=video_bytes,
                    file_name='annotated_video.mp4',
                    mime='video/mp4',
                )

    else:
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


elif page == 'Classification':
    st.header('Classification')
    st.markdown("![Alt Text](https://media.giphy.com/media/Zvgb12U8GNjvq/giphy.gif)")

    data_type = st.radio(
        "Select Data Type",
        (['Image']))

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

elif page == 'Semantic Segmentation':
    st.header('Semantic Segmentation')
    st.markdown("![Alt Text](https://media.giphy.com/media/urvsFBDfR6N32/giphy.gif)")

    data_type = st.radio(
        "Select Data Type",
        (['Image']))

    input_type = st.radio(
        "Use example or upload your own?",
        ('Example', 'Upload'))

    if input_type == 'Example':
        option = st.selectbox(
            'Which example would you like to use?',
            ('Home Office', 'Traffic', 'Barbeque', 'Car', 'Dog', 'Tropics'))
        uploaded_file = image_examples[option]
    else:
        uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

    if st.button('🔥 Run!'):
        if uploaded_file is None:
            st.error("No file uploaded yet.")
        else:
            with st.spinner("Running segmentation..."):
                img = Image.open(uploaded_file)
                labeled_image, detections = image_segmentation.classify(img)

elif page == 'Captioning':
    st.header('Captioning')
    st.markdown("![Alt Text](https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif)")

    data_type = st.radio(
        "Select Data Type",
        (['Image']))

    input_type = st.radio(
        "Use example or upload your own?",
        ('Example', 'Upload'))

    if input_type == 'Example':
        option = st.selectbox(
            'Which example would you like to use?',
            ('Home Office', 'Traffic', 'Barbeque', 'Car', 'Dog', 'Tropics'))
        uploaded_file = image_examples[option]
    else:
        uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

    if st.button('🔥 Run!'):
        if uploaded_file is None:
            st.error("No file uploaded yet.")
        else:
            with st.spinner("Running caption generation..."):
                img = image_captioning.caption(uploaded_file)

                # Create image buffer and download
                buf = BytesIO()
                img.save(buf, format="PNG")
                byte_im = buf.getvalue()

                st.subheader("Captioning Prediction")
                st.image(img)
                st.download_button('Download Image', data=byte_im, file_name="image_object_detection.png",
                                   mime="image/jpeg")