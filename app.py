import streamlit as st
from streamlit_option_menu import option_menu
from image_object_detection import ImageObjectDetection
from video_object_detection import VideoObjectDetection
from image_classification import ImageClassification
import plotly.express as px
from PIL import Image
from io import BytesIO
import base64
import json

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# @st.cache(allow_output_mutation=True)
# def load_image_object_detection():
#     return ImageObjectDetection()
#
#
# @st.cache(allow_output_mutation=True)
# def load_video_object_detection():
#     return VideoObjectDetection()


@st.cache(allow_output_mutation=True)
def load_image_classifier():
    return ImageClassification()


# image_object_detection = load_image_object_detection()
# video_object_detection = load_video_object_detection()
image_classifier = load_image_classifier()

image_examples = {'Traffic': 'examples/traffic.jpeg',
                  'BBQ': 'examples/bbq.jpeg',
                  'WFH': 'examples/wfh.jpeg',
                  'Car': 'examples/car.jpeg',
                  'Dog': 'examples/dog.jpeg',
                  'Tropics': 'examples/tropics.jpeg'}
video_examples = {'Traffic': 'examples/traffic.mp4'}

with st.sidebar:
    page = option_menu(menu_title='Menu',
                       menu_icon="robot",
                       options=["Welcome!",
                                "Object Detection",
                                "Classification",
                                "Segmentation"],
                       icons=["house-door",
                              "search",
                              "check-circle",
                              "cone-striped"],
                       default_index=0,
                       )

st.title('Open-source Computer Vision')

# Make Radio button horizontal
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# Load and display local gif file
file_ = open("camera-robot-eye.gif", "rb")
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
        Select pick example or upload file below and flip through the pages in the menu to perform CV tasks on-demand!
        """
    )

    data_type = st.radio(
        "Input Data Type",
        ('Image', 'Video'))

    input_type = st.radio(
        "Use example or upload your own?",
        ('Example', 'Upload'))

    if data_type == 'Video':
        if input_type == 'Example':
            option = st.selectbox(
                'Which example would you like to use?',
                (['Traffic']))
            uploaded_file = image_examples[option]
            vid = uploaded_file
        else:
            uploaded_file = st.file_uploader("Choose a file", type=['mp4'])
    else:
        if input_type == 'Example':
            option = st.selectbox(
                'Which example would you like to use?',
                ('Traffic', 'BBQ', 'WFH'))
            uploaded_file = image_examples[option]
        else:
            uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

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
            * Segmentation

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
        "Input Data Type",
        ('Image', 'Video'))

    input_type = st.radio(
        "Use example or upload your own?",
        ('Example', 'Upload'))

    if data_type == 'Video':
        if input_type == 'Example':
            option = st.selectbox(
                'Which example would you like to use?',
                (['Traffic']))
            uploaded_file = video_examples[option]
            vid = uploaded_file
        else:
            uploaded_file = st.file_uploader("Choose a file", type=['mp4'])
        skip_frames = st.slider("Detect in every Nth frame", value=10, min_value=1, max_value=20, step=1)

        if uploaded_file and input_type == 'Upload':
            vid = uploaded_file.name
            with open(vid, mode='wb') as f:
                f.write(uploaded_file.read())

        if st.button('🔥 Run!'):
            if uploaded_file is None:
                st.error("No file uploaded yet.")
            else:
                st.spinner("Creating video frames...")
                frames = video_object_detection.create_video_frames(vid, skip_frames)

                st.spinner("Running object detection...")
                video_object_detection.object_detection(frames, skip_frames)

    else:
        if input_type == 'Example':
            option = st.selectbox(
                'Which example would you like to use?',
                ('Traffic', 'BBQ', 'WFH'))
            uploaded_file = image_examples[option]
        else:
            uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

        if st.button('🔥 Run!'):
            if uploaded_file is None:
                st.error("No file uploaded yet.")
            else:
                st.spinner("Running object detection...")
                img = Image.open(uploaded_file)
                labeled_image, detections = image_object_detection.classify(img)

                if labeled_image and detections:
                    # Create image buffer and download
                    buf = BytesIO()
                    img.save(buf, format="JPEG")
                    byte_im = buf.getvalue()
                    st.download_button('Download Image', data=byte_im,file_name="image_object_detection.png", mime="image/jpeg")

                    # Create json and download button
                    st.json(detections)
                    st.download_button('Download Predictions', json.dumps(detections), file_name='image_object_detection.json')


elif page == 'Classification':
    st.header('Classification')
    st.markdown("![Alt Text](https://media.giphy.com/media/Zvgb12U8GNjvq/giphy.gif)")

    data_type = st.radio(
        "Input Data Type",
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
            st.spinner("Running object detection...")
            img = Image.open(uploaded_file)
            preds = image_classifier.classify(img)

            st.write("")
            st.subheader("Classification Predictions")
            fig = px.bar(preds.sort_values("Pred_Prob", ascending=True), x='Pred_Prob', y='Class', orientation='h')
            st.write(fig)

            st.write("")
            csv = preds.to_csv(index=False).encode('utf-8')
            st.download_button('Download Predictions',csv,
                               file_name='classification_predictions.csv')

elif page == 'Segmentation':
    st.header('Segmentation')
    st.markdown("![Alt Text](https://media.giphy.com/media/urvsFBDfR6N32/giphy.gif)")

    data_type = st.radio(
        "Input Data Type",
        ('Image', 'Video'))

    if data_type == 'Video':
        uploaded_file = st.file_uploader("Choose a file", type=['mp4'])
        skip_frames = st.slider("Detect in every Nth frame", value=10, min_value=1, max_value=20, step=1)
    else:
        uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

    if st.button('🔥 Run!'):
        pass