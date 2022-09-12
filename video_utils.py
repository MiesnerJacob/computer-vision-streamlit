import cv2

def create_video_frames(video_path):
    """
    XXX.

    Parameters:
        xxx (type): ___
    Returns:
        xxx (type): ___
    """

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