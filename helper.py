from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube
import pandas as pd
from shapely.geometry import Polygon
from shapely.geometry.point import Point
import numpy as np
import settings
# Install the ultralytics package
from ultralytics.solutions import object_counter



def get_pandas(results):
  # translate boxes data from a Tensor to the List of boxes info lists
  boxes_list = results[0].boxes.data.tolist()
  columns = ['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class_id']

  # iterate through the list of boxes info and make some formatting
  for i in boxes_list:
    # round float xyxy coordinates:
    i[:4] = [round(i, 1) for i in i[:4]]
    # translate float class_id to an integer
    i[5] = int(i[5])
    # add a class name as a last element
    i.append(results[0].names[i[5]])

  # create the result dataframe
  columns.append('class_name')
  result_df = pd.DataFrame(boxes_list, columns=columns)

  return result_df



def get_pandas1(results):
  # translate boxes data from a Tensor to the List of boxes info lists
  boxes_list = results[0].boxes.data.tolist()
  columns = ['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class_id','Uid']

  # iterate through the list of boxes info and make some formatting
  for i in boxes_list:
    # round float xyxy coordinates:
    i[:4] = [round(i, 1) for i in i[:4]]
    # translate float class_id to an integer
    i[5] = int(i[5])
    # add a class name as a last element
    i.append(results[0].names[i[5]])

  # create the result dataframe
  columns.append('class_name')
  result_df = pd.DataFrame(boxes_list, columns=columns)

  return result_df


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """
    st.markdown('<style>div[data-testid="stImageCaption"] { color: #080808; font-size: 20px; }</style>', unsafe_allow_html=True)


    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    #region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

    # Initialize Object Counter
    #counter = object_counter.ObjectCounter()
    #counter.set_args(view_img=True, reg_pts=region_points, classes_names=model.names, draw_tracks=True)

    

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker, )
        res_plotted = res[0].plot()
        st_frame.image(res_plotted,
                    caption='Detected Video',
                    channels="BGR",
                    use_column_width=True
                    )
        #image = counter.start_counting(image, res)
        #df=get_pandas1(res)
        #st.write(df)
        #cv2.imshow("real time object",image) 
      

        

       
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

        res_plotted = res[0].plot()
        st_frame.image(res_plotted,
                    caption='Detected Video',
                    channels="BGR",
                    use_column_width=True
                    )
        df=get_pandas(res)
        st.write(df)
        

    # # Plot the detected objects on the video frame
    
    #dfs = df['class_name'].value_counts().reset_index()
    #dfs.columns = ['class_name', 'count']
    #st.write(dfs)
    #st.table(dfs.groupby('class_name').count())
    '''count = df["class_name"].value_counts()
    combined_df = pd.concat(df, ignore_index=True)
    st.write(df)'''


def play_youtube_video(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_youtube = st.sidebar.text_input("YouTube Video url")

    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                res = model.track(image, conf=conf, persist=True, tracker=tracker)
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    # vid_cap = cv2.VideoCapture(source_rtsp)
                    # time.sleep(0.1)
                    # continue
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
