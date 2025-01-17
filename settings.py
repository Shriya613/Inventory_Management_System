from pathlib import Path
import sys


# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
DATA = 'Data'
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'

SOURCES_LIST = [DATA,IMAGE, VIDEO, WEBCAM, RTSP, YOUTUBE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'office_4.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'video_1.mp4'
VIDEO_2_PATH = VIDEO_DIR / 'video_2.mp4'
VIDEO_3_PATH = VIDEO_DIR / 'video_3.mp4'
VIDEO1_PATH = VIDEO_DIR / 'video1.mp4'
VIDEO2_PATH = VIDEO_DIR / 'video2.mp4'
VIDEO3_PATH = VIDEO_DIR / 'video3.mp4'
VIDEO4_PATH = VIDEO_DIR / 'video4.mp4'
VIDEO5_PATH = VIDEO_DIR / 'video5.mp4'

VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH,
    'video_2': VIDEO_2_PATH,
    'video_3': VIDEO_3_PATH,
    'video1': VIDEO1_PATH,
    'video2': VIDEO2_PATH,
    'video3': VIDEO3_PATH,
    'video4': VIDEO4_PATH,
    'video5': VIDEO5_PATH
}


# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL1 = MODEL_DIR / 'noodlesV8.pt'
DETECTION_MODEL2 = MODEL_DIR / 'best8.pt'
DETECTION_MODEL3 = MODEL_DIR / 'laysV8.pt'
DETECTION_MODEL4 = MODEL_DIR / 'cooldrinksV8.pt'
DETECTION_MODEL5 = MODEL_DIR / 'biscuitsv8.pt'
# In case of your custome model comment out the line above and
# Place your custom model pt file name at the line below 
# DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'


#RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
# Webcam
WEBCAM_PATH = 0
show = True
'''webrtc_streamer(
                key="webcam",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                
                async_processing=True,
        )'''
