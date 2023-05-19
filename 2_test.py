"""Video transforms with OpenCV"""

import av
import cv2
import numpy as np
import streamlit as st
import face_recognition
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import logging
import os
import imghdr
from twilio.rest import Client
from typing import NamedTuple

logger = logging.getLogger(__name__)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data  # type: ignore
def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
    We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
    but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
    See https://github.com/whitphx/streamlit-webrtc/issues/1213
    """

    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]
    client = Client(account_sid, auth_token)
    token = client.tokens.create()
    return token.ice_servers


# Функция для загрузки изображений из папки, распознавания и извлечения кодировок лиц
def load_images_from_folder(folder):
    images = {}
    for person_name in os.listdir(folder):
        person_folder = os.path.join(folder, person_name)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                if imghdr.what(image_path) is not None:  # Проверка, что файл является изображением
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if len(encodings) > 0:  # Проверка, что найдено хотя бы одно лицо на изображении
                        images[person_name] = encodings[0]
                        print(len(images))
                        # break
                    else:
                        print(f"No faces found in {image_path}. Skipping.")
    return images


class Detection(NamedTuple):
    # class_id: int
    name: str
    # score: float
    box: np.ndarray


# Session-specific caching
cache_key = "face_detection"
if 'encodings' not in st.session_state:
    known_encodings = load_images_from_folder(ROOT_DIR + "/content/face/")
    st.session_state['encodings'] = known_encodings
else:
    known_encodings = st.session_state['encodings']


def callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    # scale_percent = 50  # Процент от исходного размера
    # width = int(image.shape[1] * scale_percent / 100)
    # height = int(image.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # detections = [
    #     Detection(
    #         # name=detection[1],
    #         box=detection[0]
    #     )
    #     for detection in face_recognition.face_locations(frame)
    #     # identify_faces(known_encodings, image, tolerance=0.5, min_face_area=5000)
    # ]

    # Render bounding boxes and captions
    for (top, right, bottom, left) in face_recognition.face_locations(image):
    # for detection in detections:
    #     (top, right, bottom, left) = detection.box
        # Рисуем прямоугольник вокруг лица
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        # Добавляем текст с именем над прямоугольником (detection.name)
        cv2.putText(image, '{}'.format(len(known_encodings)), (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(
    key="opencv-filter",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

