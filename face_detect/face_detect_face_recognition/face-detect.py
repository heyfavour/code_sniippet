"""
face_recognition
pip install cmake
dlib
https://stackoverflow.com/questions/73340815/pip-installs-failing-because-cmake-not-found

帧率低效果差
"""
import datetime
import os
import pickle
import time

import cv2
import face_recognition
import numpy as np

FACE_DB = "face.db"


def get_face_db():
    if os.path.exists(FACE_DB):
        with open(FACE_DB, "rb") as db:
            face_db = pickle.load(db)
    else:
        face_db = {}
    return list(face_db.values())


def run():
    video_capture = cv2.VideoCapture(0)
    known_face_encodings = get_face_db()
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # 缩放
            rgb_small_frame = small_frame[:, :, ::-1]  # BRG->RGB
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = ["Slave"] * len(face_encodings)
            for index, face_encoding in enumerate(face_encodings):
                for kown_face in known_face_encodings:
                    diff = np.linalg.norm(face_encoding - kown_face)
                    if diff < 0.8: face_names[index] = "Master"
            last_time = datetime.datetime.now()
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        last_time = datetime.datetime.now()
        cv2.imshow('Video', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
