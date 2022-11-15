import os
import pickle
import cv2
import numpy as np

from face_detect import detect_face,recognized_face

FACE_DB = "face.db"


def encode_face(file):
    image = cv2.imread(file)
    boxes_conf_landms = detect_face(image)
    face_encodings = recognized_face(boxes_conf_landms, image)
    return face_encodings[0]


def load_face():
    if os.path.exists(FACE_DB):
        with open(FACE_DB, "rb") as db:
            face_db = pickle.load(db)
    else:
        face_db = {}



def add_face():
    if os.path.exists(FACE_DB):
        with open(FACE_DB, "rb") as db:
            face_db = pickle.load(db)
    else:
        face_db = {}
    front_encoding = encode_face("../face-pic/front.jpg")
    right_encoding = encode_face("../face-pic/right.jpg")
    left_encoding = encode_face("../face-pic/left.jpg")
    face_db["master"] = np.array([front_encoding,right_encoding,left_encoding])
    with open(FACE_DB, "wb") as db:
        pickle.dump(face_db, db)


if __name__ == '__main__':
    load_face()
    add_face()
    load_face()
