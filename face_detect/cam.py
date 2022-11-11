import datetime
import os
import pickle

FACE_DB = "face.db"


def load_face():
    if os.path.exists(FACE_DB):
        with open(FACE_DB, "rb") as db:
            face_db = pickle.load(db)
    else:
        face_db = {}
    print(face_db)


def add_face():
    if os.path.exists(FACE_DB):
        with open(FACE_DB, "rb") as db:
            face_db = pickle.load(db)
    else:
        face_db = {}
    face_db["master"] = datetime.datetime.now()
    with open(FACE_DB, "wb") as db:
        pickle.dump(face_db, db)


if __name__ == '__main__':
    load_encoding()
    add_encoding()
    load_encoding()
