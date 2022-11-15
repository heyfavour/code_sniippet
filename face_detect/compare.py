import datetime
import cv2
import face_recognition
import torch
import mediapipe as mp

from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import mediapipe as mp


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()


# face_locations                        0:00:00.040730
# mtcnn                                 0:00:00.133421
# mediapipe                             0:00:00.018108
# face_locations+ face_recognition      0:00:00.387637
# mtcnn+facenet                         0:00:00.180678
def face_recognition_test():
    start = datetime.datetime.now()
    file = "./face-pic/front.jpg"
    for i in range(100):
        image = cv2.imread(file)
        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)  # 缩放
        rgb_small_frame = small_frame[:, :, ::-1]  # BRG->RGB
        face_locations = face_recognition.face_locations(rgb_small_frame)  # (top, right, bottom, left)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    print((datetime.datetime.now() - start) / 100)


def facenet_test():
    start = datetime.datetime.now()
    file = "./face-pic/front.jpg"
    for i in range(100):
        img = Image.open(file)
        img_cropped = mtcnn(img)
        face_encodings = resnet(img_cropped.unsqueeze(0))
    print((datetime.datetime.now() - start) / 100)


def mediapipe_test():
    start = datetime.datetime.now()
    file = "./face-pic/front.jpg"
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        for i in range(1):
            image = cv2.imread(file)
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.detections:continue
    print((datetime.datetime.now() - start) / 100)


def compare_face_location():
    # face_recognition_test()
    # facenet_test()
    mediapipe_test()


def muliti_detect():
    pass


if __name__ == '__main__':
    compare_face_location()
