import numpy as np
import cv2
import torch
import pickle, os

from math import ceil
from itertools import product as product  # 笛卡尔积
from PIL import Image, ImageDraw, ImageFont

FACE_DB = "face.db"


class Anchors(object):
    def __init__(self, image_size=None):
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = False
        # ---------------------------#
        #   图片的尺寸
        # ---------------------------#
        self.image_size = image_size
        # ---------------------------#
        #   三个有效特征层高和宽
        # ---------------------------#
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]

    def get_anchors(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            # -----------------------------------------#
            #   对特征层的高和宽进行循环迭代
            # -----------------------------------------#
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip: output.clamp_(max=1, min=0)
        return output


def get_anchor():
    return Anchors(image_size=(640, 640)).get_anchors()


def letterbox_image(image, size):
    # 将图片resize->size 使用128填充
    ih, iw, _ = np.shape(image)
    w, h = size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    image = cv2.resize(image, (nw, nh))
    new_image = np.ones([size[1], size[0], 3]) * 128
    new_image[(h - nh) // 2:nh + (h - nh) // 2, (w - nw) // 2:nw + (w - nw) // 2] = image
    return new_image


def preprocess_input(image):
    image -= np.array((104, 117, 123), np.float32)
    return image


# -----------------------------------------------------------------#
#   将输出调整为相对于原图的大小
# -----------------------------------------------------------------#
def retinaface_correct_boxes(result, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    scale_for_boxs = [scale[1], scale[0], scale[1], scale[0]]
    scale_for_landmarks = [scale[1], scale[0]] * 5

    offset_for_boxs = [offset[1], offset[0], offset[1], offset[0]]
    offset_for_landmarks = [offset[1], offset[0]] * 5

    result[:, :4] = (result[:, :4] - np.array(offset_for_boxs)) * np.array(scale_for_boxs)
    result[:, 5:] = (result[:, 5:] - np.array(offset_for_landmarks)) * np.array(scale_for_landmarks)

    return result


def load_face():
    if os.path.exists(FACE_DB):
        with open(FACE_DB, "rb") as db:
            face_db = pickle.load(db)
    else:
        face_db = {}
    return face_db


def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    # (n, )
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


# ---------------------------------#
#   比较人脸
# ---------------------------------#
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=1):
    dis = face_distance(known_face_encodings, face_encoding_to_check)
    return list(dis <= tolerance), dis


# --------------------------------------#
#   写中文需要转成PIL来写。
# --------------------------------------#
def cv2ImgAddText(img, label, left, top, textColor=(255, 255, 255)):
    img = Image.fromarray(np.uint8(img))
    # ---------------#
    #   设置字体
    # ---------------#
    font = ImageFont.truetype(font='model_data/simhei.ttf', size=20)

    draw = ImageDraw.Draw(img)
    label = label.encode('utf-8')
    draw.text((left, top), str(label, 'UTF-8'), fill=textColor, font=font)
    return np.asarray(img)
