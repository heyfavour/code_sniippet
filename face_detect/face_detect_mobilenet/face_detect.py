import torch
import cv2
import numpy as np
import utils
import time
import os, sys

sys.path.append(os.getcwd())

from retinaface import RetinaFace, decode, decode_landm, non_max_suppression
from facenet import Facenet, AlignmentFace
from utils import retinaface_correct_boxes, letterbox_image, get_anchor, load_face, compare_faces, cv2ImgAddText

net = RetinaFace().eval()
state_dict = torch.load("./model/Retinaface_mobilenet0.25.pth", map_location="cpu")
net.load_state_dict(state_dict)

facenet = Facenet().eval()
state_dict = torch.load("./model/facenet_mobilenet.pth", map_location="cpu")
facenet.load_state_dict(state_dict, strict=False)

known_face_encodings = np.array(list(load_face().values())[0])  # [n,128]


def detect_face(frame):
    ##########################################################预测头像
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BRG->RGB
    image = np.array(image, np.float32)
    im_height, im_width, _ = np.shape(image)  # 640 480
    # [640, 480, 640, 480] 宽 高 宽 高
    scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
    # [640, 480, 640, 480, 640, 480, 640, 480, 640, 480]
    scale_for_landmarks = [np.shape(image)[1], np.shape(image)[0]] * 5  # 5关键点
    image = letterbox_image(image, [640, 640])
    anchors = get_anchor()  # [16800, 4]
    # ---------------------------------------------------#
    #   将处理完的图片传入Retinaface网络当中进行预测
    # ---------------------------------------------------#
    with torch.no_grad():
        #   图片预处理，归一化。
        image = torch.from_numpy(utils.preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)
        loc, conf, landms = net(image)
        # ---------------------------------------------------#
        #   Retinaface网络的解码，最终我们会获得预测框
        #   将预测结果进行解码和非极大抑制
        # ---------------------------------------------------#
        boxes = decode(loc.data.squeeze(0), anchors, [0.1, 0.2])
        conf = conf.data.squeeze(0)[:, 1:2]
        landms = decode_landm(landms.data.squeeze(0), anchors, [0.1, 0.2])
        # -----------------------------------------------------------#
        #   对人脸检测结果进行堆叠
        # -----------------------------------------------------------#
        boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
        boxes_conf_landms = non_max_suppression(boxes_conf_landms, 0.5)
        # ---------------------------------------------------#
        #   如果没有预测框则返回原图
        # ---------------------------------------------------#
        if len(boxes_conf_landms) <= 0: return []
        boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, np.array([640, 640]),
                                                     np.array([im_height, im_width]))

        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks
    return boxes_conf_landms


def recognized_face(boxes_conf_landms, frame):
    face_encodings = []
    for boxes_conf_landm in boxes_conf_landms:
        ############################################图像截取，人脸矫正
        boxes_conf_landm = np.maximum(boxes_conf_landm, 0)
        crop_img = np.array(frame)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]),
                   int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
        landmark = np.reshape(boxes_conf_landm[5:], (5, 2)) - np.array(
            [int(boxes_conf_landm[0]), int(boxes_conf_landm[1])])
        crop_img, _ = AlignmentFace(crop_img, landmark)
        # ----------------------#
        #   人脸编码
        # ----------------------#
        crop_img = np.array(letterbox_image(np.uint8(crop_img), (160, 160))) / 255
        crop_img = np.expand_dims(crop_img.transpose(2, 0, 1), 0)
        with torch.no_grad():
            crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
            face_encoding = facenet(crop_img)[0].cpu().numpy()
            face_encodings.append(face_encoding)
    return face_encodings


def draw_picture(boxes_conf_landms, face_encodings, frame):
    if not face_encodings: return frame, False
    face_names = []
    recognize_gestures = False
    for face_encoding in face_encodings:
        # -----------------------------------------------------#
        #   取出一张脸并与数据库中所有的人脸进行对比，计算得分
        # -----------------------------------------------------#
        matches, face_distances = compare_faces(known_face_encodings, face_encoding, tolerance=0.9)
        # -----------------------------------------------------#
        #   取出这个最近人脸的评分
        #   取出当前输入进来的人脸，最接近的已知人脸的序号
        # -----------------------------------------------------#
        name = "Unkown"
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = "Master"
            recognize_gestures = True
        #    #name = known_face_names[best_match_index] 目前只有一个因此不需要
        # face_names.append(name)
    # -----------------------------------------------#
    #   人脸特征比对-结束
    # -----------------------------------------------#

    # for i, b in enumerate(boxes_conf_landms):
    #     text = "{:.4f}".format(b[4])
    #     b = list(map(int, b))
    #     # ---------------------------------------------------#
    #     #   b[0]-b[3]为人脸框的坐标，b[4]为得分
    #     # ---------------------------------------------------#
    #     cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    #     cx = b[0]
    #     cy = b[1] + 12
    #     cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    #
    #     # ---------------------------------------------------#
    #     #   b[5]-b[14]为人脸关键点的坐标
    #     # ---------------------------------------------------#
    #     # cv2.circle(frame, (b[5], b[6]), 1, (0, 0, 255), 4)
    #     # cv2.circle(frame, (b[7], b[8]), 1, (0, 255, 255), 4)
    #     # cv2.circle(frame, (b[9], b[10]), 1, (255, 0, 255), 4)
    #     # cv2.circle(frame, (b[11], b[12]), 1, (0, 255, 0), 4)
    #     # cv2.circle(frame, (b[13], b[14]), 1, (255, 0, 0), 4)
    #
    #     name = face_names[i]
    #     # font = cv2.FONT_HERSHEY_SIMPLEX
    #     # cv2.putText(old_image, name, (b[0] , b[3] - 15), font, 0.75, (255, 255, 255), 2)
    #     # --------------------------------------------------------------#
    #     #   cv2不能写中文，加上这段可以，但是检测速度会有一定的下降。
    #     #   如果不是必须，可以换成cv2只显示英文。
    #     # --------------------------------------------------------------#
    #     frame = cv2ImgAddText(frame, name, b[0] + 5, b[3] - 25)
    return frame, recognize_gestures


def run():
    cap = cv2.VideoCapture(0)
    fps = 0.0
    count = 0
    recognize_gestures = False

    while cap.isOpened():
        # start = time.time()
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not success: continue
        ##########################################################检测头像
        if count % 10 == 0:
            boxes_conf_landms = detect_face(frame)
            face_encodings = recognized_face(boxes_conf_landms, frame)
            frame, recognize_gestures = draw_picture(boxes_conf_landms, face_encodings, frame)
        #########################################################FPS
        # fps = (fps + (1. / (time.time() - start))) / 2
        # frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #########################################################RECONGINZE
        notice = recognize_gestures and f"Recognizing gestures" or f""
        frame = cv2.putText(frame, notice, (480, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(100) & 0xFF == 27: break
        count = count + 1
        if count % 10 == 0: count = 0
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
