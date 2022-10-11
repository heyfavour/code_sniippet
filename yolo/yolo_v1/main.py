import os
import random
import torch
import numpy as np
from model import YOLOv1,YOLOv1ResNet
from PIL import Image,ImageDraw
from data import transform,objects_list

# random colors
COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]


def calculate_iou(box1, box2):  # x.y.w.h w.y.w.h
    box1, box2 = box1.cpu().detach().numpy().tolist(), box2.cpu().detach().numpy().tolist()

    area1, area2 = box1[2] * box1[3], box2[2] * box2[3]  # 面积 w*h

    max_left = max(box1[0] - box1[2] / 2, box2[0] - box2[2] / 2)  # box1_left,box2_left
    min_right = min(box1[0] + box1[2] / 2, box2[0] + box2[2] / 2)  # box1_right,box2_right
    max_top = max(box1[1] - box1[3] / 2, box2[1] - box2[3] / 2)  # box1_top,box2_top
    min_bottom = min(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2)  # box1_bottom,box2_bottom

    if min_right <= max_left or min_bottom <= max_top:
        return 0
    else:
        intersect = (min_right - max_left) * (min_bottom - max_top)
        return intersect / (area1 + area2 - intersect)

def afterprocessing(pred, conf_thresh, iou_thresh):
    boxs = torch.zeros((7 * 7, 5 + 20))  # 49*25
    #取较大的 c
    for x in range(7):
        for y in range(7):
            conf1, conf2 = pred[x, y, 4], pred[x, y, 9]
            if conf1 > conf2:
                # bbox1
                boxs[(x * 7 + y), 0:4] = torch.Tensor([pred[x, y, 0], pred[x, y, 1], pred[x, y, 2], pred[x, y, 3]])
                boxs[(x * 7 + y), 4] = pred[x, y, 4]
                boxs[(x * 7 + y), 5:] = pred[x, y, 10:]
            else:
                # bbox2
                boxs[(x * 7 + y), 0:4] = torch.Tensor([pred[x, y, 5], pred[x, y, 6], pred[x, y, 7], pred[x, y, 8]])
                boxs[(x * 7 + y), 4] = pred[x, y, 9]
                boxs[(x * 7 + y), 5:] = pred[x, y, 10:]
    #7*7*25 -> nms
    xywhcc = nms(boxs, conf_thresh, iou_thresh)
    return xywhcc


def nms(boxs, conf_thresh=0.1, iou_thresh=0.3):
    box_prob = boxs[:, 5:].clone().detach()  # 7*7 20 class
    box_conf = boxs[:, 4].clone().detach().unsqueeze(1).expand_as(box_prob)  # 7*7*20
    box_cls_spec_conf = box_prob * box_conf  # 7*7*20 # xywhc的c*class_prob
    box_cls_spec_conf[box_cls_spec_conf <= conf_thresh] = 0 #阈值过滤

    # for each class, sort the cls-spec-conf score
    for c in range(20):
        rank = torch.sort(box_cls_spec_conf[:, c], descending=True).indices  #降序
        # for each bbox
        for i in range(boxs.shape[0]):#98
            if box_cls_spec_conf[rank[i], c] == 0:continue
            for j in range(i + 1, boxs.shape[0]):#每一个都和最高的比较
                if box_cls_spec_conf[rank[j], c] != 0:
                    # iou > 交并比  说明识别同一个物体
                    iou = calculate_iou(boxs[rank[i], 0:4], boxs[rank[j], 0:4])
                    if iou > iou_thresh:box_cls_spec_conf[rank[j], c] = 0

    # exclude cls-specific confidence score=0
    boxs = boxs[torch.max(box_cls_spec_conf, dim=1).values > 0]#取NMS以后任有值的
    box_cls_spec_conf = box_cls_spec_conf[torch.max(box_cls_spec_conf, dim=1).values > 0]
    ret = torch.ones((boxs.size()[0], 6))
    if boxs.size()[0] == 0:return torch.tensor([])
    ret[:, 0:4] = boxs[:, 0:4]#x y w h
    ret[:, 4] = torch.max(box_cls_spec_conf, dim=1).values# c
    ret[:, 5] = torch.argmax(boxs[:, 5:], dim=1).int()#class
    return ret

def draw_rectangle(file, boxs):
    img = Image.open(file)
    w,h = img.size

    im = ImageDraw.ImageDraw(img)
    n = boxs.size()[0]#re
    boxs = boxs.detach().numpy()

    for i in range(n):
        #box[i] xywhcc
        p1 = (int((boxs[i, 0] - boxs[i, 2] / 2) * w), int((boxs[i, 1] - boxs[i, 3] / 2) * h))#x,y
        p2 = (int((boxs[i, 0] + boxs[i, 2] / 2) * w), int((boxs[i, 1] + boxs[i, 3] / 2) * h))#x,y
        class_name = objects_list[int(boxs[i, 5])]
        confidence = int(boxs[i, 4].item()*5)
        im.rectangle((p1,p2), fill=None, outline="red", width=confidence)
        x,y = p1
        im.text((x+confidence,y+confidence),class_name,fill="red",fount=8)
        img.show()


def predict(img, model,conf_thresh, iou_thresh):
    pred = model(img)[0].detach().cpu()#7*7*30
    xywhcc = afterprocessing(pred,conf_thresh, iou_thresh)#后处理nms
    return xywhcc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv1(7, 2, 20).to(device)
    weight_path = "output"
    model.load_state_dict(torch.load(weight_path))

    file = ""
    img = np.array(Image.open(f"../data/JPEGImages/{file}"))  # [96 96 3]
    img = transform(img)
    img.unsqueeze_(0)

    xywhcc = predict(img, model, 0.1, 0.3)#pic model xywh的c iou

    if xywhcc.size()[0] != 0:draw_rectangle(file, xywhcc)

