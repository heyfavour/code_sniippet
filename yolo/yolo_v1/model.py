import torch
from torch import nn
from torchvision.models import resnet34


class YOLOv1(nn.Module):
    """
    YOLOv1 model structure
    yolo-v1 = conv + fc
    """

    def __init__(self, S=7, B=2, num_classes=20):
        """
        :param S:grid num 7
        :param B:box num 2
        :param num_classes:class num 20
        """
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.conv_layers = nn.Sequential(
            # 448*448*3 -> 112*112*192
            # conv.layer 7*7*64-s-2 maxpool layer2*2-s-2
            nn.Conv2d(3, 192, 7, stride=2, padding=3),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # 112*112*192 ->56*56*256
            nn.Conv2d(192, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # 56*56*256 -> 28*28*512
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # 28*28*512 -> 14*14*1024
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # 14*14*1024 -> 7*7*1024
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            # 7*7*1024 -> 7*7*1024
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # full connection part
        self.fc_layers = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.num_classes)),
            nn.Sigmoid()  # normalized to 0~1
        )

    def forward(self, x):
        out = self.conv_layers(x)  # b*1024*7*7
        out = out.view(out.size()[0], -1)  # b*50176
        out = self.fc_layers(out)
        out = out.reshape(-1, self.S, self.S, self.B * 5 + self.num_classes)  # 7*7*(5*2+20)=1470
        return out


class YOLOv1ResNet(nn.Module):
    """YOLOv1-Resnet model structure
    yolo-v1 resnet = resnet(backbone) + conv + fc
    """

    def __init__(self, S, B, num_classes):
        super(YOLOv1ResNet, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes
        # self.resnet = resnet18()
        self.resnet = resnet34()
        # print(self.resnet.fc.in_features)
        # print(*list(self.resnet.children())[-2:])  # show last two layers

        # backbone part, (cut resnet's last two layers)
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])

        # conv part
        self.conv_layers = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            # nn.Conv2d(1024, 1024, 3, padding=1),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(0.1, inplace=True),
        )

        # full connection part
        self.fc_layers = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.num_classes)),
            nn.Sigmoid()  # normalized to 0~1
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.conv_layers(out)
        out = out.view(out.size()[0], -1)
        out = self.fc_layers(out)
        out = out.reshape(-1, self.S, self.S, self.B * 5 + self.num_classes)
        return out


class YOLOv1Loss(nn.Module):
    def __init__(self, S, B):
        super().__init__()
        self.S = S
        self.B = B

    def calculate_iou(self, box1, box2):#x.y.w.h w.y.w.h
        box1, box2 = box1.cpu().detach().numpy().tolist(), box2.cpu().detach().numpy().tolist()

        area1,area2 = box1[2] * box1[3],box2[2] * box2[3]  # 面积 w*h

        max_left = max(box1[0] - box1[2] / 2, box2[0] - box2[2] / 2)#box1_left,box2_left
        min_right = min(box1[0] + box1[2] / 2, box2[0] + box2[2] / 2)#box1_right,box2_right
        max_top = max(box1[1] - box1[3] / 2, box2[1] - box2[3] / 2)#box1_top,box2_top
        min_bottom = min(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2)#box1_bottom,box2_bottom

        if min_right <= max_left or min_bottom <= max_top:
            return 0
        else:
            # iou = intersect / union
            intersect = (min_right - max_left) * (min_bottom - max_top)
            return intersect / (area1 + area2 - intersect)

    def forward(self, preds, labels):#label (i,y,x,30) xywhc
        batch_size = labels.size(0)

        loss_coord_xy = 0.  # coord xy loss
        loss_coord_wh = 0.  # coord wh loss
        loss_obj = 0.  # obj loss
        loss_no_obj = 0.  # no obj loss
        loss_class = 0.  # class loss

        for i in range(batch_size):#一个对象
            for y in range(self.S):
                for x in range(self.S):#
                    if labels[i, y, x, 4] == 1:#区域内有东西
                        box1 = torch.Tensor([preds[i, y, x, 0], preds[i, y, x, 1], preds[i, y, x, 2], preds[i, y, x, 3]])#x.y.w.h
                        box2 = torch.Tensor([preds[i, y, x, 5], preds[i, y, x, 6], preds[i, y, x, 7], preds[i, y, x, 8]])#x.y.w.h
                        label_box = torch.Tensor([labels[i, y, x, 0], labels[i, y, x, 1], labels[i, y, x, 2], labels[i, y, x, 3]])#label x.y.w.h
                        # calculate iou of two bbox
                        iou1 = self.calculate_iou(box1, label_box)
                        iou2 = self.calculate_iou(box2, label_box)

                        # judge responsible box
                        if iou1 > iou2:#box1 catch
                            loss_coord_xy += 5 * torch.sum((labels[i, y, x, 0:2] - preds[i, y, x, 0:2]) ** 2)#x.y (x-xi)**2 (y-yi)**2
                            loss_coord_wh += 5 * torch.sum((labels[i, y, x, 2:4].sqrt() - preds[i, y, x, 2:4].sqrt()) ** 2)
                            loss_obj += (preds[i, y, x, 4] - iou1) ** 2
                            loss_no_obj += 0.5 * ((0 - preds[i, y, x, 9]) ** 2)
                        else:
                            loss_coord_xy += 5 * torch.sum((labels[i, y, x, 5:7] - preds[i, y, x, 5:7]) ** 2)
                            loss_coord_wh += 5 * torch.sum((labels[i, y, x, 7:9].sqrt() - preds[i, y, x, 7:9].sqrt()) ** 2)
                            loss_obj += (preds[i, y, x, 9] - iou2) ** 2
                            loss_no_obj += 0.5 * ((0 - preds[i, y, x, 4]) ** 2)
                        loss_class += torch.sum((labels[i, y, x, 10:] - preds[i, y, x, 10:]) ** 2)
                    else:#区域内无东西
                        loss_no_obj += 0.5 * torch.sum((0 - preds[i, y, x, [4, 9]]) ** 2)#4，9 c1 c2
        loss = loss_coord_xy + loss_coord_wh + loss_obj + loss_no_obj + loss_class  # five loss terms
        return loss / batch_size


if __name__ == '__main__':
    from torchsummary import summary

    model = YOLOv1().to("cuda")
    summary(model, input_size=(3, 448, 448))
