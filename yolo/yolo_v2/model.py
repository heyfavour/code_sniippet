"""
BN
"""


"""
anchor
不同理性的框
13*13  5个anchor  iou最大的anchor去负责
yolov1 2 (x y w h c) 20 c = 30  7*7*30
yolov2 5 (x y w h c 20 c) = 125 13*13*125
         bx = sigmod(tx)+cx
         by = sigmod(ty)+cy
         bw = pw  * e^tw
         bh = ph * e^th 
         cx cy grid cell 左上角的坐标 pw ph anchor 宽 高 

聚类->anchor 5个 

损失函数
loss = 所有anchor  (1 if iou<0.6 抛弃的iou else 0) * 参数 *(-预测置信度)**2                置信度误差 置信度越小越好
                +  (1 if t< 12800 else 0)*参数*sum(p-b)**2   p-anchar位置 b-预测框位置      让模型更快的学会预测anchor的位置
  一个GT仅分配一个anchor    + (1 if iou最大的负责 负责预测 else 0)  *(参数*sum(标注框位置-预测框位置)**2        定位误差
                                       + 参数 *(anchor_iou - 预测置信度)**2      置信度误差
                                       + 参数*(标注框类别-预测框类别)**2)           分类误差
"""

"""
"""