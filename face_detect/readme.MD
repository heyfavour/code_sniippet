人脸识别 区域->特征点提取->人脸对齐(68个点)->人脸比对(多维特征向量 128维度)


GitHub项目地址face_recognition
该项目目前star数超过3万，是GitHub上最主流的人脸识别工具包之一。
Face_recognition主要参考了OpenFace项目以及谷歌的facenet

facenet face->128dim->L2->比对 欧式距离

## facenet
model
```angular2html
mobilenetv1->轻量级深层神经网络 ->深度可分离卷积+1*1普通卷积 使用更少的参数代替3*3卷积
inception-resetv1


torch.nn.AdaptiveAvgPool2d(output_size)
区别主要体现在自适应上，对于任何输入大小，输出大小均为指定的H×W大小

L2 标准化
F.normalize(x,p=2,dim=1) batsize L2
p=2->L2
```

### LOSS
triple loss  
```angular2html
不同人脸欧式距离扩张 同一人脸欧式距离减少
```

cross-entry loss
```angular2html
人脸分类 辅助triple loss收敛
```

Face_recognition
```angular2html
pip install cmake
pip install face_recognition
```