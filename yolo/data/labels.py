import os
from xml.etree.ElementTree import parse
"""
['aeroplane', 'chair', 'person', 'bus', 'car', 'horse', 'motorbike', 'bicycle', 'boat', 'cat', 'bottle', 'sofa', 'tvmonitor', 'pottedplant', 'cow', 'train', 'diningtable', 'bird', 'dog', 'sheep']
"""
objects_list = ['aeroplane', 'chair', 'person', 'bus', 'car', 'horse', 'motorbike', 'bicycle', 'boat', 'cat', 'bottle', 'sofa', 'tvmonitor', 'pottedplant', 'cow', 'train', 'diningtable', 'bird', 'dog', 'sheep']
def run():
    files = os.listdir("Annotations")
    class_names = set()
    for file in files:
        print(file)
        with open(f"Annotations/{file}") as f:
            ele = parse(f)
            root = ele.getroot()
            _object_list = []
            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)
            for obj in root.iterfind('object'):
                name = obj.find("name").text
                class_names.add(name)
                box = obj.find("bndbox")
                xmin = int(box.find("xmin").text)
                ymin = int(box.find("ymin").text)
                xmax = int(box.find("xmax").text)
                ymax = int(box.find("ymax").text)
                x,y,w,h = (xmax+xmin)/2,(ymin+ymax)/2,xmax-xmin,ymax-ymin
                _object_list.append([x/width,y/height,w/width,h/height,objects_list.index(name)])
        file_name = file.split('.')[0]
        print(file_name)
        with open(f"Labels/{file.split('.')[0]}","w") as f:
            for k,line in enumerate(_object_list):
                line = " ".join(["%.4f"%v if k <=3 else str(v) for k,v in enumerate(line)])
                if k!=len(_object_list)-1:f.write(line+"\n")
                else:f.write(line)


if __name__ == '__main__':
    run()