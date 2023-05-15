import torch
import onnx
import onnxruntime
import time
from torchvision import models


def out_onnx():
    device = "cpu"
    model = models.resnet18(weights=models.ResNet18_Weights)
    model = model.eval().to(device)

    input = torch.randn(1, 3, 256, 256).to(device)
    print(input.shape)
    start_time = time.time()
    output = model(input)
    end_time = time.time()
    print("处理一张照片的时间为:", end_time - start_time)
    print(output.shape)

    """
    - export_params: 是否导出模型的参数，默认为 True ¹。
    - verbose: 是否打印出导出的 ONNX 模型的文本表示，默认为 False ¹。
    - training: 模型的训练状态，可以是 TrainingMode.EVAL（默认），TrainingMode.TRAINING 或 TrainingMode.PRESERVE ¹。
    - input_names 和 output_names: 模型输入和输出的名称，用于在 ONNX 模型中标识张量，默认为 None ¹。
    - opset_version: ONNX 的版本号，默认为 None，表示使用最新的版本 ¹。
    - do_constant_folding: 是否对常量进行折叠优化，默认为 True ¹。
    - dynamic_axes: 指定哪些输入或输出的维度是动态变化的，默认为 None ¹。
    - use_external_data_format: 是否将大于 2GB 的张量存储在外部文件中，默认为 False ¹。
    """
    with torch.no_grad():
        torch.onnx.export(
            model,  # 需要转换的模型
            input,  # 随机输入
            "resnet18_weights.onnx",  # 导出文件名
            opset_version=11,  # 算子版本
            input_names=["input"],
            output_names=["output"],
        )


def load_onnx():
    onnx_model = onnx.load("resnet18_weights.onnx")
    onnx.checker.check_model(onnx_model)
    # netron.app 可视化 onnx
    # print(onnx.helper.printable_graph(onnx_model.graph))


def onnxruntime_load():
    session = onnxruntime.InferenceSession('resnet18_weights.onnx')
    input = {'input': torch.randn(1, 3, 256, 256).numpy()}
    start_time = time.time()
    output = session.run(['output'], input)[0]
    end_time = time.time()
    print("处理一张照片的时间为:",end_time-start_time)
    print(output.shape)


if __name__ == '__main__':
    out_onnx()
    #load_onnx()
    onnxruntime_load()
