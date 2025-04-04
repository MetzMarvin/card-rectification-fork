import cv2
import numpy as np
from PIL import ImageEnhance

from load_model import load_model
from rectify import inference
from pdf2image import convert_from_path


"""
1. load_model()
根据模型类型，导入存储在硬盘中的模型文件至内存。

Parameters: 
None

Returns:
- model: {UNetRNN}
模型对象，包括模型各层结构和预训练的参数。
- device: {device}
torch.device类对象，表示分配给torch.Tensor进行运算的设备。包含设备类型（"cpu"或"cuda"）和设备序号。

Example:
from load_model import load_model
model, device = load_model()


2. inferecne(input_path, output_path, model, device)
校正推理，对单张图像进行校正处理。

Parameters:
- input_path: {str}
待校正图像路径

- output_path: {str}
图像保存路径

- model: {UNetRNN}
模型对象，包括模型各层结构和预训练的参数。

- device: {device}
torch.device类对象，表示分配给torch.Tensor进行运算的设备。包含设备类型（"cpu"或"cuda"）和设备序号。

Example:
from rectify import inference
from load_model import load_model
input = 'example/card.jpg'
output = 'result/card.png'
model, device = load_model()
inference(input, output, trained_model, device)

"""

if __name__ == "__main__":
    """
    Demo
    """
    image = convert_from_path('example/perso_mm.pdf')[0]
    # Rotate image (choose angle as needed: 90, 180, 270)
    rotated_image = image.rotate(90, expand=True)

    # Increase contrast (factor > 1 increases contrast, 1.5-2.0 is a good starting point)
    enhancer = ImageEnhance.Contrast(rotated_image)
    enhanced_image = enhancer.enhance(1.9)  # Adjust factor as needed

    # Convert to numpy array for OpenCV
    page_np = np.array(enhanced_image)
    # Save the image temporarily (the rectifier requires file paths)
    temp_input_path = "example/perso_mm.jpg"
    cv2.imwrite(temp_input_path, page_np)
    input1 = 'example/perso_mm.jpg'
    output1 = 'result/perso_mm.png'

    trained_model, device = load_model()
    inference(input1, output1, trained_model, device)

    print("Done.")
