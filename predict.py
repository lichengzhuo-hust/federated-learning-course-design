# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from common import MNIST_CNN
import sys

def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    model = MNIST_CNN()
    model.load_state_dict(torch.load('mnist_fed_model.pth', map_location='cpu'))
    model.eval()
    
    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img) # 白底黑字 -> 黑底白字
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()
    print(f"🤖 图片: {image_path} -> 识别结果是 【 {pred} 】")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "test_7.png"
        print(f"⚠️ 未指定图片，默认使用: {image_path}")
    predict_image(image_path)
