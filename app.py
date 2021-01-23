import io
import json

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from effdet import create_model

model_path = './output/train/model_best.pth.tar'
app = Flask(__name__)
confidence_thresh = 0.35
model = create_model(
        'efficientdet_d0',
        bench_task='predict',
        num_classes=20,
        pretrained=True,
        redundant_bias=False,
        soft_nms=False,
        checkpoint_path=model_path,
        checkpoint_ema=False,
)
model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose( [
        transforms.Resize((512, 512)),
        transforms.ToTensor(), 
        transforms.Normalize( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]) 
    image = Image.open(io.BytesIO(image_bytes)) 
    image = image.convert('RGB')
    return my_transforms(image).unsqueeze(0) 

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    detections = model.forward(tensor)
    detections = detections.detach().numpy().reshape((100, 6))
    detections = detections[detections[:, 4] > confidence_thresh].astype('float16')
    bboxes = [
      {
        'min_x': int(detections[idx, 0]), 
        'min_y': int(detections[idx, 1]), 
        'max_x': int(detections[idx, 2]),
        'max_y': int(detections[idx, 3]),
        'score': float(detections[idx, 4]),
        'category': int(detections[idx, 5]),
      }
      for idx in range(len(detections))
    ]
    return bboxes


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        output = get_prediction(image_bytes=img_bytes)
        print(output)
        return jsonify(output)

if __name__ == '__main__':
    def test_img(img_path):
        import requests
        resp = requests.post("http://localhost:5000/predict", files={"file": open(img_path,'rb')})
    app.run()

