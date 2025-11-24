import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules.conv import Conv
from torch.nn.modules.conv import Conv2d   # PyTorch 內建 Conv2d

# 設定安全白名單，允許以上類別安全反序列化
torch.serialization.add_safe_globals([DetectionModel, Sequential, Conv, Conv2d])

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

print("載入 YOLOv8 模型中...")
model = YOLO('yolov8s.pt')  # 自動下載 + 安全載入

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image"}), 400

        img_b64 = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes))

        results = model(img, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls)]
                conf = float(box.conf)
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "calories": 180 if "tea" in label.lower() else 0,
                    "portion_grams": 100
                })

        return jsonify({"detections": detections})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({"message": "Theseus AI 永久雲端版上線！", "status": "LIVE"})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
