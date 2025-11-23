# app.py —— 真正 YOLOv8 版（取代之前簡化版）
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

print("載入 YOLOv8 模型中...")
model = YOLO('yolov8s.pt')  # 自動下載入官方模型

nutrition_db = {
    "banana": {"calories": 105},
    "apple": {"calories": 95},
    "orange": {"calories": 62},
    "sandwich": {"calories": 350},
    "pizza": {"calories": 285},
    "bottle": {"calories": 0},
    "cup": {"calories": 0},
    "mouse": {"calories": 0},
    "keyboard": {"calories": 0},
    "default": {"calories": 0}
}

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_b64 = data['image'].split(',')[1] if ',' in data['image'] else data['image']
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_bytes))

    results = model(img, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            nutrition = nutrition_db.get(label.lower(), nutrition_db["default"])

            detections.append({
                "label": label,
                "confidence": conf,
                "nutrition": nutrition,
                "portion_grams": 100
            })

    return jsonify({"detections": detections})

@app.route('/')
def home():
    return jsonify({"status": "Theseus Real YOLOv8 Server Running!"})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
