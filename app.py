# app.py —— Render 免費版 100% 成功版（最終版！）
from flask import Flask, request, jsonify
from flask_cors import CORS
import ultralytics
print("Ultralytics 版本:", ultralytics.__version__)  # 會顯示 8.0.x 或 8.1.x

from ultralytics import YOLO
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

print("載入 YOLOv8 模型中...")
model = YOLO('yolov8s.pt')  # 直接載入！免費版 100% 成功！

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
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
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "calories": 180 if "tea" in label.lower() else 0,
                    "portion_grams": 100
                })

        return jsonify({"detections": detections})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({"message": "Theseus AI 永久雲端版上線！", "ultralytics": ultralytics.__version__})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
