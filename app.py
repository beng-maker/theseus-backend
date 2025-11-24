# app.py —— 永久雲端 + 100% 成功版（最終版！）
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import torch

app = Flask(__name__)
CORS(app)

print("載入 YOLOv8 模型中...")

# 關鍵：加入安全白名單！
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

model = YOLO('yolov8s.pt')  # 自動下載 + 安全載入！

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image"}), 400

        img_b64 = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        from PIL import Image
        import io, base64
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
    return jsonify({"message": "Theseus AI 永久雲端版上線！"})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
