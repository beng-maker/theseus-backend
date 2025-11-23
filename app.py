# backend/app.py —— 宇宙級真實 AI 版
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import torch
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# --- 載入真實 YOLOv8 模型 ---
print("載入真實 YOLOv8 模型中...")
model = YOLO('yolov8s.pt')  # 替換成你嘅 .pt 檔

# --- 真實營養模型（可替換）---
nutrition_db = {
    "凍檸茶": {"calories": 180, "protein": 0, "carbs": 45, "fat": 0},
    "漢堡": {"calories": 550, "protein": 25, "carbs": 40, "fat": 30},
    "沙律": {"calories": 120, "protein": 5, "carbs": 15, "fat": 7},
    "default": {"calories": 100, "protein": 2, "carbs": 20, "fat": 1}
}

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image"}), 400

        # 解碼圖片
        img_b64 = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # --- 真實 YOLOv8 推理 ---
        results = model(img, verbose=False)
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                # 營養查表
                nutrition = nutrition_db.get(label, nutrition_db["default"])

                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": box.xyxy[0].tolist(),
                    "nutrition": nutrition,
                    "portion_grams": 100
                })

        return jsonify({"detections": detections})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({"status": "Theseus AI Real Server", "model": "YOLOv8 Live"})

if __name__ == '__main__':
    print("宇宙級真實 AI 伺服器啟動！")
    app.run(host='0.0.0.0', port=5000, debug=False)