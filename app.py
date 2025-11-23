# app.py —— Render 專用版（必須在根目錄）
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods=['POST'])
def predict():
    return jsonify({
        "detections": [{
            "label": "凍檸茶",
            "confidence": 0.98,
            "nutrition": {"calories": 180},
            "portion_grams": 100
        }]
    })

@app.route('/')
def home():
    return jsonify({"message": "Render Theseus AI Running!"})

# Render 要求嘅端口變數
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)