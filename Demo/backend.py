import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify
import joblib
from joblib import dump, load
import base64
import io
from sklearn.pipeline import Pipeline
from PIL import Image
from transformers import AutoImageProcessor, ResNetModel
from transformers import CLIPModel, CLIPProcessor
import torch
import torch.nn.functional as F
from flask_cors import CORS
import torch
import torchvision.transforms as T
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

@app.route('/keepalive', methods=['GET'])
def api_health():
    return jsonify(Message="Success")

# Define a route for making predictions
@app.route("/predict-resnet50", methods=["POST"])
def predict():
    CLASS_MAP = {
        0: 'Banh beo',
        1: 'Banh bot loc',
        2: 'Banh can',
        3: 'Banh canh',
        4: 'Banh chung',
        5: 'Banh cuon',
        6: 'Banh duc',
        7: 'Banh gio',
        8: 'Banh khot',
        9: 'Banh mi',
        10: 'Banh pia',
        11: 'Banh tet',
        12: 'Banh trang nuong',
        13: 'Banh xeo',
        14: 'Bun bo Hue',
        15: 'Bun dau mam tom',
        16: 'Bun mam',
        17: 'Bun rieu',
        18: 'Bun thit nuong',
        19: 'Ca kho to',
        20: 'Canh chua',
        21: 'Cao lau',
        22: 'Chao long',
        23: 'Com tam',
        24: 'Goi cuon',
        25: 'Hu tieu',
        26: 'Mi quang',
        27: 'Nem chua',
        28: 'Pho',
        29: 'Xoi xeo'
    }   

    # Get JSON data from the request
    model = load("./models/svm_for_resnet50_joblib.pkl")
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    embed_model = ResNetModel.from_pretrained("microsoft/resnet-50")
    
    # Extract the base64 image from the request
    image_base64 = request.get_json().get("image")
    if not image_base64:
        return jsonify({"error": "No image provided"}), 400

    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Invalid image data"}), 400
    
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        outputs = embed_model(**inputs)
        feat_map = outputs.last_hidden_state
        gap = F.adaptive_avg_pool2d(feat_map, (1, 1))
        vec = gap.view(gap.size(0), -1)  
    # Use the loaded model to make predictions on the DataFrame
    # Move the tensor to the CPU and convert it to a NumPy array
    gap_numpy = vec.cpu().numpy()

    # Nếu đã train với probability=True thì dùng predict_proba
    proba = model.predict_proba(gap_numpy)[0]   # shape: (n_classes,)

    # mapping class -> probability (vd: "0": 0.8, "1": 0.2, ...)
    class_probs = {
        CLASS_MAP.get(int(cls)): float(p)
        for cls, p in zip(model.classes_, proba)
    }

    
    return jsonify({
        "Probabilities": class_probs
    })

@app.route('/predict-vitl14', methods=['POST'])
def predict_vit():
    CLASS_MAP = {
        0: 'Banh beo',
        1: 'Banh bot loc',
        2: 'Banh can',
        3: 'Banh canh',
        4: 'Banh chung',
        5: 'Banh cuon',
        6: 'Banh duc',
        7: 'Banh gio',
        8: 'Banh khot',
        9: 'Banh mi',
        10: 'Banh pia',
        11: 'Banh tet',
        12: 'Banh trang nuong',
        13: 'Banh xeo',
        14: 'Bun bo Hue',
        15: 'Bun dau mam tom',
        16: 'Bun mam',
        17: 'Bun rieu',
        18: 'Bun thit nuong',
        19: 'Ca kho to',
        20: 'Canh chua',
        21: 'Cao lau',
        22: 'Chao long',
        23: 'Com tam',
        24: 'Goi cuon',
        25: 'Hu tieu',
        26: 'Mi quang',
        27: 'Nem chua',
        28: 'Pho',
        29: 'Xoi xeo'
    }   

    # Get JSON data from the request
    model = load("./models/svm_for_vitl14_joblib.pkl")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    embed_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    embed_model.eval()
    
    # Extract the base64 image from the request
    image_base64 = request.get_json().get("image")
    if not image_base64:
        return jsonify({"error": "No image provided"}), 400

    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Invalid image data"}), 400
    
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        outputs = embed_model.get_image_features(**inputs)[0]

    feat = np.array(outputs).reshape(1, -1)  # (1, D)

    proba = model.predict_proba(feat)[0]     # (n_classes,)

    # mapping class -> probability (vd: "0": 0.8, "1": 0.2, ...)
    class_probs = {
        CLASS_MAP.get(int(cls)): float(p)
        for cls, p in zip(model.classes_, proba)
    }

    return jsonify({
        "Probabilities": class_probs
    })


@app.route('/predict-dinov2', methods=['POST'])
def predict_dino():
    CLASS_MAP = {
    "Banh beo": 0,
    "Banh bot loc": 1,
    "Banh can": 2,
    "Banh canh": 3,
    "Banh chung": 4,
    "Banh cuon": 5,
    "Banh duc": 6,
    "Banh gio": 7,
    "Banh khot": 8,
    "Banh mi": 9,
    "Banh pia": 10,
    "Banh tet": 11,
    "Banh trang nuong": 12,
    "Banh xeo": 13,
    "Bun bo Hue": 14,
    "Bun dau mam tom": 15,
    "Bun mam": 16,
    "Bun rieu": 17,
    "Bun thit nuong": 18,
    "Ca kho to": 19,
    "Canh chua": 20,
    "Cao lau": 21,
    "Chao long": 22,
    "Com tam": 23,
    "Goi cuon": 24,
    "Hu tieu": 25,
    "Mi quang": 26,
    "Nem chua": 27,
    "Pho": 28,
    "Xoi xeo": 29
    }
    
    CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
    
    embed_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    embed_model.eval()
    dinov2_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    svm = load("./models/dinov2_svm_best.pkl")
    # Extract the base64 image from the request
    image_base64 = request.get_json().get("image")
    if not image_base64:
        return jsonify({"error": "No image provided"}), 400

    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Invalid image data"}), 400
    
    image = dinov2_transform(image).unsqueeze(0)
    output = embed_model(image)
    
    proba = svm.predict_proba(output.detach().cpu().numpy())[0]   # (n_classes,)
    
    class_probs = { CLASS_MAP.get(int(cls)): float(p) for cls, p in zip(svm.classes_, proba) }
    
    return jsonify({
        "Probabilities": class_probs
    })
    

# Run the Flask app when this script is executed
if __name__ == "__main__":
    app.run(debug=True)
