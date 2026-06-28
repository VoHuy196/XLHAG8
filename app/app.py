import gradio as gr
import cv2
import numpy as np
import json
import joblib
import sys
from pathlib import Path
import os
import mlflow

# Thêm thư mục gốc vào sys.path để có thể import từ src
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.segment_plant import preprocess_image as rf_preprocess
from src.feature_extraction import extract_features
from src.utils import preprocess_image as cnn_preprocess

MODELS_DIR = BASE_DIR / "models"

# ==========================================
# 1. Load Random Forest model + scaler + class names
# ==========================================
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

rf_model = None
try:
    print("Loading RF model from MLflow Model Registry...")
    model_uri = "models:/PlantDisease_RF_Model/Production"
    rf_model = mlflow.sklearn.load_model(model_uri)
    print("RF Model loaded successfully from MLflow.")
except Exception as e:
    print(f"Warning: Failed to load RF model from MLflow: {e}")
    print("Trying local models/ fallback...")
    rf_candidates = sorted(MODELS_DIR.glob("rf_best_plant_*.pkl"), key=lambda p: p.stat().st_mtime)
    if not rf_candidates:
        print("Warning: Không tìm thấy model Random Forest.")
    else:
        rf_model = joblib.load(rf_candidates[-1])
        print("RF model loaded from local fallback.")

try:
    rf_scaler = joblib.load(BASE_DIR / "data" / "scaler.pkl")
except:
    rf_scaler = None
    print("Warning: scaler.pkl not found.")

try:
    with open(BASE_DIR / "config" / "label_map.json", "r", encoding="utf-8") as f:
        rf_class_names = json.load(f)["class_names"]
except:
    rf_class_names = []
    print("Warning: label_map.json not found.")

# ==========================================
# 2. Load CNN model (Optional)
# ==========================================
cnn_model = None
cnn_class_names = []
try:
    import tensorflow as tf
    cnn_model = tf.keras.models.load_model(str(BASE_DIR / 'models/plant_model.h5'))
    print("CNN model loaded successfully.")
    cnn_class_names = np.load(str(BASE_DIR / 'models/classes.npy'))
except ImportError:
    print("Warning: Thư viện tensorflow không được cài đặt. Không thể dùng mô hình CNN.")
except Exception as e:
    print(f"Warning: Lỗi khi load mô hình CNN: {e}")

def predict_plants(input_img, model_type):
    if input_img is None:
        return None, {}, "Vui lòng chọn ảnh"

    # Lưu ảnh tạm để các hàm preprocess có thể đọc
    temp_path = str(BASE_DIR / "temp_input.png")
    cv2.imwrite(temp_path, cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))

    if model_type == "Random Forest":
        if rf_model is None or rf_scaler is None or not rf_class_names:
            return None, {}, "Mô hình RF hoặc dữ liệu chưa sẵn sàng."

        # Tách nền lá cho RF
        segmented_leaf_resized, mask, _ = rf_preprocess(temp_path)
        if segmented_leaf_resized is None:
            return None, {}, "Lỗi xử lý ảnh (RF)"

        # Trích xuất feature + chuẩn hóa
        feature_vector = extract_features(temp_path)
        input_vector = rf_scaler.transform(feature_vector.reshape(1, -1)).astype(np.float32)
        prediction = rf_model.predict_proba(input_vector)

        class_idx = int(np.argmax(prediction[0]))
        result_text = rf_class_names[class_idx]
        confidences = {rf_class_names[i]: float(prediction[0][i]) for i in range(len(rf_class_names))}
        
        return segmented_leaf_resized, confidences, f"Loài cây (Random Forest): {result_text}"

    elif model_type == "CNN":
        if cnn_model is None or len(cnn_class_names) == 0:
            return None, {}, "Mô hình CNN chưa sẵn sàng hoặc thiếu thư viện tensorflow."

        # Xử lý hình ảnh cho CNN
        processed_img, mask, _ = cnn_preprocess(temp_path)
        if processed_img is None:
            return None, {}, "Lỗi xử lý ảnh (CNN)"

        input_tensor = np.expand_dims(processed_img, axis=0).astype(np.float32)
        prediction = cnn_model.predict(input_tensor)
        
        class_idx = int(np.argmax(prediction))
        result_text = str(cnn_class_names[class_idx])
        
        confidences = {str(cnn_class_names[i]): float(prediction[0][i]) for i in range(len(cnn_class_names))}
        
        return mask, confidences, f"Loài cây (CNN): {result_text}"

    return None, {}, "Loại mô hình không hợp lệ."

# --- GIAO DIỆN APP ---
with gr.Blocks(title="Plant Recognition System") as demo:
    gr.Markdown("# 🌱 Hệ thống Nhận diện mầm cây - Nhóm G5")
    gr.Markdown("### Môn: Xử lý hình ảnh (DIP) - Hỗ trợ Random Forest & CNN Classifier")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Chọn ảnh lá cây cần nhận diện")
            model_dropdown = gr.Dropdown(choices=["Random Forest", "CNN"], value="Random Forest", label="Chọn thuật toán dự đoán")
            btn = gr.Button("🔍 Quét nhiễu & Nhận diện", variant="primary")

        with gr.Column():
            output_mask = gr.Image(label="Kết quả tách lá (Mask / Segmented)")
            output_label = gr.Label(label="Xác suất dự đoán", num_top_classes=3)
            output_text = gr.Textbox(label="Kết luận cuối cùng")

    # Sự kiện khi bấm nút
    btn.click(fn=predict_plants,
              inputs=[input_image, model_dropdown],
              outputs=[output_mask, output_label, output_text])

# Chạy App
if __name__ == "__main__":
    demo.launch()