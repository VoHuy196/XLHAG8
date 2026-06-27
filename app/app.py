import gradio as gr
import cv2
import numpy as np
import json
import joblib
import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path để có thể import từ src
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.segment_plant import preprocess_image
from src.feature_extraction import extract_features

MODELS_DIR = BASE_DIR / "models"

# 1. Load Random Forest model + scaler + class names
rf_candidates = sorted(MODELS_DIR.glob("rf_best_plant_*.pkl"), key=lambda p: p.stat().st_mtime)
if not rf_candidates:
    print("Warning: Không tìm thấy model Random Forest trong thư mục models/. App vẫn khởi động nhưng sẽ không predict được cho đến khi train xong.")
    model = None
else:
    model = joblib.load(rf_candidates[-1])

try:
    scaler = joblib.load(BASE_DIR / "data" / "scaler.pkl")
except:
    scaler = None
    print("Warning: scaler.pkl not found.")

try:
    with open(BASE_DIR / "config" / "label_map.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)["class_names"]
except:
    class_names = []
    print("Warning: label_map.json not found.")


def predict_plants(input_img):
    if model is None or scaler is None or not class_names:
        return None, {}, "Mô hình hoặc dữ liệu chưa sẵn sàng. Hãy chạy Airflow DAG để train model."
        
    if input_img is None:
        return None, {}, "Vui lòng chọn ảnh"

    # Lưu ảnh tạm để hàm preprocess_image có thể đọc (vì utils.py của Huy đọc từ path)
    temp_path = str(BASE_DIR / "temp_input.png")
    # Gradio truyền vào ảnh dạng RGB, OpenCV cần BGR để lưu
    cv2.imwrite(temp_path, cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))

    # 2. Tách nền lá để hiển thị ảnh segmented_leaf_resized
    segmented_leaf_resized, mask, _ = preprocess_image(temp_path)

    if segmented_leaf_resized is None:
        return None, {}, "Lỗi xử lý ảnh"

    # 3. Trích xuất feature + chuẩn hóa theo scaler đã dùng lúc train
    feature_vector = extract_features(temp_path)
    input_vector = scaler.transform(feature_vector.reshape(1, -1)).astype(np.float32)
    prediction = model.predict_proba(input_vector)

    class_idx = int(np.argmax(prediction[0]))
    result_text = class_names[class_idx]

    # 4. Trình bày kết quả
    # Tạo từ điển xác suất cho các loài (Gradio Label)
    confidences = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}

    return segmented_leaf_resized, confidences, f"Loài cây: {result_text}"


# --- GIAO DIỆN APP ---
with gr.Blocks(title="Plant Recognition System") as demo:
    gr.Markdown("# 🌱 Hệ thống Nhận diện mầm cây - Nhóm G5")
    gr.Markdown("### Môn: Xử lý hình ảnh (DIP) - Random Forest Classifier")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Chọn ảnh lá cây cần nhận diện")
            btn = gr.Button("🔍 Quét nhiễu & Nhận diện", variant="primary")

        with gr.Column():
            output_mask = gr.Image(label="Kết quả tách lá (segmented_leaf_resized)")
            output_label = gr.Label(label="Xác suất dự đoán", num_top_classes=3)
            output_text = gr.Textbox(label="Kết luận cuối cùng")

    # Sự kiện khi bấm nút
    btn.click(fn=predict_plants,
              inputs=input_image,
              outputs=[output_mask, output_label, output_text])

# Chạy App
if __name__ == "__main__":
    demo.launch()