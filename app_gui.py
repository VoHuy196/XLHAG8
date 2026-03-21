import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from utils import preprocess_image

# 1. Load Model đã train
model = tf.keras.models.load_model('models/plant_model.h5')
class_names = np.load('models/classes.npy')

def predict_plants(input_img):
    if input_img is None:
        return None, "Vui lòng chọn ảnh", ""

    # Lưu ảnh tạm để hàm preprocess_image có thể đọc (vì utils.py của Huy đọc từ path)
    temp_path = "temp_input.png"
    # Gradio truyền vào ảnh dạng RGB, OpenCV cần BGR để lưu
    cv2.imwrite(temp_path, cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))
    
    # 2. Gọi hàm Xử lý hình ảnh từ utils.py
    processed_img, mask, _ = preprocess_image(temp_path)
    
    if processed_img is None:
        return None, "Lỗi xử lý ảnh", ""

    # 3. Dự đoán bằng CNN
    input_tensor = np.expand_dims(processed_img, axis=0).astype(np.float32)
    prediction = model.predict(input_tensor)
    
    class_idx = np.argmax(prediction)
    result_text = class_names[class_idx]
    confidence = float(prediction[0][class_idx])

    # 4. Trình bày kết quả
    # Tạo từ điển xác suất cho các loài (Gradio Label)
    confidences = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
    
    return mask, confidences, f"Loài cây: {result_text}"

# --- GIAO DIỆN APP ---
with gr.Blocks(title="Plant Recognition System") as demo:
    gr.Markdown("# 🌱 Hệ thống Nhận diện mầm cây - Nhóm G5")
    gr.Markdown("### Môn: Xử lý hình ảnh (DIP) - CNN Classifier")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Chọn ảnh lá cây cần nhận diện")
            btn = gr.Button("🔍 Quét nhiễu & Nhận diện", variant="primary")
        
        with gr.Column():
            output_mask = gr.Image(label="Kết quả Quét nhiễu & Tách nền (Mask)")
            output_label = gr.Label(label="Xác suất dự đoán", num_top_classes=3)
            output_text = gr.Textbox(label="Kết luận cuối cùng")

    # Sự kiện khi bấm nút
    btn.click(fn=predict_plants, 
              inputs=input_image, 
              outputs=[output_mask, output_label, output_text])

# Chạy App
if __name__ == "__main__":
    demo.launch()