import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.utils import preprocess_image
import os

# 1. Load model và danh sách loài
model = load_model(str(BASE_DIR / 'models/plant_model.h5'))
class_names = np.load(str(BASE_DIR / 'models/classes.npy'))

def run_demo(img_path):
    # Kiểm tra file có tồn tại không trước khi làm bất cứ việc gì
    if not os.path.exists(img_path):
        print(f"Loi: Khong tim thay file '{img_path}'. Huy hay kiem tra lai duong dan!")
        return

    # 2. Tien xu ly (Quet nhiu, tach nen)
    processed_img, mask, original = preprocess_image(img_path)
    
    # Kiem tra neu ham preprocess tra ve None (loi doc file)
    if processed_img is None:
        print("Loi: Khong the xu ly hinh anh nay.")
        return
    
    # 3. Predict (Du doan)
    # Chuyen anh ve dang 4D (1, 128, 128, 3) de phu hop voi CNN
    input_data = np.expand_dims(processed_img, axis=0).astype(np.float32)
    
    pred = model.predict(input_data)
    class_idx = np.argmax(pred)
    result = class_names[class_idx]
    confidence = pred[0][class_idx] * 100
    
    print("-" * 30)
    print(f"KET QUA NHAN DIEN: {result}")
    print(f"Do tin cay: {confidence:.2f}%")
    print("-" * 30)
    
    # 4. Hien thi ket qua quet nhiu de bao cao
    # Resize mask de hien thi chung voi anh goc
    mask_visual = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((cv2.resize(original, (300, 300)), 
                          cv2.resize(mask_visual, (300, 300))))
    
    cv2.imshow("Original vs Mask (DIP Result)", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Chay thu
run_demo('D:\G5\data\Cleavers\1.png')