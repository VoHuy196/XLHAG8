import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import preprocess_image
import os

# 1. Load model và data test
model = tf.keras.models.load_model('models/plant_model.h5')
class_names = np.load('models/classes.npy')

# Giả sử đã có X_test và y_test từ file train, 
# Nếu không, ta sẽ load lại một phần nhỏ từ folder data để test
def evaluate_on_folder(test_path='data'):
    X_test, y_true = [], []
    categories = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d)) and d != 'nonsegmentedv2']
    
    print("Dang chuan bi du lieu de tinh F1-score...")
    for cat in categories:
        folder = os.path.join(test_path, cat)
        # Lay moi loài 20 anh de test nhanh
        for img_name in os.listdir(folder)[:20]: 
            processed_img, _, _ = preprocess_image(os.path.join(folder, img_name))
            if processed_img is not None:
                X_test.append(processed_img)
                y_true.append(cat)
    
    X_test = np.array(X_test)
    
    # 2. Du doan
    y_pred_probs = model.predict(X_test)
    y_pred_labels = [class_names[np.argmax(p)] for p in y_pred_probs]
    
    # 3. In Classification Report (Co F1-score o day)
    print("\n--- KET QUA DANH GIA CHI TIET (F1-SCORE) ---")
    print(classification_report(y_true, y_pred_labels, target_names=class_names))
    
    # 4. Ve Confusion Matrix (Ma tran nham lan)
    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Du doan')
    plt.ylabel('Thuc te')
    plt.title('Confusion Matrix - Nhom G5')
    plt.show()

evaluate_on_folder()