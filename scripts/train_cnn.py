import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.utils import preprocess_image 

# 1. An cac thong bao log khong can thiet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
tf.get_logger().setLevel('ERROR')

# 2. Cau hinh
DATA_PATH = str(BASE_DIR / 'data') 
IMG_SIZE = 128

def load_data():
    X, y = [], []
    # Lay danh sach thu muc con, loai bo folder rac
    categories = [d for d in os.listdir(DATA_PATH) 
                  if os.path.isdir(os.path.join(DATA_PATH, d)) and d != 'nonsegmentedv2']
    
    print(f"--- Bat dau doc du lieu tu {len(categories)} loai cay ---")

    for cat in categories:
        folder_path = os.path.join(DATA_PATH, cat)
        files = os.listdir(folder_path)
        print(f"Dang xu ly: {cat}...")
        
        for img_name in files:
            img_path = os.path.join(folder_path, img_name)
            # Dung ham preprocess_image tu file utils.py
            processed_img, _, _ = preprocess_image(img_path, (IMG_SIZE, IMG_SIZE))
            
            if processed_img is not None:
                X.append(processed_img)
                y.append(cat)
                
    return np.array(X), np.array(y)

# Chay load du lieu
X, y_raw = load_data()

# Chuyen nhan chu thanh so
le = LabelEncoder()
y_int = le.fit_transform(y_raw)
y_cat = tf.keras.utils.to_categorical(y_int)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# 3. Xay dung CNN (Sua loi UserWarning bang cach dung Input layer)
model = models.Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)), # Lop Input rieng biet de het canh bao
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Huan luyen
print("\n--- Bat dau huan luyen (Training) ---")
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# 5. Luu ket qua
if not os.path.exists(str(BASE_DIR / 'models')): os.makedirs(str(BASE_DIR / 'models'))
model.save(str(BASE_DIR / 'models/plant_model.h5'))
np.save(str(BASE_DIR / 'models/classes.npy'), le.classes_)
print("\n--- Da hoan thanh! Model duoc luu tai folder 'models/' ---")