import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from skimage import color
from skimage.transform import resize
from segment_plant import segment_plant

"""
Extract Color Histogram features
"""
def extract_color_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])

    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()

    return np.concatenate([h_hist, s_hist, v_hist])

"""
Extract texture features
choose HOG
"""
def extract_hog(image):
    image_resized = resize(image, (128, 128))
    gray = color.rgb2gray(image_resized)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features

"""
Extract Shape features
choose Contour area
feature size = 4
"""
def extract_shape(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.zeros(4)
    
    c = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)

    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h

    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else 0

    return np.array([area, perimeter, aspect_ratio, solidity])


def extract_lbp(image):
    """Trích xuất kết cấu gân lá bằng LBP"""
    # Các tham số chuẩn của LBP
    radius = 3
    n_points = 8 * radius

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tính toán LBP (sử dụng phương pháp 'uniform' để giảm chiều dữ liệu và đạt bất biến với phép xoay)
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

    # Cố định số bin để đảm bảo vector luôn có cùng độ dài (n_points + 2)
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

    # Chuẩn hóa histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist

def final_vector(image):
    color_feat = extract_color_histogram(image)
    texture_hog = extract_hog(image)
    shape_feat = extract_shape(image)
    texture_lbp = extract_lbp(image)

    # Gộp tất cả (Màu sắc, HOG, LBP, Hình dáng cơ bản)
    return np.concatenate([color_feat, texture_hog, texture_lbp, shape_feat])

def _segment_for_features(image_path):
    result = segment_plant(image_path, show_steps=False)
    segmented_leaf = result.get("segmented_leaf")
    if segmented_leaf is None:
        raise KeyError("segment_plant did not return 'segmented_leaf'")
    return cv2.cvtColor(segmented_leaf, cv2.COLOR_RGB2BGR)


def extract_features(image_path):
    segmented_bgr = _segment_for_features(image_path)
    return final_vector(segmented_bgr).astype(np.float32)

def build_dataset(dataset_path):
    X         = []
    y_plant   = []   # integer label -> 12 plant classes

    print("Scanning dataset folders...")

    all_plants = sorted([
        folder_name
        for folder_name in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, folder_name))
    ])
    plant_to_idx = {name: idx for idx, name in enumerate(all_plants)}

    print(f"Plant classes ({len(all_plants)}): {all_plants}")
    if len(all_plants) != 13:
        print(f"[WARN] Expected 13 classes in dataset, found {len(all_plants)}")

    for folder_name in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        p_idx = plant_to_idx[folder_name]
        print(f"Processing '{folder_name}' -> class={p_idx}")

        for img_name in tqdm(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img_name)
            if not os.path.isfile(img_path):
                continue

            try:
                segmented_bgr = _segment_for_features(img_path)
                X.append(final_vector(segmented_bgr))
            except Exception as exc:
                print(f"  [SKIP] {img_path}: {exc}")
                continue

            y_plant.append(p_idx)

    X         = np.array(X,         dtype=np.float32)
    y_plant   = np.array(y_plant,   dtype=np.int32)

    print("\nFinal dataset shape :", X.shape)
    print("y_plant  shape      :", y_plant.shape)

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    np.save(os.path.join(data_dir, "X.npy"),         X)
    np.save(os.path.join(data_dir, "y_plant.npy"),   y_plant)

    label_map = {
        "plant": plant_to_idx,
        "class_names": all_plants,
        "num_classes": len(all_plants)
    }
    with open(os.path.join(base_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print("  data/X.npy")
    print("  data/y_plant.npy  (class index for 12 plant classes)")
    print("  label_map.json")


if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), "data")
    build_dataset(dataset_path)
