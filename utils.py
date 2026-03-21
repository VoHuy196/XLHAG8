import cv2
import numpy as np

def preprocess_image(img_path, target_size=(128, 128)):
    """
    Ham tien xu ly: Quet nhiu, tach nen va chuan hoa anh.
    Giai quyet loi duong dan Unicode tren Windows.
    """
    # 1. Doc anh bang numpy de tranh loi ky tu la (Unicode/Encoding) tren Windows
    try:
        nparr = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None, None
    except Exception as e:
        print(f"Loi khi doc file {img_path}: {e}")
        return None, None, None

    # Chuyen sang RGB de hien thi dung mau
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. QUET NHIEU: dung Bilateral Filter de giu lai canh la sac net
    # d=9: duong kinh vung loc, 75, 75: do loc mau va khong gian
    filtered = cv2.bilateralFilter(img_rgb, 9, 75, 75)
    
    # 3. PHAN DOAN (Segmentation): Chuyen sang HSV de tach mau xanh la
    hsv = cv2.cvtColor(filtered, cv2.COLOR_RGB2HSV)
    
    # Dai mau xanh la cay (Green range)
    lower_green = np.array([25, 35, 35])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 4. MORPHOLOGY: Xu ly hinh thai hoc de lam sach mat na (mask)
    kernel = np.ones((3, 3), np.uint8)
    # Opening: Xoa cac dom nhiu trang nho vung nen (dat, da)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Closing: Lap day cac lo thung nho tren be mat la do phan xa anh sang
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 5. TACH LA: Ket hop mask voi anh da loc nhiu
    final_leaf = cv2.bitwise_and(filtered, filtered, mask=mask)
    
    # 6. RESIZE & CHUAN HOA
    final_leaf = cv2.resize(final_leaf, target_size)
    normalized_img = final_leaf / 255.0
    
    return normalized_img, mask, img_rgb

def extract_visual_features(mask):
    """
    Ham trich xuat dac trung hinh hoc tu mat na la cay.
    Dung de so sanh ML truyen thong hoac phan tich du lieu.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0, 0, 0, 0] # Tra ve 0 neu khong tim thay doi tuong
    
    # Lay duong bao lon nhat (la cay chinh)
    cnt = max(contours, key=cv2.contourArea)
    
    # Dien tich va chu vi
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # Do tron (Circularity)
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
    
    # Ty le dai/rong (Aspect Ratio)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h if h > 0 else 0
    
    return [area, perimeter, circularity, aspect_ratio]