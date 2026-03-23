import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_pipeline_steps(images_and_titles):
    """Hiển thị toàn bộ các bước xử lý theo dạng lưới."""
    cols = 3
    rows = int(np.ceil(len(images_and_titles) / cols))
    plt.figure(figsize=(18, 5 * rows))

    for i, (title, image, is_gray) in enumerate(images_and_titles, start=1):
        plt.subplot(rows, cols, i)
        plt.imshow(image, cmap="gray" if is_gray else None)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()


def preprocess_image(img_path, target_size=(128, 128), debug_steps=None):
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
    if debug_steps is not None:
        debug_steps["img_rgb"] = img_rgb

    # 2. KHU NHIU GAUSS: lam muot nhieu tan so cao truoc khi tao mask
    filtered = cv2.GaussianBlur(img_rgb, (5, 5), 0)
    if debug_steps is not None:
        debug_steps["filtered"] = filtered

    # 3. PHAN DOAN (Segmentation): Chuyen sang HSV de tach mau xanh la
    hsv = cv2.cvtColor(filtered, cv2.COLOR_RGB2HSV)
    if debug_steps is not None:
        debug_steps["hsv_display"] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Dai mau xanh la cay (Green range)
    lower_green = np.array([25, 35, 35])
    upper_green = np.array([95, 255, 255])
    mask_raw = cv2.inRange(hsv, lower_green, upper_green)
    if debug_steps is not None:
        debug_steps["mask_raw"] = mask_raw

    # 4. MORPHOLOGY: Xu ly hinh thai hoc de lam sach mat na (mask)
    kernel = np.ones((3, 3), np.uint8)
    # Opening: Xoa cac dom nhiu trang nho vung nen (dat, da)
    mask_open = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    if debug_steps is not None:
        debug_steps["mask_open"] = mask_open
    # Closing: Lap day cac lo thung nho tren be mat la do phan xa anh sang
    mask = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel, iterations=1)
    if debug_steps is not None:
        debug_steps["mask"] = mask

    # 5. TACH LA: Ket hop mask voi anh da loc nhiu
    final_leaf = cv2.bitwise_and(filtered, filtered, mask=mask)
    if debug_steps is not None:
        debug_steps["segmented_raw"] = final_leaf

    # 6. RESIZE
    final_leaf = cv2.resize(final_leaf, target_size)
    if debug_steps is not None:
        debug_steps["segmented_resized"] = final_leaf

    return final_leaf, mask, img_rgb


def segment_plant(image_path, show_steps=True):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Khong tim thay anh: {image_path}")

    target_size = (128, 128)
    debug_steps = {}
    segmented_leaf_resized, mask, img_rgb = preprocess_image(
        str(image_path), target_size=target_size, debug_steps=debug_steps
    )
    if segmented_leaf_resized is None:
        raise ValueError(f"Khong doc duoc anh: {image_path}")

    if show_steps:
        steps = []

        def add_step(title, key, is_gray):
            image = debug_steps.get(key)
            if image is not None:
                steps.append((title, image, is_gray))

        add_step("1. Ảnh gốc (RGB)", "img_rgb", False)
        add_step("2. Sau Gaussian blur", "filtered", False)
        add_step("3. HSV (hiển thị RGB)", "hsv_display", False)
        add_step("4. Mask xanh ban đầu", "mask_raw", True)
        add_step("5. Mask sau opening", "mask_open", True)
        add_step("6. Mask sau closing", "mask", True)
        add_step("7. Kết quả tách lá", "segmented_raw", False)
        add_step("8. Kết quả resize 128x128", "segmented_resized", False)

        if not steps:
            steps = [
                ("1. Ảnh gốc (RGB)", img_rgb, False),
                ("2. Mask sau morphology", mask, True),
                ("3. Kết quả resize 128x128", segmented_leaf_resized, False),
            ]
        show_pipeline_steps(steps)

    return {
        "image_path": str(image_path),
        "img_rgb": img_rgb,
        "mask": mask,
        "target_size": target_size,
        "segmented_leaf": segmented_leaf_resized,
    }


def main():

    parser = argparse.ArgumentParser(description="Segment green plant regions from an image.")
    default_image = Path(__file__).resolve().parent / "dataset" / "Black-grass" / "2.png"
    parser.add_argument("--image", default=str(default_image), help="Path to the input image")
    parser.add_argument("--no-show", action="store_true", help="Do not display matplotlib windows")
    args = parser.parse_args()

    segment_plant(args.image, show_steps=not args.no_show)


if __name__ == "__main__":
    main()
