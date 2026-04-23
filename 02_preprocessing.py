import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import config

# Đường dẫn (giữ nguyên như bài trước)
BASE_DIR = config.BASE_DIR
DATASET_PATH = config.DATASET_PATH
SAMPLE_IMAGE = "101_1.tif"

def normalize_image(img):
    """
    Chuẩn hóa ảnh về mean 0 và variance 1
    Công thức đơn giản: (Pixel - Mean) / Std_Dev
    """
    img = img.astype(np.float32)
    mean = np.mean(img)
    std = np.std(img)
    if std == 0: std = 1 # Tránh chia cho 0
    norm_img = (img - mean) / std
    return norm_img

def segment_fingerprint(img, block_size=16, threshold=0.1):
    """
    Tách nền dựa trên phương sai (Variance).
    Vùng vân tay có phương sai cao (do đen trắng xen kẽ).
    Vùng nền có phương sai thấp (màu trắng đều).
    """
    rows, cols = img.shape
    mask = np.zeros_like(img)
    
    # Duyệt qua từng block
    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            # Cắt block
            block = img[r:min(r+block_size, rows), c:min(c+block_size, cols)]
            
            # Tính phương sai của block (đã chuẩn hóa về 0-1)
            # Chia cho 255 vì ảnh gốc là uint8 (0-255) nên phương sai sẽ rất lớn
            block_var = np.var(block / 255.0)
            
            # Nếu phương sai đủ lớn -> Là vân tay
            if block_var > threshold:
                mask[r:min(r+block_size, rows), c:min(c+block_size, cols)] = 1
                
    return mask

def process_day2():
    img_path = os.path.join(DATASET_PATH, SAMPLE_IMAGE)
    if not os.path.exists(img_path):
        print(f"Lỗi: Không tìm thấy {img_path}")
        return

    # 1. Đọc ảnh
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return

    # 2. Chuẩn hóa (Để hiển thị rõ hơn, ta scale lại về 0-255)
    norm_img = normalize_image(img)
    # Scale về 0-255 để vẽ hình (min-max scaling)
    display_norm = cv2.normalize(norm_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. Tách nền (Segmentation)
    # Ta dùng ảnh gốc để tính variance (vì nhiễu nền thường thấp trong ảnh gốc)
    mask = segment_fingerprint(img, block_size=16, threshold=0.001) # Threshold thấp vì variance của nền cực nhỏ
    
    # 4. Áp dụng Mask vào ảnh
    segmented_img = img * mask

    # 5. Hiển thị kết quả
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title("1. Ảnh Gốc")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(display_norm, cmap='gray')
    plt.title("2. Chuẩn Hóa")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(mask, cmap='gray')
    plt.title("3. Mask (Vùng Vân Tay)")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(segmented_img, cmap='gray')
    plt.title("4. Đã Tách Nền")
    plt.axis('off')

    plt.tight_layout()
    print("Đang lưu biểu đồ thành file preprocessing.png...")
    plt.savefig("preprocessing.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    process_day2()
