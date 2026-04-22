"""
Bước 8: Feature Extraction - Jain's Fingercode
=============================================================
Mục tiêu: Thay thế hoàn toàn thuật toán vét cạn Minutiae bằng thuật toán 
Fingercode của Jain nhằm trích xuất ra một vector có số chiều cố định 
(ví dụ: 320 chiều) để có thể tìm kiếm k-NN qua Inverted File Index.

Thuật toán Fingercode:
  1. Xác định Core Point (Điểm trung tâm của vân tay).
  2. Định nghĩa một vùng quan tâm (ROI) quanh Core Point 
     được chia thành nhiều cung (sectors/bands).
  3. Đưa ảnh vân tay qua tập hợp các bộ lọc Gabor với 8 hướng cố định.
  4. Tính toán phương sai (Variance) hoặc độ lệch chuẩn (StdDev) 
     tại mỗi cung cho từng ảnh đã qua bộ lọc → Nối lại thành 1 vector duy nhất.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# ============================================================================
# CẤU HÌNH
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "FVC2002", "DB1_B")
SAMPLE_IMAGE = "101_1.tif"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thông số Fingercode
NUM_BANDS = 5
NUM_SECTORS_PER_BAND = 8
BAND_WIDTH = 20
INNER_RADIUS = 20
GABOR_ANGLES = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
FINGERCODE_DIM = NUM_BANDS * NUM_SECTORS_PER_BAND * len(GABOR_ANGLES)

from importlib.util import spec_from_file_location, module_from_spec
def _import_module(name, filepath):
    spec = spec_from_file_location(name, filepath)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

step03 = _import_module("s03", os.path.join(BASE_DIR, "03_enhancement.py"))
step04 = _import_module("s04", os.path.join(BASE_DIR, "04_orientation_field.py"))
step05 = _import_module("s05", os.path.join(BASE_DIR, "05_frequency_estimation.py"))
step06 = _import_module("s06", os.path.join(BASE_DIR, "06_gabor_filter.py"))

# ============================================================================
# TÌM CORE POINT THAY THẾ
# ============================================================================
def find_core_point(orient_img, mask):
    """
    Tìm điểm Core (điểm trung tâm) dựa vào sự biến thiên góc hướng 
    (Orientation Variance). Nơi giao nhau của nhiều hướng (vùng lõi vân tay)
    thường có sự biến thiên góc hướng cao nhất.
    """
    rows, cols = orient_img.shape
    variance_map = np.zeros_like(orient_img)
    window_size = 15
    pad = window_size // 2
    
    # Tính sin và cos của góc kép để việc tính toán variance không bị lỗi ở 0-pi
    sin2 = np.sin(2 * orient_img)
    cos2 = np.cos(2 * orient_img)
    
    # Khảo sát biến thiên góc
    for r in range(pad, rows - pad):
        for c in range(pad, cols - pad):
            if mask[r, c] == 0:
                continue
            
            patch_sin = sin2[r-pad:r+pad+1, c-pad:c+pad+1]
            patch_cos = cos2[r-pad:r+pad+1, c-pad:c+pad+1]
            
            # Nếu patch chạm vào mask=0 (nền), bỏ qua để tránh nhận dạng sai vùng biên
            if np.any(mask[r-pad:r+pad+1, c-pad:c+pad+1] == 0):
                continue
                
            mean_sin = np.mean(patch_sin)
            mean_cos = np.mean(patch_cos)
            
            # Tính độ lớn của vector trung bình
            R = np.sqrt(mean_sin**2 + mean_cos**2)
            
            # Circular variance = 1 - R
            var = 1.0 - R
            variance_map[r, c] = var
            
    # Lấy tọa độ có variance lớn nhất
    max_idx = np.argmax(variance_map)
    core_r, core_c = np.unravel_index(max_idx, variance_map.shape)
    
    # Nếu không tìm thấy điểm hợp lý (ảnh lỗi/trống), chọn điểm center của mask
    if variance_map[core_r, core_c] == 0:
        ys, xs = np.where(mask > 0)
        if len(ys) > 0:
            core_r, core_c = int(np.median(ys)), int(np.median(xs))
        else:
            core_r, core_c = rows//2, cols//2
            
    return core_r, core_c


# ============================================================================
# TRÍCH XUẤT FINGERCODE (FEATURE VECTOR)
# ============================================================================
def extract_fingercode(img, core_r, core_c, median_freq, num_bands=NUM_BANDS, 
                       num_sectors=NUM_SECTORS_PER_BAND, inner_radius=INNER_RADIUS, 
                       band_width=BAND_WIDTH):
    """
    Áp dụng 8 bộ lọc Gabor theo cấu trúc Jain's Fingercode và trích 
    xuất đặc trưng phương sai từ các sector.
    """
    rows, cols = img.shape
    feature_vector = []
    
    # Tạo bản đồ sector (Tesselation Map) quanh core_r, core_c
    sector_map = np.full((rows, cols), -1, dtype=int)
    
    for r in range(rows):
        for c in range(cols):
            dy = r - core_r
            dx = c - core_c
            dist = np.sqrt(dx**2 + dy**2)
            
            # Bỏ qua vùng bán kính trong cùng
            if dist < inner_radius:
                continue
                
            # Tính xem bán kính nằm ở band nào
            band = int((dist - inner_radius) // band_width)
            if band < 0 or band >= num_bands:
                continue
                
            # Tính góc (từ 0 đến 2pi) và đổi sang sector index
            angle = np.arctan2(dy, dx)
            if angle < 0:
                angle += 2 * np.pi
                
            sector_angle = 2 * np.pi / num_sectors
            sector = int(angle // sector_angle) % num_sectors
            
            sector_index = band * num_sectors + sector
            sector_map[r, c] = sector_index

    # 8 hướng Gabor
    for angle_deg in GABOR_ANGLES:
        angle_rad = np.radians(angle_deg)
        # Sử dụng create_gabor_filter của step06
        kernel = step06.create_gabor_filter(angle_rad, median_freq, kx=0.5, ky=0.5)
        
        # Convolve ảnh với Gabor (sử dụng filter2D của cv2 cho tốc độ)
        filtered = cv2.filter2D(img.astype(np.float64), cv2.CV_64F, kernel)
        
        # Absolute Average Deviation (AAD) hoặc Variance trên từng sector
        for s in range(num_bands * num_sectors):
            # Các index thuộc sector s
            mask_sector = (sector_map == s)
            pixels = filtered[mask_sector]
            
            if len(pixels) > 0:
                mean_val = np.mean(pixels)
                aad = np.mean(np.abs(pixels - mean_val))
                var = aad  # Fingercode nguyên thủy dùng AAD (Absolute Average Deviation)
            else:
                var = 0.0
                
            feature_vector.append(var)
            
    # Normalize vector để đảm bảo matching ổn định (Cosine / L2 distance scale)
    feature_vector = np.array(feature_vector, dtype=np.float32)
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector = feature_vector / norm
        
    return feature_vector, sector_map


# ============================================================================
# ENTRY POINT CHO PIPELINE HỆ THỐNG MỚI
# ============================================================================
def extract_features(img_path):
    """
    Hàm thay thế extract_features() trả về feature_vector có độ dài cố định.
    Dùng cho: database_system và matching.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None

    # Enhancement (Tiền xử lý tăng cường giống hệt cũ)
    enhanced, mask, _ = step03.full_enhancement_pipeline(
        img, clip_limit=2.5, grid_size=(8, 8), block_size=16, var_threshold=0.005)

    # Orientation (Đóng góp cho Core Point)
    orient_img, reliability = step04.estimate_orientation(enhanced)

    # Frequency (Cần dải tần để setup Gabor Filter cố định)
    freq_img, median_freq = step05.ridge_frequency(
        enhanced, mask, orient_img, block_size=32, wind_size=5, min_wave_length=5, max_wave_length=15)
    
    if median_freq <= 0:
        median_freq = 0.1 # Tránh lỗi chia cho 0

    # 1. Tìm Core Point
    core_r, core_c = find_core_point(orient_img, mask)
    
    # 2. Sinh Fingercode
    feature_vector, _ = extract_fingercode(enhanced, core_r, core_c, median_freq)

    return feature_vector, img


def process_fingercode():
    img_path = os.path.join(DATASET_PATH, SAMPLE_IMAGE)
    if not os.path.exists(img_path):
        print(f"Lỗi: Không tìm thấy ảnh tại {img_path}")
        return

    vector, img = extract_features(img_path)
    
    print("\n" + "=" * 60)
    print("KẾT QUẢ RÚT ĐẶC TRƯNG FINGERCODE JAIN")
    print("=" * 60)
    if vector is not None:
        print(f"Thành công! Vector Fingercode có độ dài: {len(vector)} (chiều).")
        print(f"Một số giá trị đầu: {vector[:10]}")
    else:
        print("Lỗi trong quá trình trích xuất.")


if __name__ == "__main__":
    process_fingercode()
