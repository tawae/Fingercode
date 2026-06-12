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
import os

# ============================================================================
# CẤU HÌNH
# ============================================================================
import config

BASE_DIR = config.BASE_DIR

# Thông số Fingercode
NUM_BANDS = 5
NUM_SECTORS_PER_BAND = 8
BAND_WIDTH = 20
INNER_RADIUS = 20
GABOR_ANGLES = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
FINGERCODE_DIM = NUM_BANDS * NUM_SECTORS_PER_BAND * len(GABOR_ANGLES)

from importlib.util import spec_from_file_location, module_from_spec
def _import_module(name, filepath):
    """
    Mục đích:
      Import module pipeline từ file path.

    Tham số:
      name: Tên module tạm.
      filepath: Đường dẫn tới file `.py`.

    Vì sao chọn tham số này:
      Dự án giữ tên file dạng đánh số nên cần import động thay vì import chuẩn.

    Đầu ra:
      Module object đã nạp.

    Vì sao đầu ra như vậy mà không trả từng hàm:
      `08` cần nhiều dependency từ `03-06`; trả module giúp code rõ nguồn gốc
      từng hàm (`step03`, `step04`, ...).
    """
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
    Mục đích:
      Tìm core point, tức tâm tương đối của vùng vân tay, dựa trên biến thiên
      orientation cục bộ.

    Tham số:
      orient_img: Bản đồ hướng radian từ `estimate_orientation`.
      mask: Mask vùng vân tay, dùng để bỏ nền và vùng biên.

    Vì sao chọn tham số này:
      Fingercode cần một tâm để chia ROI thành các band/sector. Orientation
      variance thường cao quanh vùng lõi, còn mask tránh chọn nhầm nền hoặc mép
      ảnh nơi hướng không ổn định.

    Đầu ra:
      Tuple `(core_r, core_c)` là tọa độ hàng/cột của core point.

    Vì sao đầu ra như vậy mà không trả cả variance map:
      Pipeline chính chỉ cần tọa độ tâm để trích vector. Variance map là dữ liệu
      trung gian lớn, không cần lưu nếu không vẽ debug.
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
    Mục đích:
      Trích vector Fingercode bằng cách chia ROI quanh core thành các sector,
      lọc ảnh bằng 8 hướng Gabor và lấy AAD trên từng sector.

    Tham số:
      img: Ảnh grayscale đã enhancement.
      core_r: Tọa độ hàng của core point.
      core_c: Tọa độ cột của core point.
      median_freq: Tần số ridge đại diện để tạo Gabor kernels.
      num_bands: Số vòng đồng tâm quanh core.
      num_sectors: Số sector trên mỗi vòng.
      inner_radius: Bán kính bỏ qua vùng lõi trong cùng.
      band_width: Độ rộng mỗi band.

    Vì sao chọn tham số này:
      Cấu hình mặc định `5 bands * 8 sectors * 8 hướng Gabor` tạo vector 320
      chiều, đủ cố định cho FAISS. `inner_radius` tránh vùng core quá cong và
      nhiễu; `band_width=20` giữ mỗi sector có đủ pixel để thống kê ổn định.

    Đầu ra:
      Tuple `(feature_vector, sector_map)`.

    Vì sao đầu ra như vậy mà không trả danh sách feature thô:
      `feature_vector` được chuẩn hóa L2 để so khoảng cách ổn định trong FAISS.
      `sector_map` được trả kèm để có thể debug/visualize cách chia ROI nếu cần.
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
    Mục đích:
      Entry point trích đặc trưng cho toàn hệ thống: đọc ảnh, enhancement,
      orientation, frequency, tìm core và tạo Fingercode.

    Tham số:
      img_path: Đường dẫn ảnh vân tay cần xử lý.

    Vì sao chọn tham số này:
      DB, GUI và evaluation đều bắt đầu từ file ảnh trên đĩa, nên nhận path giúp
      các tầng gọi không phải lặp lại logic đọc ảnh và xử lý lỗi định dạng.

    Đầu ra:
      Tuple `(feature_vector, img)`; nếu đọc ảnh lỗi thì trả `(None, None)`.

    Vì sao đầu ra như vậy mà không chỉ trả vector:
      Vector dùng cho matching/FAISS, còn ảnh gốc dùng để hiển thị query trong
      GUI và báo cáo demo. Trả `None` thay exception giúp batch enrollment và
      evaluation bỏ qua ảnh lỗi an toàn.
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
