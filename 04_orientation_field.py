"""
Bước 3: Orientation Field Estimation (Ước lượng Trường Hướng Vân)
==================================================================
Mục tiêu: Tại MỖI pixel trên ảnh vân tay, xác định GÓC NGHIÊNG của đường vân
           đi qua pixel đó. Kết quả là một "bản đồ hướng" (Orientation Map).

Tại sao cần Orientation Field?
  → Bộ lọc Gabor (bước sau) cần biết chính xác góc nghiêng tại mỗi vùng
    để "nối" các đường vân bị đứt và loại bỏ nhiễu theo đúng hướng vân.

Thuật toán (giống ridgeorient.m của tác giả MATLAB):
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 1. Tính Gradient (đạo hàm) theo 2 chiều X và Y bằng bộ lọc Sobel │
  │    → Gx (gradient ngang), Gy (gradient dọc)                       │
  │                                                                     │
  │ 2. Tính ma trận Hiệp phương sai (Covariance) của gradient:        │
  │    → Gxx = Gx²,  Gxy = Gx × Gy,  Gyy = Gy²                      │
  │                                                                     │
  │ 3. Làm mịn Covariance bằng bộ lọc Gaussian                       │
  │    → Tổng hợp thông tin gradient từ các pixel lân cận              │
  │                                                                     │
  │ 4. Tính góc hướng: θ = π/2 + atan2(sin2θ, cos2θ) / 2             │
  │    → sin2θ = Gxy / denom                                           │
  │    → cos2θ = (Gxx - Gyy) / denom                                  │
  │    → denom = √(Gxy² + (Gxx - Gyy)²)                               │
  │                                                                     │
  │ 5. Làm mịn góc hướng bằng Gaussian (smooth lần 2)                 │
  │                                                                     │
  │ 6. Tính Reliability (độ tin cậy): vùng nào có hướng rõ ràng        │
  └─────────────────────────────────────────────────────────────────────┘

Lý thuyết quan trọng - TẠI SAO GRADIENT VUÔNG GÓC VỚI ĐƯỜNG VÂN:
  - Gradient luôn chỉ hướng thay đổi màu sắc MẠNH NHẤT
  - Tại mép đường vân: pixel đen (vân) → pixel trắng (thung lũng)
  - Sự thay đổi này xảy ra VUÔNG GÓC với đường vân
  - Do đó: hướng vân = hướng gradient xoay thêm 90° (π/2)
  - Công thức: θ_vân = θ_gradient + π/2
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import os

# ============================================================================
# CẤU HÌNH
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "FVC2002", "DB1_B")
SAMPLE_IMAGE = "101_1.tif"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# IMPORT HÀM TỪ CÁC BƯỚC TRƯỚC
# ============================================================================
from importlib.util import spec_from_file_location, module_from_spec

def _import_module(name, filepath):
    """Import module từ file path cụ thể."""
    spec = spec_from_file_location(name, filepath)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Import các hàm đã viết
step02 = _import_module("step02", os.path.join(BASE_DIR, "02_preprocessing.py"))
step03_enh = _import_module("step03_enh", os.path.join(BASE_DIR, "03_enhancement.py"))

normalize_image = step02.normalize_image
segment_fingerprint = step02.segment_fingerprint
full_enhancement_pipeline = step03_enh.full_enhancement_pipeline


# ============================================================================
# BƯỚC 3A: TÍNH GRADIENT BẰNG SOBEL
# ============================================================================
def compute_gradient(img, ksize=3):
    """
    Tính Gradient (đạo hàm) của ảnh theo 2 chiều X và Y.
    
    Sử dụng bộ lọc Sobel - tương đương với gradient of Gaussian trong MATLAB.
    
    Bộ lọc Sobel 3×3:
        Gx (ngang):          Gy (dọc):
        [-1  0  +1]          [-1  -2  -1]
        [-2  0  +2]          [ 0   0   0]
        [-1  0  +1]          [+1  +2  +1]
    
    Nguyên lý:
      - Sobel quét qua ảnh, tại mỗi pixel nó tính:
        * Gx = sự thay đổi pixel theo chiều NGANG (trái → phải)
        * Gy = sự thay đổi pixel theo chiều DỌC (trên → dưới)
      - Tại BIÊN đường vân (nơi pixel thay đổi đột ngột từ đen sang trắng):
        → Gx hoặc Gy sẽ có giá trị LỚN
      - Tại GIỮA đường vân hoặc GIỮA thung lũng (vùng pixel đồng đều):
        → Gx và Gy ≈ 0 (không có sự thay đổi)
    
    Tham số:
      img:   Ảnh grayscale (nên đã normalize/enhance)
      ksize: Kích thước kernel Sobel (3, 5, 7...). Số lớn hơn = mịn hơn nhưng mất chi tiết.
    
    Returns:
      Gx: Gradient theo X (float64)
      Gy: Gradient theo Y (float64)
    """
    # Chuyển sang float để tránh overflow (uint8 chỉ chứa 0-255)
    img_float = img.astype(np.float64)

    # cv2.Sobel: Tương đương filter2(fx, im) và filter2(fy, im) trong MATLAB
    Gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=ksize)  # Đạo hàm theo X
    Gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=ksize)  # Đạo hàm theo Y

    return Gx, Gy


# ============================================================================
# BƯỚC 3B: TÍNH ORIENTATION FIELD
# ============================================================================
def estimate_orientation(img, gradient_sigma=1.0, block_sigma=3.0,
                         orient_smooth_sigma=3.0, sobel_ksize=3):
    """
    Ước lượng Trường Hướng Vân (Orientation Field).
    
    Đây là phiên bản Python tương đương với ridgeorient.m trong MATLAB.
    Tham số mặc định: gradientsigma=1, blocksigma=3, orientsmoothsigma=3
    (giống tác giả gốc gọi: ridgeorient(normim, 1, 3, 3))
    
    Thuật toán chi tiết:
    
    PHẦN 1 - GRADIENT:
      Tính đạo hàm Gx, Gy bằng Sobel (thay cho Gaussian gradient trong MATLAB)
    
    PHẦN 2 - COVARIANCE (Ma trận hiệp phương sai):
      Gxx = Gx²    → "Năng lượng" gradient theo X
      Gxy = Gx×Gy  → "Tương quan" giữa gradient X và Y  
      Gyy = Gy²    → "Năng lượng" gradient theo Y
      
      Tại sao cần bình phương?
      → Gradient có thể dương (+) hoặc âm (-) tùy vào chiều đen→trắng hay trắng→đen.
        Nếu cộng trực tiếp, gradient (+) và (-) triệt tiêu nhau → kết quả = 0.
        Bình phương sẽ luôn dương → giữ được thông tin hướng.
      
      Tại sao cần Gxy?
      → Gxy cho biết gradient nghiêng theo hướng nào. 
        Nếu Gxy > 0: gradient nghiêng về phía 45° (↗)
        Nếu Gxy < 0: gradient nghiêng về phía 135° (↘)
    
    PHẦN 3 - LÀM MỊN COVARIANCE:
      Dùng Gaussian blur trên Gxx, Gxy, Gyy
      → Tích hợp thông tin gradient từ VÙNG LÂN CẬN (không chỉ 1 pixel)
      → Cho hướng ổn định hơn, giảm nhiễu
    
    PHẦN 4 - TÍNH GÓC:
      sin(2θ) = Gxy / √(Gxy² + (Gxx-Gyy)²)
      cos(2θ) = (Gxx-Gyy) / √(Gxy² + (Gxx-Gyy)²)
      θ = π/2 + atan2(sin2θ, cos2θ) / 2
      
      Tại sao chia 2? → Vì ta tính trên góc gấp đôi (doubled angle) để loại bỏ
        sự mơ hồ 180° (vân hướng 0° và 180° là CÙNG HỆ THỐNG vân)
      
      Tại sao cộng π/2? → Vì gradient VUÔNG GÓC với đường vân, cộng 90° để
        chuyển từ hướng gradient → hướng vân
    
    PHẦN 5 - LÀM MỊN GÓC:
      Làm mịn sin2θ và cos2θ (KHÔNG mịn θ trực tiếp vì θ có điểm nhảy 0↔π)
      Rồi tính lại θ từ sin2θ, cos2θ đã mịn
    
    PHẦN 6 - ĐỘ TIN CẬY (Reliability):
      reliability = 1 - Imin/Imax
      → Nếu Imin ≈ Imax: gradient đều mọi hướng → KHÔNG CÓ hướng rõ ràng (nền, core)
      → Nếu Imin << Imax: gradient tập trung 1 hướng → hướng vân RÕ RÀNG
    
    Returns:
      orient_img:  Bản đồ hướng vân (radian, 0 đến π)
      reliability: Bản đồ độ tin cậy (0 đến 1)
    """
    img_float = img.astype(np.float64)

    # --- PHẦN 1: Tính Gradient ---
    Gx, Gy = compute_gradient(img, ksize=sobel_ksize)

    # --- PHẦN 2: Tính Covariance ---
    Gxx = Gx ** 2        # Năng lượng gradient theo X
    Gxy = Gx * Gy        # Tương quan gradient X-Y
    Gyy = Gy ** 2        # Năng lượng gradient theo Y

    # --- PHẦN 3: Làm mịn Covariance bằng Gaussian ---
    # Tương đương: f = fspecial('gaussian', sze, blocksigma); Gxx = filter2(f, Gxx)
    sze = int(np.fix(6 * block_sigma))
    if sze % 2 == 0:
        sze += 1  # Đảm bảo kích thước kernel là SỐ LẺ

    Gxx = cv2.GaussianBlur(Gxx, (sze, sze), block_sigma)
    Gxy = 2 * cv2.GaussianBlur(Gxy, (sze, sze), block_sigma)  # Nhân 2 giống MATLAB
    Gyy = cv2.GaussianBlur(Gyy, (sze, sze), block_sigma)

    # --- PHẦN 4: Tính góc hướng ---
    # Mẫu số (denominator) - tránh chia cho 0 bằng eps
    denom = np.sqrt(Gxy ** 2 + (Gxx - Gyy) ** 2) + np.finfo(float).eps

    sin2theta = Gxy / denom       # sin(2θ)
    cos2theta = (Gxx - Gyy) / denom  # cos(2θ)

    # --- PHẦN 5: Làm mịn góc ---
    # Làm mịn trên sin2θ và cos2θ (KHÔNG mịn θ trực tiếp!)
    sze = int(np.fix(6 * orient_smooth_sigma))
    if sze % 2 == 0:
        sze += 1

    sin2theta = cv2.GaussianBlur(sin2theta, (sze, sze), orient_smooth_sigma)
    cos2theta = cv2.GaussianBlur(cos2theta, (sze, sze), orient_smooth_sigma)

    # Công thức cuối: θ = π/2 + atan2(sin2θ, cos2θ) / 2
    orient_img = np.pi / 2 + np.arctan2(sin2theta, cos2theta) / 2

    # --- PHẦN 6: Tính Reliability ---
    # Imin: moment quán tính nhỏ nhất (dọc theo hướng vân)
    # Imax: moment quán tính lớn nhất (vuông góc với hướng vân)
    Imin = (Gyy + Gxx) / 2 - (Gxx - Gyy) * cos2theta / 2 - Gxy * sin2theta / 2
    Imax = Gyy + Gxx - Imin

    reliability = 1 - Imin / (Imax + 0.001)

    # Đánh dấu vùng có denom quá nhỏ là không tin cậy
    reliability = reliability * (denom > 0.001)

    return orient_img, reliability


# ============================================================================
# HÀM TRỰC QUAN HÓA ORIENTATION FIELD
# ============================================================================
def visualize_orientation_field(img, orient_img, mask, block_size=16,
                                 scale=0.8, reliability=None, rel_threshold=0.3):
    """
    Vẽ Trường Hướng Vân dưới dạng các đường ngắn (line segments) trên ảnh.
    
    Mỗi block sẽ được biểu diễn bằng 1 đường thẳng ngắn cho thấy
    hướng của đường vân tại vùng đó.
    
    Tham số:
      block_size:    Kích thước mỗi block (pixel)
      scale:         Độ dài đường vẽ (tỷ lệ so với block_size)
      reliability:   Bản đồ reliability (nếu có, chỉ vẽ vùng tin cậy)
      rel_threshold: Ngưỡng reliability tối thiểu để vẽ
    """
    rows, cols = img.shape
    line_len = block_size * scale / 2

    # Tạo ảnh nền màu từ ảnh grayscale
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for r in range(block_size // 2, rows - block_size // 2, block_size):
        for c in range(block_size // 2, cols - block_size // 2, block_size):
            # Bỏ qua vùng nền
            if mask[r, c] == 0:
                continue

            # Bỏ qua vùng có reliability thấp
            if reliability is not None and reliability[r, c] < rel_threshold:
                continue

            # Lấy góc hướng tại block này
            angle = orient_img[r, c]

            # Tính 2 đầu mút của đường thẳng biểu diễn hướng
            dx = line_len * np.cos(angle)
            dy = line_len * np.sin(angle)

            x1, y1 = int(c - dx), int(r - dy)
            x2, y2 = int(c + dx), int(r + dy)

            # Màu dựa trên hướng: 0°=đỏ, 45°=xanh lá, 90°=xanh dương, 135°=tím
            hue = angle / np.pi  # 0 → 1
            rgb = hsv_to_rgb([hue, 1.0, 1.0])
            color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # BGR

            cv2.line(vis_img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    return vis_img


def create_orientation_colormap(orient_img, mask):
    """
    Tạo bản đồ màu (colormap) của Orientation Field.
    
    Mỗi hướng được gán một màu khác nhau theo bánh xe màu HSV:
      - 0° (ngang)      → Đỏ
      - 45° (chéo ↗)    → Xanh lá
      - 90° (dọc)       → Xanh dương  
      - 135° (chéo ↘)   → Tím/Hồng
    """
    rows, cols = orient_img.shape

    # Chuẩn hóa góc về 0-1 cho kênh Hue
    hue = (orient_img / np.pi * 180).astype(np.float32)  # 0-180 độ
    hue = hue / 180.0  # Chuẩn hóa về 0-1

    # Tạo ảnh HSV
    hsv = np.zeros((rows, cols, 3), dtype=np.float32)
    hsv[:, :, 0] = hue                     # Hue = hướng
    hsv[:, :, 1] = 1.0                     # Saturation = 1 (màu đậm)
    hsv[:, :, 2] = mask.astype(np.float32) # Value = mask (tối = nền)

    # Chuyển HSV → RGB
    color_map = np.zeros((rows, cols, 3), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] > 0:
                color_map[r, c] = hsv_to_rgb([hsv[r, c, 0], hsv[r, c, 1], hsv[r, c, 2]])

    return color_map


# ============================================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================================
def process_orientation():
    img_path = os.path.join(DATASET_PATH, SAMPLE_IMAGE)
    if not os.path.exists(img_path):
        print(f"Lỗi: Không tìm thấy ảnh tại {img_path}")
        return

    # === Đọc ảnh ===
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Lỗi: Không đọc được ảnh.")
        return

    print(f"Đã đọc ảnh: {SAMPLE_IMAGE} | Kích thước: {img.shape}")

    # === Bước 1+2: Enhancement (từ bước trước) ===
    print("Đang chạy Enhancement pipeline...")
    enhanced_img, mask, _ = full_enhancement_pipeline(
        img, clip_limit=2.5, grid_size=(8, 8),
        block_size=16, var_threshold=0.005
    )

    # === Bước 3: Tính Gradient ===
    print("Đang tính Gradient (Sobel)...")
    Gx, Gy = compute_gradient(enhanced_img, ksize=3)

    # Tính magnitude (cường độ gradient) để minh họa
    gradient_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    grad_display = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # === Bước 3: Tính Orientation Field ===
    print("Đang tính Orientation Field...")
    orient_img, reliability = estimate_orientation(
        enhanced_img,
        gradient_sigma=1.0,      # Sigma cho gradient (giống MATLAB: 1)
        block_sigma=3.0,         # Sigma cho smoothing covariance (giống MATLAB: 3)
        orient_smooth_sigma=3.0, # Sigma cho smoothing orientation (giống MATLAB: 3)
        sobel_ksize=3
    )

    # === Trực quan hóa ===
    print("Đang vẽ kết quả...")

    # Tạo line visualization
    orient_lines = visualize_orientation_field(
        img, orient_img, mask, block_size=14, scale=0.9,
        reliability=reliability, rel_threshold=0.3
    )

    # Tạo color map
    orient_colormap = create_orientation_colormap(orient_img, mask)

    # =========================================================================
    # HÌNH 1: Pipeline tổng quan - Gradient → Orientation
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Bước 3: Orientation Field Estimation", fontsize=16, fontweight='bold')

    # Row 1: Input → Gradient
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("1. Ảnh Gốc")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(enhanced_img, cmap='gray')
    axes[0, 1].set_title("2. Sau Enhancement\n(CLAHE + Segment)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(grad_display, cmap='hot')
    axes[0, 2].set_title("3. Gradient Magnitude\n(Sobel: |Gx| + |Gy|)")
    axes[0, 2].axis('off')

    # Row 2: Orientation Results
    axes[1, 0].imshow(orient_colormap)
    axes[1, 0].set_title("4. Orientation Colormap\n(Màu = Hướng vân)")
    axes[1, 0].axis('off')

    # Convert BGR → RGB for matplotlib
    axes[1, 1].imshow(cv2.cvtColor(orient_lines, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("5. Orientation Lines\n(Đường = Hướng vân tại block)")
    axes[1, 1].axis('off')

    # Reliability map
    rel_display = np.where(mask == 1, reliability, 0)
    axes[1, 2].imshow(rel_display, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 2].set_title("6. Reliability Map\n(Xanh=tin cậy, Đỏ=không)")
    axes[1, 2].axis('off')

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "04_orientation_overview.png")
    plt.savefig(path1, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {path1}")

    # =========================================================================
    # HÌNH 2: Chi tiết Gradient Gx và Gy
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle("Chi tiết: Gradient theo X và Y (Sobel Filter)", fontsize=14, fontweight='bold')

    # Cắt vùng trung tâm để zoom
    h, w = img.shape
    ys, ye = h // 4, 3 * h // 4
    xs, xe = w // 4, 3 * w // 4

    axes2[0].imshow(Gx[ys:ye, xs:xe], cmap='RdBu_r', vmin=-100, vmax=100)
    axes2[0].set_title("Gx (Gradient ngang)\nĐỏ=sáng→tối, Xanh=tối→sáng")
    axes2[0].axis('off')

    axes2[1].imshow(Gy[ys:ye, xs:xe], cmap='RdBu_r', vmin=-100, vmax=100)
    axes2[1].set_title("Gy (Gradient dọc)\nĐỏ=sáng→tối, Xanh=tối→sáng")
    axes2[1].axis('off')

    axes2[2].imshow(enhanced_img[ys:ye, xs:xe], cmap='gray')
    axes2[2].set_title("Ảnh gốc (Zoom)")
    axes2[2].axis('off')

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "04_orientation_gradient_detail.png")
    plt.savefig(path2, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {path2}")

    # =========================================================================
    # HÌNH 3: Orientation Lines zoom vào vùng trung tâm
    # =========================================================================
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle("Zoom: Orientation Lines trên ảnh vân tay", fontsize=14, fontweight='bold')

    # Zoom orientation lines
    orient_lines_zoom = orient_lines[ys:ye, xs:xe]
    axes3[0].imshow(cv2.cvtColor(orient_lines_zoom, cv2.COLOR_BGR2RGB))
    axes3[0].set_title("Orientation Lines (Zoom)")
    axes3[0].axis('off')

    # Orientation color map zoom
    axes3[1].imshow(orient_colormap[ys:ye, xs:xe])
    axes3[1].set_title("Orientation Colormap (Zoom)")
    axes3[1].axis('off')

    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "04_orientation_zoom.png")
    plt.savefig(path3, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {path3}")

    # =========================================================================
    # TÓM TẮT
    # =========================================================================
    print("\n" + "=" * 60)
    print("TÓM TẮT BƯỚC 3: ORIENTATION FIELD ESTIMATION")
    print("=" * 60)
    print(f"  Phương pháp       : Sobel Gradient → Covariance → atan2")
    print(f"  Tham số Sobel     : ksize=3")
    print(f"  Block sigma       : 3.0 (smooth covariance)")
    print(f"  Orient smooth     : 3.0 (smooth final orientation)")
    print(f"  Dải góc           : {np.min(orient_img[mask==1]):.2f} → {np.max(orient_img[mask==1]):.2f} rad")
    print(f"                      ({np.degrees(np.min(orient_img[mask==1])):.1f}° → {np.degrees(np.max(orient_img[mask==1])):.1f}°)")
    avg_rel = np.mean(reliability[mask == 1])
    print(f"  Reliability TB    : {avg_rel:.3f} (1.0 = hoàn hảo)")
    high_rel = np.mean(reliability[mask == 1] > 0.5) * 100
    print(f"  Vùng tin cậy cao  : {high_rel:.1f}% (reliability > 0.5)")
    print("=" * 60)
    print(f"\n  Kết quả đã lưu tại: {OUTPUT_DIR}/")
    print("  → 04_orientation_overview.png           (Tổng quan pipeline)")
    print("  → 04_orientation_gradient_detail.png    (Chi tiết Gradient Gx, Gy)")
    print("  → 04_orientation_zoom.png               (Zoom orientation lines)")
    print("\n  Bước tiếp theo: 05_frequency_estimation.py")
    print("  (Ước lượng tần số đường vân - Ridge Frequency)")


if __name__ == "__main__":
    process_orientation()
