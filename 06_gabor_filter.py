"""
Bước 5: Gabor Filtering (Lọc Gabor 2D)
========================================
Mục tiêu: Dùng bộ lọc Gabor có HƯỚNG và TẦN SỐ phù hợp để tăng cường
           đường vân lần cuối, chuẩn bị cho bước Nhị phân hóa và Làm mảnh.

Gabor Filter là gì?
  → Là tích của 1 hàm Gaussian (phần bao - envelope) × 1 sóng cos (phần sóng).
  → Gaussian giới hạn phạm vi tác động (cục bộ).
  → Sóng cos chọn lọc theo tần số và hướng cụ thể.
  → Kết quả: chỉ GIỮ LẠI đường vân khớp hướng + tần số, LOẠI BỎ nhiễu.

Công thức Gabor 2D:
  g(x, y, θ, f) = exp(-(x'²/σx² + y'²/σy²) / 2) × cos(2π × f × x')

  Trong đó:
    x' =  x×cos(θ) + y×sin(θ)   (tọa độ xoay theo hướng θ)
    y' = -x×sin(θ) + y×cos(θ)
    f  = tần số đường vân (từ Bước 4)
    θ  = hướng đường vân (từ Bước 3)
    σx = kx / f   (σ dọc theo hướng vân - kiểm soát bandwidth)
    σy = ky / f   (σ vuông góc hướng vân - kiểm soát độ chọn lọc hướng)

Tương đương: ridgefilter.m trong MATLAB (Peter Kovesi, 2005)

Tham khảo:
  Hong, L., Wan, Y., and Jain, A. K. "Fingerprint image enhancement:
  Algorithm and performance evaluation." IEEE TPAMI 20(8), 1998.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as scipy_rotate
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
    spec = spec_from_file_location(name, filepath)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

step03_enh = _import_module("step03_enh", os.path.join(BASE_DIR, "03_enhancement.py"))
step04_ori = _import_module("step04_ori", os.path.join(BASE_DIR, "04_orientation_field.py"))
step05_freq = _import_module("step05_freq", os.path.join(BASE_DIR, "05_frequency_estimation.py"))

full_enhancement_pipeline = step03_enh.full_enhancement_pipeline
estimate_orientation = step04_ori.estimate_orientation
ridge_frequency = step05_freq.ridge_frequency


# ============================================================================
# BƯỚC 5A: TẠO GABOR FILTER
# ============================================================================
def create_gabor_filter(angle, frequency, kx=0.5, ky=0.5):
    """
    Tạo 1 bộ lọc Gabor 2D cho 1 hướng và 1 tần số cụ thể.
    Tương đương: phần tạo reffilter trong ridgefilter.m

    Công thức:
      g(x,y) = exp(-(x'²/σx² + y'²/σy²)/2) × cos(2π × f × x')

    Tham số:
      angle:     Hướng vân tại vị trí cần lọc (radian)
      frequency: Tần số vân tại vị trí cần lọc (1/pixel)
      kx:        Hệ số scale cho σx (dọc theo vân). Mặc định 0.5 (giống MATLAB)
      ky:        Hệ số scale cho σy (vuông góc vân). Mặc định 0.5

    Giải thích kx, ky:
      σx = kx / frequency,  σy = ky / frequency
      → σ tỷ lệ nghịch với frequency nên bộ lọc tự điều chỉnh kích thước
        theo bước sóng: bước sóng lớn → filter lớn, bước sóng nhỏ → filter nhỏ
      → kx lớn → filter dài theo hướng vân → bandwidth hẹp → chọn lọc tần số tốt hơn
      → ky lớn → filter rộng vuông góc → chọn lọc hướng kém hơn (chấp nhận nhiều hướng)

    Returns:
      gabor_filter: Ma trận 2D chứa bộ lọc Gabor
    """
    if frequency <= 0:
        return None

    # Tính σ (sigma) - phạm vi tác động của Gaussian envelope
    sigma_x = 1.0 / frequency * kx
    sigma_y = 1.0 / frequency * ky

    # Kích thước filter = 3σ mỗi bên (giống MATLAB: sze = round(3*max(sigmax,sigmay)))
    sze = int(np.round(3 * max(sigma_x, sigma_y)))
    if sze < 1:
        sze = 1

    # Tạo lưới tọa độ
    x, y = np.meshgrid(np.arange(-sze, sze + 1), np.arange(-sze, sze + 1))

    # Xoay tọa độ theo hướng vân (angle)
    # +90° vì orientation image cho hướng DỌC THEO vân,
    # nhưng Gabor cần hướng VUÔNG GÓC với vân để lọc
    x_theta = x * np.cos(angle) + y * np.sin(angle)
    y_theta = -x * np.sin(angle) + y * np.cos(angle)

    # Công thức Gabor:
    # Gaussian envelope × Cosine wave
    gaussian = np.exp(-0.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))
    cosine = np.cos(2 * np.pi * frequency * x_theta)
    gabor_filter = gaussian * cosine

    return gabor_filter


# ============================================================================
# BƯỚC 5B: ÁP DỤNG GABOR FILTER LÊN TOÀN ẢNH
# ============================================================================
def gabor_ridge_filter(img, orient_img, freq_img, mask, kx=0.5, ky=0.5,
                        angle_inc=3):
    """
    Lọc ảnh vân tay bằng Gabor Filter theo hướng và tần số cục bộ.
    Tương đương: ridgefilter.m trong MATLAB.

    Chiến lược tối ưu (giống MATLAB):
      Thay vì tạo 1 filter riêng cho MỖI pixel (quá chậm), ta:
      1. Làm tròn tần số → nhóm thành vài giá trị tần số khác nhau
      2. Chia hướng thành các bước angle_inc=3° → 180/3 = 60 hướng
      3. Tạo sẵn bảng filter cho mỗi cặp (tần số, hướng)
      4. Với mỗi pixel, tra bảng lấy filter phù hợp nhất → áp dụng

    Tham số:
      img:        Ảnh đã enhanced (grayscale, float)
      orient_img: Bản đồ hướng vân (radian)
      freq_img:   Bản đồ tần số vân
      mask:       Mask vùng vân tay
      kx, ky:     Hệ số scale cho Gabor (mặc định 0.5)
      angle_inc:  Bước nhảy góc (độ). 3° → 60 filters mỗi tần số

    Returns:
      filtered_img: Ảnh sau khi lọc Gabor
    """
    img = img.astype(np.float64)
    rows, cols = img.shape
    filtered_img = np.zeros((rows, cols), dtype=np.float64)

    # --- Bước 1: Tìm các pixel hợp lệ (có tần số > 0) ---
    valid_r, valid_c = np.where(freq_img > 0)
    if len(valid_r) == 0:
        print("  Cảnh báo: Không có vùng nào có tần số hợp lệ!")
        return filtered_img

    # --- Bước 2: Làm tròn tần số ---
    freq_rounded = np.round(freq_img * 100) / 100
    valid_freqs = freq_rounded[freq_rounded > 0]
    unique_freqs = np.unique(valid_freqs)

    print(f"  Số tần số khác nhau: {len(unique_freqs)}")

    # --- Bước 3: Tạo bảng filter cho tất cả cặp (tần số, hướng) ---
    num_orientations = int(180 / angle_inc)
    filter_bank = {}
    filter_sizes = {}

    for freq in unique_freqs:
        # Tạo filter gốc (hướng 0°)
        sigma_x = kx / freq
        sigma_y = ky / freq
        sze = int(np.round(3 * max(sigma_x, sigma_y)))
        if sze < 1:
            sze = 1
        filter_sizes[freq] = sze

        x, y = np.meshgrid(np.arange(-sze, sze + 1), np.arange(-sze, sze + 1))
        # Filter gốc (reference filter) - hướng 0°
        ref_filter = np.exp(-0.5 * (x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y ** 2)) \
                     * np.cos(2 * np.pi * freq * x)

        # Tạo các phiên bản xoay
        for o_idx in range(num_orientations):
            angle_deg = o_idx * angle_inc + 90  # +90 vì orient theo hướng vân
            rotated_filter = scipy_rotate(ref_filter, -angle_deg, reshape=False,
                                          order=1, mode='nearest')
            filter_bank[(freq, o_idx)] = rotated_filter

    print(f"  Tổng filter đã tạo: {len(filter_bank)}")

    # --- Bước 4: Tìm kích thước filter lớn nhất (để tránh tràn biên) ---
    max_sze = max(filter_sizes.values()) if filter_sizes else 0

    # --- Bước 5: Áp dụng filter cho từng pixel hợp lệ ---
    # Chỉ xử lý pixel cách biên ít nhất max_sze
    max_orient_idx = num_orientations

    # Chuyển orientation → index
    orient_index = np.round(orient_img / np.pi * 180 / angle_inc).astype(int)
    orient_index = np.clip(orient_index, 1, max_orient_idx)
    orient_index[orient_index < 1] += max_orient_idx
    orient_index[orient_index > max_orient_idx] -= max_orient_idx
    orient_index -= 1  # Chuyển sang 0-indexed

    total = len(valid_r)
    processed = 0

    for idx in range(total):
        r = valid_r[idx]
        c = valid_c[idx]

        freq_val = freq_rounded[r, c]
        if freq_val <= 0:
            continue

        s = filter_sizes.get(freq_val, 0)
        if s == 0:
            continue

        # Kiểm tra biên
        if r < s or r >= rows - s or c < s or c >= cols - s:
            continue

        # Lấy filter từ bảng
        o_idx = orient_index[r, c]
        filt = filter_bank.get((freq_val, o_idx))
        if filt is None:
            continue

        # Áp dụng: Tích chập cục bộ (convolution tại 1 điểm)
        # Lấy vùng ảnh cùng kích thước với filter → nhân → cộng tất cả
        img_patch = img[r - s:r + s + 1, c - s:c + s + 1]

        # Đảm bảo kích thước khớp
        if img_patch.shape == filt.shape:
            filtered_img[r, c] = np.sum(img_patch * filt)
            processed += 1

    print(f"  Pixel đã lọc: {processed}/{total} ({processed/total*100:.1f}%)")

    return filtered_img


# ============================================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================================
def process_gabor():
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

    # === Bước 1+2: Enhancement ===
    print("[1/4] Enhancement pipeline...")
    enhanced_img, mask, _ = full_enhancement_pipeline(
        img, clip_limit=2.5, grid_size=(8, 8),
        block_size=16, var_threshold=0.005
    )

    # === Bước 3: Orientation ===
    print("[2/4] Orientation Field...")
    orient_img, reliability = estimate_orientation(enhanced_img)

    # === Bước 4: Frequency ===
    print("[3/4] Frequency Estimation...")
    freq_img, median_freq = ridge_frequency(
        enhanced_img, mask, orient_img,
        block_size=32, wind_size=5,
        min_wave_length=5, max_wave_length=15
    )
    print(f"  Tần số trung vị: {median_freq:.4f} (bước sóng ≈ {1/median_freq:.1f}px)")

    # Dùng tần số trung vị cho toàn ảnh (giống khuyến nghị tác giả MATLAB)
    # Thay thế freq_img bằng median_freq tại vùng có mask
    freq_img_median = np.where(mask == 1, median_freq, 0).astype(np.float64)

    # === Bước 5: Gabor Filtering ===
    print("[4/4] Gabor Filtering...")
    gabor_result = gabor_ridge_filter(
        enhanced_img, orient_img, freq_img_median, mask,
        kx=0.5, ky=0.5, angle_inc=3
    )

    # Nhị phân hóa kết quả Gabor (> 0 = vân, <= 0 = nền)
    gabor_binary = (gabor_result > 0).astype(np.uint8) * 255

    # Áp mask
    gabor_binary_masked = np.where(mask == 1, gabor_binary, 255).astype(np.uint8)

    # Chuẩn hóa gabor result để hiển thị
    gabor_display = cv2.normalize(gabor_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gabor_display = np.where(mask == 1, gabor_display, 255).astype(np.uint8)

    # =========================================================================
    # HÌNH 1: Tổng quan Pipeline đến Gabor
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Bước 5: Gabor Filtering", fontsize=16, fontweight='bold')

    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("1. Ảnh Gốc")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(enhanced_img, cmap='gray')
    axes[0, 1].set_title("2. Enhanced (CLAHE)")
    axes[0, 1].axis('off')

    # Orientation overlay
    from matplotlib.colors import hsv_to_rgb
    orient_color = np.zeros((*orient_img.shape, 3))
    for r in range(orient_img.shape[0]):
        for c in range(orient_img.shape[1]):
            if mask[r, c] > 0:
                hue = orient_img[r, c] / np.pi
                orient_color[r, c] = hsv_to_rgb([hue, 1.0, 1.0])
    axes[0, 2].imshow(orient_color)
    axes[0, 2].set_title("3. Orientation Map")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(gabor_display, cmap='gray')
    axes[1, 0].set_title("4. Gabor Result\n(Continuous)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(gabor_binary_masked, cmap='gray')
    axes[1, 1].set_title("5. Gabor Binary\n(Đen=Vân, Trắng=Nền)")
    axes[1, 1].axis('off')

    # So sánh zoom
    h, w = img.shape
    ys, ye = h // 4, 3 * h // 4
    xs, xe = w // 4, 3 * w // 4
    axes[1, 2].imshow(gabor_binary_masked[ys:ye, xs:xe], cmap='gray')
    axes[1, 2].set_title("6. Gabor Binary (Zoom)")
    axes[1, 2].axis('off')

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "06_gabor_overview.png")
    plt.savefig(path1, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {path1}")

    # =========================================================================
    # HÌNH 2: Minh họa Gabor Filter kernels ở các hướng khác nhau
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    fig2.suptitle(f"Gabor Filter Kernels (freq={median_freq:.3f}, wavelength={1/median_freq:.1f}px)",
                  fontsize=14, fontweight='bold')

    angles_deg = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
    for i, angle_deg in enumerate(angles_deg):
        r_idx = i // 4
        c_idx = i % 4
        angle_rad = np.radians(angle_deg)
        kernel = create_gabor_filter(angle_rad, median_freq, kx=0.5, ky=0.5)
        if kernel is not None:
            axes2[r_idx, c_idx].imshow(kernel, cmap='RdBu_r', interpolation='nearest')
            axes2[r_idx, c_idx].set_title(f"{angle_deg}°")
        axes2[r_idx, c_idx].axis('off')

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "06_gabor_kernels.png")
    plt.savefig(path2, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {path2}")

    # =========================================================================
    # HÌNH 3: So sánh chi tiết Trước vs Sau Gabor
    # =========================================================================
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle("So sánh chi tiết: Trước vs Sau Gabor Filtering",
                  fontsize=14, fontweight='bold')

    axes3[0].imshow(img[ys:ye, xs:xe], cmap='gray')
    axes3[0].set_title("Ảnh Gốc (Zoom)")
    axes3[0].axis('off')

    axes3[1].imshow(enhanced_img[ys:ye, xs:xe], cmap='gray')
    axes3[1].set_title("Sau Enhancement (Zoom)")
    axes3[1].axis('off')

    axes3[2].imshow(gabor_binary_masked[ys:ye, xs:xe], cmap='gray')
    axes3[2].set_title("Sau Gabor (Zoom)")
    axes3[2].axis('off')

    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "06_gabor_comparison.png")
    plt.savefig(path3, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {path3}")

    # =========================================================================
    # TÓM TẮT
    # =========================================================================
    print("\n" + "=" * 60)
    print("TÓM TẮT BƯỚC 5: GABOR FILTERING")
    print("=" * 60)
    print(f"  Tần số sử dụng     : {median_freq:.4f} (median)")
    print(f"  Bước sóng           : {1/median_freq:.1f} pixel")
    print(f"  Tham số kx, ky      : 0.5, 0.5")
    print(f"  Bước nhảy góc       : 3° (60 hướng)")
    print(f"  Pixel vân (đen)     : {np.sum(gabor_binary_masked[mask==1] == 0)}")
    print(f"  Pixel nền (trắng)   : {np.sum(gabor_binary_masked[mask==1] == 255)}")
    ratio = np.sum(gabor_binary_masked[mask==1] == 0) / np.sum(mask==1) * 100
    print(f"  Tỷ lệ vân/tổng     : {ratio:.1f}%")
    print("=" * 60)
    print(f"\n  Kết quả đã lưu tại: {OUTPUT_DIR}/")
    print("  → 06_gabor_overview.png     (Tổng quan pipeline)")
    print("  → 06_gabor_kernels.png      (Gabor kernels 8 hướng)")
    print("  → 06_gabor_comparison.png   (So sánh trước/sau)")
    print(f"\n  Bước tiếp theo: 07_binarize_thin.py")
    print("  (Nhị phân hóa + Làm mảnh)")


if __name__ == "__main__":
    process_gabor()
