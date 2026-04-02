"""
Bước 4: Frequency Estimation (Ước lượng Tần số Đường Vân)
==========================================================
Mục tiêu: Xác định KHOẢNG CÁCH giữa các đường vân (ridge spacing) tại mỗi vùng.
           Kết quả: tần số = 1 / bước sóng (wavelength) tính bằng pixel.

Tại sao cần Frequency?
  → Bộ lọc Gabor cần 2 tham số: HƯỚNG (orientation) + TẦN SỐ (frequency).
  → Nếu chỉ có hướng mà thiếu tần số, Gabor sẽ không khớp với cấu trúc vân.
  → Ảnh 500dpi thông thường: bước sóng ~ 5-15 pixel → tần số ~ 1/5 đến 1/15.

Thuật toán (giống ridgefreq.m + freqest.m của MATLAB):
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Với MỖI block (32×32 pixel) trên ảnh:                              │
  │                                                                      │
  │  1. Lấy hướng trung bình của block từ Orientation Field              │
  │                                                                      │
  │  2. XOAY block sao cho đường vân thẳng đứng (vertical)              │
  │     → Sau khi xoay, các đường vân song song theo chiều dọc           │
  │                                                                      │
  │  3. "Chiếu" (project) block xuống theo chiều dọc                     │
  │     → Cộng tất cả pixel trong mỗi cột → được 1 tín hiệu 1D          │
  │     → Tín hiệu này có dạng sóng: đỉnh = vân, đáy = thung lũng      │
  │                                                                      │
  │  4. Đếm số đỉnh (peaks) trong tín hiệu 1D                           │
  │     → Khoảng cách giữa các đỉnh = bước sóng (wavelength)            │
  │     → Tần số = 1 / bước sóng                                        │
  └──────────────────────────────────────────────────────────────────────┘

Tham khảo:
  Hong, L., Wan, Y., and Jain, A. K. "Fingerprint image enhancement:
  Algorithm and performance evaluation." IEEE TPAMI 20(8), 1998.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter1d, rotate
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

full_enhancement_pipeline = step03_enh.full_enhancement_pipeline
estimate_orientation = step04_ori.estimate_orientation


# ============================================================================
# BƯỚC 4A: ƯỚC LƯỢNG TẦN SỐ CHO 1 BLOCK
# ============================================================================
def freqest(block, block_orient, wind_size=5, min_wave_length=5, max_wave_length=15):
    """
    Ước lượng tần số đường vân trong 1 block ảnh.
    Tương đương: freqest.m trong MATLAB.

    Thuật toán chi tiết:

    BƯỚC 1 - Tìm hướng trung bình:
      Lấy trung bình các góc trong block. Dùng trick "doubled angle"
      (nhân đôi góc trước khi trung bình) để tránh lỗi quanh 0°/180°.
      VD: Nếu có 2 góc 1° và 179° → trung bình thông thường = 90° (SAI!)
          Nhưng doubled angle trick cho kết quả ≈ 0° hoặc 180° (ĐÚNG!)

    BƯỚC 2 - Xoay block:
      Xoay ảnh sao cho các đường vân thẳng đứng.
      Góc xoay = hướng trung bình + 90° (vì ta muốn vân DỌC)

    BƯỚC 3 - Chiếu (Projection):
      Cộng tất cả pixel theo chiều DỌC (sum columns).
      Kết quả là mảng 1D có dạng sóng hình sin.

      Hình dung (block sau khi xoay, vân thẳng đứng):
        ║  ║  ║  ║      → Cột 1: toàn pixel tối → tổng THẤP
        ║  ║  ║  ║      → Cột 2: toàn pixel sáng → tổng CAO
        ║  ║  ║  ║      → Kết quả projection: ↓ ↑ ↓ ↑ ↓ ↑ (hình sóng)

    BƯỚC 4 - Tìm đỉnh:
      Dùng "greyscale dilation" (giãn nở) để tìm local maxima.
      Đỉnh = vị trí mà giá trị projection = giá trị dilation.
      Khoảng cách trung bình giữa các đỉnh = bước sóng (wavelength).

    Returns:
      frequency: Tần số (1/bước sóng). Trả về 0 nếu không xác định được.
    """
    rows, cols = block.shape

    # --- BƯỚC 1: Hướng trung bình (doubled angle trick) ---
    orient_doubled = 2 * block_orient.ravel()
    cos_orient = np.mean(np.cos(orient_doubled))
    sin_orient = np.mean(np.sin(orient_doubled))
    mean_orient = np.arctan2(sin_orient, cos_orient) / 2

    # --- BƯỚC 2: Xoay block cho vân thẳng đứng ---
    # Góc xoay (độ): hướng vân + 90° để vân trở thành VERTICAL
    rotate_angle = np.degrees(mean_orient) + 90
    rotated = rotate(block, rotate_angle, reshape=False, order=1, mode='nearest')

    # Cắt bỏ viền không hợp lệ sau khi xoay (tránh ảnh hưởng projection)
    crop_size = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - crop_size) / 2))
    if crop_size < 3 or offset < 0 or offset + crop_size > rows:
        return 0.0
    cropped = rotated[offset:offset + crop_size, offset:offset + crop_size]

    if cropped.size == 0:
        return 0.0

    # --- BƯỚC 3: Chiếu (projection) xuống theo chiều dọc ---
    # Cộng tất cả các hàng → mỗi cột cho 1 giá trị
    projection = np.sum(cropped, axis=0)

    # --- BƯỚC 4: Tìm đỉnh (peaks) ---
    # Greyscale dilation: thay mỗi điểm bằng MAX trong cửa sổ wind_size
    # Tương đương: ordfilt2(proj, windsze, ones(1,windsze)) trong MATLAB
    dilation = maximum_filter1d(projection, size=wind_size)

    # Đỉnh = nơi dilation bằng giá trị gốc VÀ giá trị > trung bình
    mean_proj = np.mean(projection)
    max_points = (dilation == projection) & (projection > mean_proj)
    max_indices = np.where(max_points)[0]

    # Tính tần số từ khoảng cách giữa các đỉnh
    if len(max_indices) < 2:
        return 0.0

    num_peaks = len(max_indices)
    wave_length = (max_indices[-1] - max_indices[0]) / (num_peaks - 1)

    # Kiểm tra bước sóng có nằm trong giới hạn hợp lệ không
    if min_wave_length < wave_length < max_wave_length:
        return 1.0 / wave_length
    else:
        return 0.0


# ============================================================================
# BƯỚC 4B: TÍNH TẦN SỐ CHO TOÀN BỘ ẢNH
# ============================================================================
def ridge_frequency(img, mask, orient_img, block_size=32, wind_size=5,
                    min_wave_length=5, max_wave_length=15):
    """
    Tính tần số đường vân cho toàn bộ ảnh, chia thành các block.
    Tương đương: ridgefreq.m trong MATLAB.

    Tham số khuyến nghị cho ảnh 500dpi (giống MATLAB):
      block_size=32, wind_size=5, min_wave_length=5, max_wave_length=15

    Returns:
      freq_img:    Ảnh tần số (cùng kích thước với img), mỗi block có 1 giá trị freq
      median_freq: Tần số trung vị (median) trên toàn bộ vùng vân tay hợp lệ
    """
    rows, cols = img.shape
    freq_img = np.zeros_like(img, dtype=np.float64)

    for r in range(0, rows - block_size, block_size):
        for c in range(0, cols - block_size, block_size):
            # Cắt block ảnh và block orientation
            blk_img = img[r:r + block_size, c:c + block_size].astype(np.float64)
            blk_orient = orient_img[r:r + block_size, c:c + block_size]

            # Ước lượng tần số cho block này
            freq = freqest(blk_img, blk_orient, wind_size,
                           min_wave_length, max_wave_length)

            # Gán tần số cho toàn block
            freq_img[r:r + block_size, c:c + block_size] = freq

    # Áp mask: chỉ giữ tần số ở vùng vân tay
    freq_img = freq_img * mask

    # Tần số trung vị (median) - giá trị này rất hữu ích
    # Tác giả MATLAB cũng lưu ý: median frequency thường cho kết quả tốt hơn
    # so với freq_img chi tiết khi đưa vào Gabor filter
    valid_freqs = freq_img[freq_img > 0]
    if len(valid_freqs) > 0:
        median_freq = np.median(valid_freqs)
    else:
        median_freq = 1.0 / 9.0  # Giá trị mặc định cho ảnh 500dpi

    return freq_img, median_freq


# ============================================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================================
def process_frequency():
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
    print("Đang chạy Enhancement pipeline...")
    enhanced_img, mask, _ = full_enhancement_pipeline(
        img, clip_limit=2.5, grid_size=(8, 8),
        block_size=16, var_threshold=0.005
    )

    # === Bước 3: Orientation Field ===
    print("Đang tính Orientation Field...")
    orient_img, reliability = estimate_orientation(enhanced_img)

    # === Bước 4: Frequency Estimation ===
    print("Đang tính Frequency (tần số đường vân)...")
    freq_img, median_freq = ridge_frequency(
        enhanced_img, mask, orient_img,
        block_size=32, wind_size=5,
        min_wave_length=5, max_wave_length=15
    )

    median_wavelength = 1.0 / median_freq if median_freq > 0 else 0
    print(f"  Tần số trung vị: {median_freq:.4f} (bước sóng ≈ {median_wavelength:.1f} pixel)")

    # === Minh họa freqest cho 1 block cụ thể ===
    # Chọn 1 block ở giữa ảnh để minh họa chi tiết thuật toán
    h, w = img.shape
    demo_r, demo_c = h // 2, w // 2
    bs = 32
    demo_block = enhanced_img[demo_r:demo_r + bs, demo_c:demo_c + bs].astype(np.float64)
    demo_orient = orient_img[demo_r:demo_r + bs, demo_c:demo_c + bs]

    # Tính hướng trung bình và xoay
    orient_doubled = 2 * demo_orient.ravel()
    mean_orient = np.arctan2(np.mean(np.sin(orient_doubled)),
                             np.mean(np.cos(orient_doubled))) / 2
    rotate_angle = np.degrees(mean_orient) + 90
    rotated = rotate(demo_block, rotate_angle, reshape=False, order=1, mode='nearest')
    crop_sz = int(np.fix(bs / np.sqrt(2)))
    off = int(np.fix((bs - crop_sz) / 2))
    cropped = rotated[off:off + crop_sz, off:off + crop_sz]
    projection = np.sum(cropped, axis=0)

    # Tìm peaks trong projection
    dilation = maximum_filter1d(projection, size=5)
    mean_proj = np.mean(projection)
    max_pts = (dilation == projection) & (projection > mean_proj)
    peak_indices = np.where(max_pts)[0]

    # =========================================================================
    # HÌNH 1: Tổng quan Frequency Estimation
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Bước 4: Frequency Estimation (Ước lượng Tần số Đường Vân)",
                 fontsize=15, fontweight='bold')

    # Ảnh gốc
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("1. Ảnh Gốc")
    axes[0, 0].axis('off')

    # Ảnh enhanced
    axes[0, 1].imshow(enhanced_img, cmap='gray')
    axes[0, 1].set_title("2. Sau Enhancement")
    axes[0, 1].axis('off')

    # Frequency Map
    freq_display = np.where(freq_img > 0, freq_img, np.nan)
    im1 = axes[0, 2].imshow(freq_display, cmap='jet', vmin=0.05, vmax=0.2)
    axes[0, 2].set_title(f"3. Frequency Map\n(Median freq = {median_freq:.4f})")
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, label="Tần số (1/pixel)")

    # Wavelength Map (1/freq)
    with np.errstate(divide='ignore', invalid='ignore'):
        wavelength_img = np.where(freq_img > 0, 1.0 / freq_img, np.nan)
    im2 = axes[1, 0].imshow(wavelength_img, cmap='viridis', vmin=5, vmax=15)
    axes[1, 0].set_title(f"4. Wavelength Map\n(Median ≈ {median_wavelength:.1f} pixel)")
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, label="Bước sóng (pixel)")

    # Histogram tần số
    valid_f = freq_img[freq_img > 0]
    if len(valid_f) > 0:
        axes[1, 1].hist(valid_f, bins=30, color='steelblue', alpha=0.8, edgecolor='white')
        axes[1, 1].axvline(x=median_freq, color='red', linestyle='--', linewidth=2,
                           label=f'Median = {median_freq:.4f}')
        axes[1, 1].set_title("5. Phân bố Tần số")
        axes[1, 1].set_xlabel("Tần số (1/pixel)")
        axes[1, 1].set_ylabel("Số block")
        axes[1, 1].legend()

    # Histogram bước sóng
    valid_wl = 1.0 / valid_f if len(valid_f) > 0 else np.array([])
    if len(valid_wl) > 0:
        axes[1, 2].hist(valid_wl, bins=30, color='darkorange', alpha=0.8, edgecolor='white')
        axes[1, 2].axvline(x=median_wavelength, color='red', linestyle='--', linewidth=2,
                           label=f'Median = {median_wavelength:.1f} px')
        axes[1, 2].set_title("6. Phân bố Bước Sóng")
        axes[1, 2].set_xlabel("Bước sóng (pixel)")
        axes[1, 2].set_ylabel("Số block")
        axes[1, 2].legend()

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "05_frequency_overview.png")
    plt.savefig(path1, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {path1}")

    # =========================================================================
    # HÌNH 2: Minh họa chi tiết thuật toán cho 1 block
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle("Chi tiết thuật toán freqest cho 1 block (32×32)",
                  fontsize=14, fontweight='bold')

    # Block gốc
    axes2[0, 0].imshow(demo_block, cmap='gray')
    axes2[0, 0].set_title(f"A. Block gốc\n(Hướng TB ≈ {np.degrees(mean_orient):.1f}°)")
    axes2[0, 0].axis('off')

    # Block đã xoay
    axes2[0, 1].imshow(cropped, cmap='gray')
    axes2[0, 1].set_title(f"B. Sau xoay {rotate_angle:.1f}°\n(Vân giờ thẳng đứng)")
    axes2[0, 1].axis('off')

    # Projection 1D
    axes2[1, 0].plot(projection, 'b-', linewidth=1.5, label='Projection')
    axes2[1, 0].axhline(y=mean_proj, color='gray', linestyle=':', label=f'Mean={mean_proj:.0f}')
    if len(peak_indices) > 0:
        axes2[1, 0].plot(peak_indices, projection[peak_indices], 'rv', markersize=10,
                         label=f'{len(peak_indices)} đỉnh')
    axes2[1, 0].set_title("C. Projection (cộng pixel theo cột)")
    axes2[1, 0].set_xlabel("Vị trí cột (pixel)")
    axes2[1, 0].set_ylabel("Tổng giá trị pixel")
    axes2[1, 0].legend()
    axes2[1, 0].grid(True, alpha=0.3)

    # Giải thích kết quả
    axes2[1, 1].axis('off')
    if len(peak_indices) >= 2:
        wl = (peak_indices[-1] - peak_indices[0]) / (len(peak_indices) - 1)
        freq_val = 1.0 / wl if wl > 0 else 0
        explanation = (
            f"KẾT QUẢ PHÂN TÍCH BLOCK\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"  Hướng vân trung bình : {np.degrees(mean_orient):.1f}°\n"
            f"  Góc xoay             : {rotate_angle:.1f}°\n"
            f"  Số đỉnh tìm được    : {len(peak_indices)}\n\n"
            f"  Đỉnh đầu           : vị trí {peak_indices[0]}\n"
            f"  Đỉnh cuối          : vị trí {peak_indices[-1]}\n"
            f"  Bước sóng (λ)      : {wl:.1f} pixel\n"
            f"  Tần số (f = 1/λ)   : {freq_val:.4f}\n\n"
            f"  ► Nghĩa là: Cứ mỗi {wl:.1f} pixel\n"
            f"    lại có 1 đường vân mới."
        )
    else:
        explanation = (
            f"KẾT QUẢ: Không đủ đỉnh\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"  Số đỉnh: {len(peak_indices)} (cần ≥ 2)\n"
            f"  → Block này không xác định\n"
            f"    được tần số.\n"
            f"  → Gán tần số = 0"
        )
    axes2[1, 1].text(0.1, 0.9, explanation, transform=axes2[1, 1].transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "05_frequency_block_detail.png")
    plt.savefig(path2, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {path2}")

    # =========================================================================
    # TÓM TẮT
    # =========================================================================
    coverage = np.sum(freq_img > 0) / np.sum(mask > 0) * 100 if np.sum(mask > 0) > 0 else 0

    print("\n" + "=" * 60)
    print("TÓM TẮT BƯỚC 4: FREQUENCY ESTIMATION")
    print("=" * 60)
    print(f"  Block size          : 32×32 pixel")
    print(f"  Bước sóng hợp lệ   : {5} → {15} pixel")
    print(f"  Tần số trung vị     : {median_freq:.4f}")
    print(f"  Bước sóng trung vị  : {median_wavelength:.1f} pixel")
    print(f"  Vùng xác định được  : {coverage:.1f}% vùng vân tay")
    print(f"  Tổng block hợp lệ   : {len(valid_f)}")
    print("=" * 60)
    print(f"\n  Kết quả đã lưu tại: {OUTPUT_DIR}/")
    print("  → 05_frequency_overview.png       (Tổng quan)")
    print("  → 05_frequency_block_detail.png   (Chi tiết 1 block)")
    print(f"\n  Bước tiếp theo: 06_gabor_filter.py")
    print("  (Lọc Gabor 2D - Tăng cường đường vân cuối cùng)")


if __name__ == "__main__":
    process_frequency()
