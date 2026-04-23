"""
Bước 2: Enhancement (Tăng cường ảnh vân tay)
=============================================
Mục tiêu: Làm rõ các đường vân (ridge) và giảm nhiễu trước khi đưa vào
           các bước xử lý tiếp theo (Orientation, Gabor, Thinning...).

Tác giả gốc (MATLAB) dùng FFT Enhancement (fft_enhance_cubs.m) - rất phức tạp.
Ở đây ta dùng các kỹ thuật tương đương trong OpenCV:
  1. Histogram Equalization (Cân bằng histogram toàn cục)
  2. CLAHE (Cân bằng histogram thích ứng cục bộ) ← Phương pháp chính
  3. Kết hợp: Chuẩn hóa + Tách nền + CLAHE → Ảnh tăng cường hoàn chỉnh

So sánh với tác giả gốc:
  - fft_enhance_cubs.m: Chia ảnh thành block → FFT → bandpass filter → IFFT
  - CLAHE: Chia ảnh thành block → cân bằng histogram cục bộ → nội suy ghép nối
  → Cả hai đều xử lý CỤC BỘ (theo block), mục đích giống nhau: tăng độ tương 
    phản cục bộ để đường vân rõ hơn ở mọi vùng trên ảnh.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================================
import config

BASE_DIR = config.BASE_DIR
DATASET_PATH = config.DATASET_PATH
OUTPUT_DIR = config.OUTPUT_DIR
SAMPLE_IMAGE = "101_1.tif"

# ============================================================================
# HÀM TỪ BƯỚC TRƯỚC (Preprocessing)
# ============================================================================
def normalize_image(img):
    """
    Chuẩn hóa ảnh về mean=0, variance=1
    Tương đương: normalise.m trong MATLAB
    """
    img = img.astype(np.float32)
    mean = np.mean(img)
    std = np.std(img)
    if std == 0:
        std = 1
    return (img - mean) / std


def segment_fingerprint(img, block_size=16, threshold=0.1):
    """
    Tách nền dựa trên phương sai (Variance) theo từng block.
    Tương đương: ridgesegment.m trong MATLAB
    
    - Vùng vân tay → phương sai CAO (đen trắng xen kẽ)
    - Vùng nền     → phương sai THẤP (màu đồng đều)
    """
    rows, cols = img.shape
    mask = np.zeros_like(img, dtype=np.uint8)

    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            block = img[r:min(r + block_size, rows), c:min(c + block_size, cols)]
            block_var = np.var(block / 255.0)
            if block_var > threshold:
                mask[r:min(r + block_size, rows), c:min(c + block_size, cols)] = 1

    return mask


# ============================================================================
# HÀM MỚI: ENHANCEMENT (Tăng cường ảnh)
# ============================================================================
def histogram_equalization(img):
    """
    Cân bằng Histogram TOÀN CỤC (Global Histogram Equalization).
    
    Nguyên lý:
      - Kéo giãn histogram để các giá trị pixel phân bố đều trên dải 0-255.
      - Giúp tăng độ tương phản tổng thể.
    
    Nhược điểm:
      - Xử lý trên TOÀN BỘ ảnh → nếu nền quá sáng, nó sẽ kéo nền xuống
        nhưng đồng thời cũng kéo luôn vùng vân tay → mất chi tiết cục bộ.
      - Không phù hợp khi ảnh có vùng sáng/tối khác nhau nhiều.
    """
    return cv2.equalizeHist(img)


def clahe_enhancement(img, clip_limit=2.0, grid_size=(8, 8)):
    """
    CLAHE - Contrast Limited Adaptive Histogram Equalization.
    (Cân bằng Histogram Thích Ứng Cục Bộ có Giới Hạn Tương Phản)
    
    Đây là phương pháp CHÍNH mà ta sử dụng thay cho FFT Enhancement của tác giả.
    
    Nguyên lý:
      1. Chia ảnh thành các block nhỏ (tile), kích thước = grid_size
         VD: (8,8) nghĩa là chia ảnh thành lưới 8x8 = 64 block
      2. Với MỖI block, tính histogram riêng và cân bằng cục bộ
         → Những vùng tối sẽ được làm sáng hơn, vùng sáng được giảm bớt
      3. clip_limit: Giới hạn mức khuếch đại tương phản
         - Nếu clip_limit QUÁ CAO → nhiễu sẽ bị khuếch đại theo
         - Nếu clip_limit QUÁ THẤP → hiệu quả tăng cường kém
         - Giá trị thường dùng: 2.0 - 3.0
      4. Nội suy song tuyến (bilinear interpolation) giữa các block
         → Tránh hiện tượng "đường viền nhân tạo" ở biên các block
    
    So sánh với FFT Enhancement (fft_enhance_cubs.m):
      ┌──────────────────┬─────────────────────────────────┐
      │      CLAHE       │     FFT Enhancement (MATLAB)    │
      ├──────────────────┼─────────────────────────────────┤
      │ Chia block       │ Chia block (BLKSZ=6,12,24)      │
      │ Histogram cục bộ │ FFT cục bộ                      │
      │ Clip limit       │ Bandpass filter (RMIN, RMAX)     │
      │ Nội suy bilinear │ Overlap + spectral window       │
      │ Đơn giản, nhanh  │ Phức tạp, chính xác hơn         │
      └──────────────────┴─────────────────────────────────┘
    
    Tham số:
      clip_limit: Giới hạn tương phản (mặc định = 2.0)
      grid_size:  Kích thước lưới chia block (mặc định = (8,8))
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)


def full_enhancement_pipeline(img, clip_limit=2.0, grid_size=(8, 8),
                               block_size=16, var_threshold=0.1):
    """
    Pipeline tăng cường hoàn chỉnh: Chuẩn hóa → Tách nền → CLAHE → Áp mask
    
    Đây là hàm tổng hợp sẽ được tái sử dụng ở các bước sau.
    
    Returns:
        enhanced_img: Ảnh đã tăng cường (uint8, chỉ giữ vùng vân tay)
        mask:         Mask vùng vân tay (0/1)
    """
    # Bước 1: Chuẩn hóa 
    norm_img = normalize_image(img)
    # Scale lại về 0-255 cho các bước tiếp
    norm_display = cv2.normalize(norm_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Bước 2: Tách nền (Segmentation)
    mask = segment_fingerprint(img, block_size=block_size, threshold=var_threshold)

    # Bước 3: CLAHE Enhancement
    enhanced = clahe_enhancement(norm_display, clip_limit=clip_limit, grid_size=grid_size)

    # Bước 4: Áp mask - chỉ giữ vùng vân tay, nền chuyển thành trắng (255)
    # Nền = trắng vì ở bước Nhị phân hóa sau này, nền trắng = Valley (thung lũng)
    enhanced_masked = np.where(mask == 1, enhanced, 255).astype(np.uint8)

    return enhanced_masked, mask, enhanced


# ============================================================================
# CHƯƠNG TRÌNH CHÍNH: SO SÁNH CÁC PHƯƠNG PHÁP ENHANCEMENT
# ============================================================================
def process_enhancement():
    img_path = os.path.join(DATASET_PATH, SAMPLE_IMAGE)
    if not os.path.exists(img_path):
        print(f"Lỗi: Không tìm thấy ảnh tại {img_path}")
        print("Hãy đảm bảo thư mục FVC2002/DB1_B tồn tại.")
        return

    # === Đọc ảnh gốc ===
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Lỗi: Không đọc được ảnh.")
        return

    print(f"Đã đọc ảnh: {SAMPLE_IMAGE} | Kích thước: {img.shape}")

    # === Áp dụng các phương pháp Enhancement ===
    # 1. Histogram Equalization (toàn cục)
    hist_eq = histogram_equalization(img)

    # 2. CLAHE với các tham số khác nhau
    clahe_low = clahe_enhancement(img, clip_limit=1.0, grid_size=(8, 8))
    clahe_mid = clahe_enhancement(img, clip_limit=2.0, grid_size=(8, 8))
    clahe_high = clahe_enhancement(img, clip_limit=4.0, grid_size=(8, 8))

    # 3. Pipeline hoàn chỉnh (Normalize + Segment + CLAHE)
    enhanced_final, mask, _ = full_enhancement_pipeline(
        img, clip_limit=2.5, grid_size=(8, 8),
        block_size=16, var_threshold=0.005
    )

    # =========================================================================
    # HÌNH 1: So sánh các phương pháp Enhancement
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("So sánh các phương pháp Enhancement", fontsize=16, fontweight='bold')

    images = [
        (img, "1. Ảnh Gốc"),
        (hist_eq, "2. Histogram EQ\n(Toàn cục)"),
        (clahe_low, "3. CLAHE\n(clip=1.0, nhẹ)"),
        (clahe_mid, "4. CLAHE\n(clip=2.0, vừa)"),
        (clahe_high, "5. CLAHE\n(clip=4.0, mạnh)"),
        (enhanced_final, "6. Pipeline Hoàn Chỉnh\n(Norm + Segment + CLAHE)"),
    ]

    for ax, (image, title) in zip(axes.ravel(), images):
        ax.imshow(image, cmap='gray')
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    output_path_1 = os.path.join(OUTPUT_DIR, "03_enhancement_comparison.png")
    plt.savefig(output_path_1, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {output_path_1}")

    # =========================================================================
    # HÌNH 2: So sánh Histogram trước và sau Enhancement
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    fig2.suptitle("So sánh Histogram trước và sau Enhancement", fontsize=16, fontweight='bold')

    # Ảnh gốc + Histogram
    axes2[0, 0].imshow(img, cmap='gray')
    axes2[0, 0].set_title("Ảnh Gốc")
    axes2[0, 0].axis('off')

    axes2[0, 1].hist(img.ravel(), bins=256, range=[0, 256], color='steelblue', alpha=0.7)
    axes2[0, 1].set_title("Histogram Ảnh Gốc")
    axes2[0, 1].set_xlabel("Giá trị Pixel")
    axes2[0, 1].set_ylabel("Số lượng Pixel")
    axes2[0, 1].axvline(x=np.mean(img), color='red', linestyle='--', label=f'Mean={np.mean(img):.0f}')
    axes2[0, 1].legend()

    # Ảnh sau CLAHE + Histogram
    axes2[1, 0].imshow(enhanced_final, cmap='gray')
    axes2[1, 0].set_title("Sau Pipeline Enhancement")
    axes2[1, 0].axis('off')

    # Chỉ lấy histogram của vùng vân tay (bỏ nền trắng 255)
    fp_pixels = enhanced_final[mask == 1]
    axes2[1, 1].hist(fp_pixels.ravel(), bins=256, range=[0, 256], color='darkorange', alpha=0.7)
    axes2[1, 1].set_title("Histogram Sau Enhancement\n(chỉ vùng vân tay)")
    axes2[1, 1].set_xlabel("Giá trị Pixel")
    axes2[1, 1].set_ylabel("Số lượng Pixel")
    axes2[1, 1].axvline(x=np.mean(fp_pixels), color='red', linestyle='--',
                        label=f'Mean={np.mean(fp_pixels):.0f}')
    axes2[1, 1].legend()

    plt.tight_layout()
    output_path_2 = os.path.join(OUTPUT_DIR, "03_enhancement_histogram.png")
    plt.savefig(output_path_2, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {output_path_2}")

    # =========================================================================
    # HÌNH 3: Zoom vào chi tiết vân tay - Trước vs Sau
    # =========================================================================
    # Cắt một vùng nhỏ để thấy rõ sự khác biệt ở mức chi tiết đường vân
    h, w = img.shape
    # Lấy vùng trung tâm (nơi có nhiều vân tay nhất)
    y_start, y_end = h // 4, 3 * h // 4
    x_start, x_end = w // 4, 3 * w // 4

    crop_original = img[y_start:y_end, x_start:x_end]
    crop_enhanced = enhanced_final[y_start:y_end, x_start:x_end]

    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    fig3.suptitle("Zoom chi tiết: Trước vs Sau Enhancement", fontsize=14, fontweight='bold')

    axes3[0].imshow(crop_original, cmap='gray')
    axes3[0].set_title("Trước Enhancement (Gốc)")
    axes3[0].axis('off')

    axes3[1].imshow(crop_enhanced, cmap='gray')
    axes3[1].set_title("Sau Enhancement (CLAHE + Segment)")
    axes3[1].axis('off')

    plt.tight_layout()
    output_path_3 = os.path.join(OUTPUT_DIR, "03_enhancement_zoom.png")
    plt.savefig(output_path_3, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {output_path_3}")

    # =========================================================================
    # TÓM TẮT
    # =========================================================================
    print("\n" + "=" * 60)
    print("TÓM TẮT BƯỚC 2: ENHANCEMENT")
    print("=" * 60)
    print(f"  Ảnh gốc      - Mean: {np.mean(img):.1f}, Std: {np.std(img):.1f}")
    print(f"  Sau CLAHE     - Mean: {np.mean(fp_pixels):.1f}, Std: {np.std(fp_pixels):.1f}")
    print(f"  Phương pháp   : CLAHE (clip_limit=2.5, grid=8x8)")
    print(f"  Kết hợp       : Normalize → Segment → CLAHE → Mask")
    print("=" * 60)
    print(f"\n  Các file kết quả đã được lưu tại: {OUTPUT_DIR}/")
    print("  → 03_enhancement_comparison.png  (So sánh phương pháp)")
    print("  → 03_enhancement_histogram.png   (So sánh histogram)")
    print("  → 03_enhancement_zoom.png        (Zoom chi tiết)")
    print("\n  Bước tiếp theo: 04_orientation_field.py")
    print("  (Tính hướng vân - Orientation Field Estimation)")


if __name__ == "__main__":
    process_enhancement()
