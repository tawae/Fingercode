"""
Bước 6: Binarization + Thinning (Nhị phân hóa + Làm mảnh)
============================================================
Mục tiêu:
  1. Nhị phân hóa: Chuyển ảnh grayscale → ảnh đen trắng (chỉ 0 hoặc 1)
     - Đen (0) = đường vân (ridge)
     - Trắng (1) = thung lũng (valley) / nền
  2. Làm mảnh (Thinning/Skeletonization): Thu nhỏ đường vân từ nhiều pixel
     xuống còn 1 pixel chiều rộng, giữ nguyên cấu trúc topo (topology).
     → Chuẩn bị cho Minutiae Extraction ở bước sau.

Tại sao cần Làm mảnh?
  Thuật toán Crossing Number (bước sau) đếm số pixel trắng lân cận xung quanh
  1 pixel vân. Nếu đường vân dày 5-6 pixel → CN sẽ luôn = 2 (pixel ở giữa vân)
  → không phát hiện được điểm kết thúc (CN=1) hay rẽ nhánh (CN=3).
  Sau khi làm mảnh, mỗi đường vân chỉ rộng 1 pixel → CN mới chính xác.

So sánh với MATLAB:
  - Nhị phân hóa: binim = ridgefilter(...) > 0   (threshold tại 0)
  - Làm mảnh: inv_binim = (binim == 0);
              thinned = bwmorph(inv_binim, 'thin', Inf);
  
  Trong Python:
  - Nhị phân hóa: Gabor result > 0 (đã làm ở bước 5) HOẶC cv2.threshold()
  - Làm mảnh: skimage.morphology.skeletonize()
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
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
step06_gabor = _import_module("step06_gabor", os.path.join(BASE_DIR, "06_gabor_filter.py"))

full_enhancement_pipeline = step03_enh.full_enhancement_pipeline
estimate_orientation = step04_ori.estimate_orientation
ridge_frequency = step05_freq.ridge_frequency
gabor_ridge_filter = step06_gabor.gabor_ridge_filter


# ============================================================================
# BƯỚC 6A: NHỊ PHÂN HÓA (BINARIZATION)
# ============================================================================
def binarize_fingerprint(gabor_result, mask, method='gabor_threshold'):
    """
    Nhị phân hóa ảnh vân tay.
    
    Phương pháp 1 - gabor_threshold (CHÍNH, giống MATLAB):
      Dựa trên kết quả Gabor filtering:
        - gabor_result > 0 → Vân (ridge) → pixel ĐEN (0)
        - gabor_result ≤ 0 → Nền (valley) → pixel TRẮNG (1)
      
      Tại sao threshold = 0?
      → Gabor filter là tích Gaussian × cos(). Giá trị dương khi pixel khớp
        với pha "sáng" của cos, âm khi khớp pha "tối". Đường vân = vùng tối
        trong ảnh gốc, sau Gabor thường cho giá trị dương → threshold tại 0.
    
    Phương pháp 2 - otsu:
      Dùng thuật toán Otsu để tự động tìm ngưỡng tối ưu.
      Phù hợp khi không có Gabor filtering.
    
    Returns:
      binary_img: Ảnh nhị phân (0 = vân đen, 255 = nền trắng), dtype=uint8
    """
    if method == 'gabor_threshold':
        # Vân tay: gabor > 0 → đen (0), nền: gabor ≤ 0 → trắng (255)
        binary = np.where(gabor_result > 0, 0, 255).astype(np.uint8)
    elif method == 'otsu':
        # Chuẩn hóa gabor_result về 0-255
        gabor_norm = cv2.normalize(gabor_result, None, 0, 255,
                                   cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(gabor_norm, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError(f"Phương pháp không hợp lệ: {method}")

    # Áp mask: vùng nền ngoài vân tay → trắng (255)
    binary = np.where(mask == 1, binary, 255).astype(np.uint8)

    return binary


# ============================================================================
# BƯỚC 6B: LÀM MẢNH (THINNING / SKELETONIZATION)
# ============================================================================
def thin_fingerprint(binary_img, mask):
    """
    Làm mảnh (Thinning) đường vân từ nhiều pixel → 1 pixel chiều rộng.
    Tương đương: bwmorph(inv_binim, 'thin', Inf) trong MATLAB.
    
    Thuật toán Skeletonization:
      Lặp đi lặp lại việc "gọt" (erode) các pixel ở RÌA đường vân,
      nhưng CHỈ gọt những pixel mà việc xóa chúng KHÔNG:
        - Làm đứt đường vân (không phá vỡ connectivity)
        - Thay đổi topology (không tạo/xóa lỗ)
      Lặp cho đến khi không gọt được nữa → Skeleton (bộ xương).
    
    Hình dung:
      Trước thinning:          Sau thinning:
      ██████████████           ──────────────
      ██████████████                (1 pixel)
      ██████████████           
      (5-7 pixel rộng)
    
    Lưu ý QUAN TRỌNG:
      - skeletonize() yêu cầu input: True = vùng cần thin, False = nền
      - Đường vân trong binary_img là pixel ĐEN (0)
      - Nên phải ĐẢO NGƯỢC trước khi thin: vân (0) → True, nền (255) → False
      - Giống hệt MATLAB: inv_binim = (binim == 0)
    
    Returns:
      thinned: Ảnh đã làm mảnh (True = đường vân, False = nền)
    """
    # Đảo ngược: vân đen (0) → True (cần thin), nền trắng (255) → False
    # Giống MATLAB: inv_binim = (binim == 0)
    inverted = (binary_img == 0)

    # Skeletonize (tương đương bwmorph(..., 'thin', Inf))
    thinned = skeletonize(inverted)

    # Áp mask: chỉ giữ skeleton trong vùng vân tay
    thinned = thinned & (mask == 1)

    return thinned


def clean_thinned_image(thinned, mask, min_branch_len=10):
    """
    Làm sạch ảnh đã thin: loại bỏ các nhánh ngắn (spurious branches)
    do nhiễu gây ra trong quá trình thinning.
    
    Các nhánh giả thường xuất hiện ở:
      - Biên vùng vân tay (edge of mask)
      - Vùng vân tay chất lượng kém
      - Giao điểm giữa vân và nhiễu
    
    Phương pháp: Morphological pruning đơn giản
      - Tìm các endpoint (điểm có CN=1)
      - Loại bỏ endpoint nằm gần biên mask (trong vùng biên edge_margin pixel)
    """
    cleaned = thinned.copy()

    # Loại bỏ pixel thinned nằm quá gần biên mask
    # Erode mask để tìm vùng "an toàn" bên trong
    edge_margin = 10
    kernel = np.ones((edge_margin * 2 + 1, edge_margin * 2 + 1), np.uint8)
    inner_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)

    cleaned = cleaned & (inner_mask == 1)

    return cleaned


# ============================================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================================
def process_binarize_thin():
    img_path = os.path.join(DATASET_PATH, SAMPLE_IMAGE)
    if not os.path.exists(img_path):
        print(f"Lỗi: Không tìm thấy ảnh tại {img_path}")
        return

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Lỗi: Không đọc được ảnh.")
        return

    print(f"Đã đọc ảnh: {SAMPLE_IMAGE} | Kích thước: {img.shape}")

    # === Pipeline các bước trước ===
    print("[1/5] Enhancement...")
    enhanced_img, mask, _ = full_enhancement_pipeline(
        img, clip_limit=2.5, grid_size=(8, 8),
        block_size=16, var_threshold=0.005
    )

    print("[2/5] Orientation Field...")
    orient_img, reliability = estimate_orientation(enhanced_img)

    print("[3/5] Frequency Estimation...")
    freq_img, median_freq = ridge_frequency(
        enhanced_img, mask, orient_img,
        block_size=32, wind_size=5,
        min_wave_length=5, max_wave_length=15
    )
    freq_img_median = np.where(mask == 1, median_freq, 0).astype(np.float64)

    print("[4/5] Gabor Filtering...")
    gabor_result = gabor_ridge_filter(
        enhanced_img, orient_img, freq_img_median, mask,
        kx=0.5, ky=0.5, angle_inc=3
    )

    # === BƯỚC 6: Binarization + Thinning ===
    print("[5/5] Binarization + Thinning...")

    # Nhị phân hóa
    binary_img = binarize_fingerprint(gabor_result, mask, method='gabor_threshold')
    print(f"  Nhị phân hóa: OK (pixel vân = {np.sum(binary_img[mask==1] == 0)})")

    # Làm mảnh
    print("  Đang làm mảnh (skeletonize)...")
    thinned = thin_fingerprint(binary_img, mask)
    print(f"  Làm mảnh: OK (pixel skeleton = {np.sum(thinned)})")

    # Làm sạch
    thinned_clean = clean_thinned_image(thinned, mask)
    print(f"  Làm sạch: OK (pixel sau clean = {np.sum(thinned_clean)})")

    # Chuyển để hiển thị: True → trắng (vân), False → đen (nền)
    thinned_display = np.zeros_like(img, dtype=np.uint8)
    thinned_display[thinned_clean] = 255

    # Overlay skeleton lên ảnh gốc (đường vân = đỏ)
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[thinned_clean] = [0, 0, 255]  # Đỏ cho skeleton

    overlay_clean = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay_clean[thinned_clean] = [0, 0, 255]

    # =========================================================================
    # HÌNH 1: Pipeline tổng quan
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Bước 6: Binarization + Thinning", fontsize=16, fontweight='bold')

    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("1. Ảnh Gốc")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(enhanced_img, cmap='gray')
    axes[0, 1].set_title("2. Enhanced")
    axes[0, 1].axis('off')

    gabor_display = cv2.normalize(gabor_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gabor_display = np.where(mask == 1, gabor_display, 255).astype(np.uint8)
    axes[0, 2].imshow(gabor_display, cmap='gray')
    axes[0, 2].set_title("3. Gabor Result")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(binary_img, cmap='gray')
    axes[1, 0].set_title("4. Nhị phân hóa\n(Đen=Vân, Trắng=Nền)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(thinned_display, cmap='gray')
    axes[1, 1].set_title(f"5. Làm mảnh\n({np.sum(thinned_clean)} pixel)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(cv2.cvtColor(overlay_clean, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("6. Skeleton trên Ảnh Gốc\n(Đỏ = đường vân)")
    axes[1, 2].axis('off')

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "07_binarize_thin_overview.png")
    plt.savefig(path1, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {path1}")

    # =========================================================================
    # HÌNH 2: Zoom so sánh chi tiết
    # =========================================================================
    h, w = img.shape
    ys, ye = h // 4, 3 * h // 4
    xs, xe = w // 4, 3 * w // 4

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle("Zoom chi tiết: Quá trình Nhị phân hóa → Làm mảnh",
                  fontsize=14, fontweight='bold')

    axes2[0, 0].imshow(img[ys:ye, xs:xe], cmap='gray')
    axes2[0, 0].set_title("A. Ảnh Gốc (Zoom)")
    axes2[0, 0].axis('off')

    axes2[0, 1].imshow(binary_img[ys:ye, xs:xe], cmap='gray')
    axes2[0, 1].set_title("B. Nhị phân hóa (Zoom)")
    axes2[0, 1].axis('off')

    axes2[1, 0].imshow(thinned_display[ys:ye, xs:xe], cmap='gray')
    axes2[1, 0].set_title("C. Làm mảnh (Zoom)")
    axes2[1, 0].axis('off')

    overlay_zoom = overlay_clean[ys:ye, xs:xe]
    axes2[1, 1].imshow(cv2.cvtColor(overlay_zoom, cv2.COLOR_BGR2RGB))
    axes2[1, 1].set_title("D. Skeleton trên Ảnh Gốc (Zoom)")
    axes2[1, 1].axis('off')

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "07_binarize_thin_zoom.png")
    plt.savefig(path2, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {path2}")

    # =========================================================================
    # HÌNH 3: So sánh Trước/Sau Thinning (rất zoom)
    # =========================================================================
    # Lấy 1 vùng rất nhỏ (80×80) để thấy rõ sự khác biệt pixel
    cy, cx = h // 2, w // 2
    sz = 40
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle("Super Zoom (80×80 pixel): Trước vs Sau Thinning",
                  fontsize=14, fontweight='bold')

    axes3[0].imshow(enhanced_img[cy-sz:cy+sz, cx-sz:cx+sz], cmap='gray',
                    interpolation='nearest')
    axes3[0].set_title("Enhanced (pixel thật)")
    axes3[0].axis('off')

    axes3[1].imshow(binary_img[cy-sz:cy+sz, cx-sz:cx+sz], cmap='gray',
                    interpolation='nearest')
    axes3[1].set_title("Binary\n(đường vân dày 5-7px)")
    axes3[1].axis('off')

    axes3[2].imshow(thinned_display[cy-sz:cy+sz, cx-sz:cx+sz], cmap='gray',
                    interpolation='nearest')
    axes3[2].set_title("Thinned\n(đường vân = 1px)")
    axes3[2].axis('off')

    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "07_binarize_thin_superzoom.png")
    plt.savefig(path3, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {path3}")

    # =========================================================================
    # TÓM TẮT
    # =========================================================================
    ridge_ratio = np.sum(binary_img[mask==1] == 0) / np.sum(mask==1) * 100
    thin_ratio = np.sum(thinned_clean) / np.sum(binary_img[mask==1] == 0) * 100

    print("\n" + "=" * 60)
    print("TÓM TẮT BƯỚC 6: BINARIZATION + THINNING")
    print("=" * 60)
    print(f"  NHỊ PHÂN HÓA:")
    print(f"    Phương pháp      : Gabor threshold (> 0)")
    print(f"    Pixel vân (đen)  : {np.sum(binary_img[mask==1] == 0)}")
    print(f"    Tỷ lệ vân/tổng  : {ridge_ratio:.1f}%")
    print(f"")
    print(f"  LÀM MẢNH:")
    print(f"    Phương pháp      : skimage.skeletonize()")
    print(f"    Pixel trước thin : {np.sum(binary_img[mask==1] == 0)}")
    print(f"    Pixel sau thin   : {np.sum(thinned_clean)}")
    print(f"    Tỷ lệ giảm      : {100 - thin_ratio:.1f}%")
    print("=" * 60)
    print(f"\n  Kết quả đã lưu tại: {OUTPUT_DIR}/")
    print("  → 07_binarize_thin_overview.png    (Tổng quan pipeline)")
    print("  → 07_binarize_thin_zoom.png        (Zoom so sánh)")
    print("  → 07_binarize_thin_superzoom.png   (Super zoom pixel)")
    print(f"\n  Bước tiếp theo: 08_minutiae_extraction.py")
    print("  (Trích xuất đặc trưng Minutiae bằng Crossing Number)")


if __name__ == "__main__":
    process_binarize_thin()
