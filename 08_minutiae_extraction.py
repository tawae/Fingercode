"""
Bước 7: Minutiae Extraction (Trích xuất đặc trưng Minutiae)
=============================================================
Mục tiêu: Tìm 2 loại đặc trưng chính trên ảnh vân tay đã làm mảnh:
  1. Termination (Điểm kết thúc): Nơi đường vân bị NGẮT, chỉ có 1 hướng đi tiếp
  2. Bifurcation (Điểm rẽ nhánh): Nơi 1 đường vân TÁCH thành 2 nhánh

Thuật toán Crossing Number (CN) - Raymond Thai:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Tại mỗi pixel VÂN (trắng) trong ảnh đã thin:                     │
  │                                                                     │
  │  1. Lấy 8 pixel lân cận theo thứ tự vòng tròn:                    │
  │                                                                     │
  │         P4 | P3 | P2                                                │
  │         P5 | PC | P1     (PC = pixel trung tâm)                    │
  │         P6 | P7 | P8                                                │
  │                                                                     │
  │  2. Tính CN = Σ |P(i) - P(i+1)| / 2   (với P9 = P1)               │
  │                                                                     │
  │  3. Phân loại:                                                      │
  │     CN = 0 → Pixel cô lập (isolated, nhiễu)                       │
  │     CN = 1 → TERMINATION (điểm kết thúc) ★                        │
  │     CN = 2 → Pixel bình thường (tiếp tục đường vân)               │
  │     CN = 3 → BIFURCATION (điểm rẽ nhánh) ★                        │
  │     CN > 3 → Crossing (giao điểm, thường là nhiễu)                │
  └─────────────────────────────────────────────────────────────────────┘

Hình dung CN:
  CN=1 (Termination):    CN=2 (Normal):     CN=3 (Bifurcation):
       · · ·                · ■ ·              · ■ ·
       · ■ ·                · ■ ·              · ■ ·
       · ■ ·                · ■ ·              ■ · ■
       (1 neighbor)        (2 neighbors)     (3 neighbors)

Lọc Minutiae giả (False Minutiae Removal):
  Tương đương dòng 342-425 trong ext_finger.m:
  1. Loại bỏ minutiae gần biên mask (vùng nền)
  2. Loại bỏ cặp minutiae quá gần nhau (khoảng cách < threshold)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
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
# IMPORT TỪ CÁC BƯỚC TRƯỚC
# ============================================================================
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
step07 = _import_module("s07", os.path.join(BASE_DIR, "07_binarize_thin.py"))


# ============================================================================
# BƯỚC 7A: CROSSING NUMBER
# ============================================================================
def compute_crossing_number(thinned):
    """
    Tính Crossing Number (CN) cho toàn bộ ảnh đã thin.
    Tương đương: phần "Finding Minutiae" (dòng 261-340) trong ext_finger.m

    8 pixel lân cận theo thứ tự vòng tròn (giống p.m trong MATLAB):
        P4 | P3 | P2
        P5 | PC | P1      →  Thứ tự: P1→P2→P3→P4→P5→P6→P7→P8→P1
        P6 | P7 | P8

    CN = (1/2) × Σ|P(i) - P(i+1)|   với i = 1..8, P9 = P1

    Returns:
      cn_map: Ma trận CN cùng kích thước ảnh
    """
    rows, cols = thinned.shape
    cn_map = np.zeros((rows, cols), dtype=np.int32)

    # Thinned image phải dạng 0/1 (int)
    t = thinned.astype(np.int32)

    # 8 neighbor offsets theo thứ tự vòng tròn giống p.m:
    # P1(0,+1), P2(-1,+1), P3(-1,0), P4(-1,-1), P5(0,-1), P6(+1,-1), P7(+1,0), P8(+1,+1)
    # (dy, dx) format
    neighbors = [
        (0, 1),    # P1: phải
        (-1, 1),   # P2: trên-phải
        (-1, 0),   # P3: trên
        (-1, -1),  # P4: trên-trái
        (0, -1),   # P5: trái
        (1, -1),   # P6: dưới-trái
        (1, 0),    # P7: dưới
        (1, 1),    # P8: dưới-phải
    ]

    # Duyệt qua tất cả pixel (tránh biên)
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if t[r, c] == 0:  # Chỉ tính CN cho pixel VÂN (trắng/True/1)
                continue

            # Lấy 8 giá trị lân cận
            p = []
            for dy, dx in neighbors:
                p.append(t[r + dy, c + dx])

            # CN = (1/2) × Σ|P(i) - P(i+1)|, P9 = P1
            cn = 0
            for i in range(8):
                cn += abs(p[i] - p[(i + 1) % 8])
            cn = cn // 2

            cn_map[r, c] = cn

    return cn_map


# ============================================================================
# BƯỚC 7B: TRÍCH XUẤT MINUTIAE
# ============================================================================
def extract_minutiae(thinned, cn_map, orient_img, mask, border_margin=20):
    """
    Trích xuất điểm Minutiae từ Crossing Number map.

    Phân loại:
      CN = 1 → Termination (điểm kết thúc)
      CN = 3 → Bifurcation (điểm rẽ nhánh)

    Mỗi minutia lưu: [x, y, type, angle]
      x, y:   Tọa độ pixel
      type:   1 = Termination, 3 = Bifurcation
      angle:  Hướng vân tại vị trí đó (lấy từ orientation image)

    Tham số:
      border_margin: Bỏ qua minutiae trong vùng biên (pixel).
                     Giống MATLAB: for y=20:size(img,1)-14

    Returns:
      minutiae: Danh sách [x, y, type, angle] cho mỗi minutia
    """
    rows, cols = thinned.shape
    minutiae = []

    for r in range(border_margin, rows - border_margin):
        for c in range(border_margin, cols - border_margin):
            if not thinned[r, c]:
                continue

            cn = cn_map[r, c]

            if cn == 1 or cn == 3:
                # Kiểm tra vùng lân cận có nằm trong mask không
                # Giống MATLAB dòng 280-289: kiểm tra mask trong vùng 11×11
                neighborhood = mask[max(r-5, 0):min(r+6, rows),
                                    max(c-5, 0):min(c+6, cols)]
                if np.any(neighborhood == 0):
                    continue  # Bỏ qua nếu gần biên nền

                # Lấy hướng vân (median của vùng 3×3)
                orient_patch = orient_img[max(r-1, 0):min(r+2, rows),
                                          max(c-1, 0):min(c+2, cols)]
                angle = np.median(orient_patch)

                minutiae.append([c, r, cn, angle])  # [x, y, type, angle]

    return np.array(minutiae) if minutiae else np.array([]).reshape(0, 4)


# ============================================================================
# BƯỚC 7C: LỌC MINUTIAE GIẢ (FALSE MINUTIAE REMOVAL)
# ============================================================================
def remove_false_minutiae(minutiae, mask, dist_threshold=10):
    """
    Loại bỏ Minutiae giả.
    Tương đương: "Filtering False Minutiae" (dòng 342-425) trong ext_finger.m.

    Bộ lọc 1 - Biên mask:
      Loại minutiae nằm gần biên vùng vân tay (lân cận 5×5 có pixel nền).
      → Đã thực hiện trong extract_minutiae()

    Bộ lọc 2 - Khoảng cách:
      Loại cặp minutiae quá gần nhau (< dist_threshold pixel).
      Giống MATLAB dòng 374-383: dist_test=49 (7²), tức khoảng cách < 7 pixel.
      
      Tại sao minutiae gần nhau thường là giả?
      → 2 minutiae quá gần thường do:
        - Đường vân bị đứt gãy cục bộ (tạo 2 termination giả đối diện)
        - Nhiễu thinning (tạo nhánh ngắn → bifurcation giả + termination giả)

    Bộ lọc 3 - Border cleanup:
      Loại minutiae nằm sát biên ảnh.

    Returns:
      filtered: Danh sách minutiae đã lọc
    """
    if len(minutiae) == 0:
        return minutiae

    # --- Bộ lọc 2: Khoảng cách ---
    coords = minutiae[:, :2].astype(np.float64)  # [x, y]
    dist_matrix = cdist(coords, coords, metric='euclidean')

    # Đánh dấu minutiae cần loại bỏ
    to_remove = set()
    n = len(minutiae)

    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            if dist_matrix[i, j] < dist_threshold:
                # Loại cả 2 nếu quá gần
                to_remove.add(i)
                to_remove.add(j)

    # Giữ lại minutiae không bị đánh dấu
    keep_indices = [i for i in range(n) if i not in to_remove]
    filtered = minutiae[keep_indices]

    return filtered


# ============================================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================================
def process_minutiae():
    img_path = os.path.join(DATASET_PATH, SAMPLE_IMAGE)
    if not os.path.exists(img_path):
        print(f"Lỗi: Không tìm thấy ảnh tại {img_path}")
        return

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Lỗi: Không đọc được ảnh.")
        return

    print(f"Đã đọc ảnh: {SAMPLE_IMAGE} | Kích thước: {img.shape}")

    # === Chạy toàn bộ pipeline trước đó ===
    print("[1/6] Enhancement...")
    enhanced_img, mask, _ = step03.full_enhancement_pipeline(
        img, clip_limit=2.5, grid_size=(8, 8),
        block_size=16, var_threshold=0.005
    )

    print("[2/6] Orientation Field...")
    orient_img, reliability = step04.estimate_orientation(enhanced_img)

    print("[3/6] Frequency Estimation...")
    freq_img, median_freq = step05.ridge_frequency(
        enhanced_img, mask, orient_img,
        block_size=32, wind_size=5,
        min_wave_length=5, max_wave_length=15
    )
    freq_img_median = np.where(mask == 1, median_freq, 0).astype(np.float64)

    print("[4/6] Gabor Filtering...")
    gabor_result = step06.gabor_ridge_filter(
        enhanced_img, orient_img, freq_img_median, mask,
        kx=0.5, ky=0.5, angle_inc=3
    )

    print("[5/6] Binarization + Thinning...")
    binary_img = step07.binarize_fingerprint(gabor_result, mask)
    thinned = step07.thin_fingerprint(binary_img, mask)
    thinned = step07.clean_thinned_image(thinned, mask)

    # === BƯỚC 7: Minutiae Extraction ===
    print("[6/6] Minutiae Extraction...")

    # 7A: Crossing Number
    print("  Đang tính Crossing Number...")
    cn_map = compute_crossing_number(thinned)

    # Thống kê CN
    cn_values, cn_counts = np.unique(cn_map[thinned], return_counts=True)
    print("  Phân bố CN trên skeleton:")
    for val, cnt in zip(cn_values, cn_counts):
        label = {0: "Isolated", 1: "Termination", 2: "Normal",
                 3: "Bifurcation"}.get(val, f"CN={val}")
        print(f"    CN={val} ({label}): {cnt} pixel")

    # 7B: Trích xuất
    print("  Đang trích xuất minutiae...")
    minutiae_raw = extract_minutiae(thinned, cn_map, orient_img, mask, border_margin=20)
    n_term_raw = np.sum(minutiae_raw[:, 2] == 1) if len(minutiae_raw) > 0 else 0
    n_bif_raw = np.sum(minutiae_raw[:, 2] == 3) if len(minutiae_raw) > 0 else 0
    print(f"  Trước lọc: {len(minutiae_raw)} minutiae "
          f"({n_term_raw} termination, {n_bif_raw} bifurcation)")

    # 7C: Lọc minutiae giả
    print("  Đang lọc minutiae giả...")
    minutiae_filtered = remove_false_minutiae(minutiae_raw, mask, dist_threshold=10)
    n_term = np.sum(minutiae_filtered[:, 2] == 1) if len(minutiae_filtered) > 0 else 0
    n_bif = np.sum(minutiae_filtered[:, 2] == 3) if len(minutiae_filtered) > 0 else 0
    print(f"  Sau lọc : {len(minutiae_filtered)} minutiae "
          f"({n_term} termination, {n_bif} bifurcation)")

    # =========================================================================
    # TRỰC QUAN HÓA
    # =========================================================================

    def draw_minutiae_on_image(bg_img, minutiae, arrow_len=12):
        """Vẽ minutiae lên ảnh: đỏ=termination, xanh=bifurcation, mũi tên=hướng."""
        vis = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR) if len(bg_img.shape) == 2 else bg_img.copy()
        for m in minutiae:
            x, y, mtype, angle = int(m[0]), int(m[1]), int(m[2]), m[3]
            if mtype == 1:  # Termination
                color = (0, 0, 255)     # Đỏ (BGR)
                marker_size = 4
            else:           # Bifurcation
                color = (255, 0, 0)     # Xanh dương (BGR)
                marker_size = 4

            # Vẽ vòng tròn đánh dấu
            cv2.circle(vis, (x, y), marker_size, color, 1, cv2.LINE_AA)

            # Vẽ mũi tên chỉ hướng
            dx = int(arrow_len * np.cos(angle))
            dy = int(arrow_len * np.sin(angle))
            cv2.arrowedLine(vis, (x, y), (x + dx, y + dy), color, 1, cv2.LINE_AA,
                            tipLength=0.3)

        return vis

    # Ảnh skeleton trên nền trắng
    skeleton_display = np.ones_like(img, dtype=np.uint8) * 255
    skeleton_display[thinned] = 0

    # HÌNH 1: Tổng quan
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Bước 7: Minutiae Extraction (Crossing Number)",
                 fontsize=16, fontweight='bold')

    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("1. Ảnh Gốc")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(binary_img, cmap='gray')
    axes[0, 1].set_title("2. Binary")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(skeleton_display, cmap='gray')
    axes[0, 2].set_title(f"3. Skeleton\n({np.sum(thinned)} pixel)")
    axes[0, 2].axis('off')

    # CN map (chỉ hiện trên skeleton)
    cn_display = np.zeros_like(img, dtype=np.float32)
    cn_display[thinned] = cn_map[thinned].astype(np.float32)
    axes[1, 0].imshow(cn_display, cmap='jet', vmin=0, vmax=4)
    axes[1, 0].set_title("4. Crossing Number Map\n(1=Term, 2=Normal, 3=Bif)")
    axes[1, 0].axis('off')

    # Minutiae trước lọc (trên skeleton)
    vis_raw = draw_minutiae_on_image(skeleton_display, minutiae_raw)
    axes[1, 1].imshow(cv2.cvtColor(vis_raw, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"5. Trước lọc: {len(minutiae_raw)} minutiae\n"
                         f"(Đỏ=Term, Xanh=Bif)")
    axes[1, 1].axis('off')

    # Minutiae sau lọc (trên ảnh gốc)
    vis_filtered = draw_minutiae_on_image(img, minutiae_filtered)
    axes[1, 2].imshow(cv2.cvtColor(vis_filtered, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f"6. Sau lọc: {len(minutiae_filtered)} minutiae\n"
                         f"({n_term} Term + {n_bif} Bif)")
    axes[1, 2].axis('off')

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "08_minutiae_overview.png")
    plt.savefig(path1, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {path1}")

    # HÌNH 2: Zoom minutiae trên ảnh gốc
    h, w = img.shape
    ys, ye = h // 4, 3 * h // 4
    xs, xe = w // 4, 3 * w // 4

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle("Zoom: Minutiae trên ảnh vân tay", fontsize=14, fontweight='bold')

    axes2[0].imshow(img[ys:ye, xs:xe], cmap='gray')
    axes2[0].set_title("Ảnh Gốc (Zoom)")
    axes2[0].axis('off')

    vis_zoom_raw = vis_raw[ys:ye, xs:xe]
    axes2[1].imshow(cv2.cvtColor(vis_zoom_raw, cv2.COLOR_BGR2RGB))
    axes2[1].set_title(f"Trước lọc (Zoom)")
    axes2[1].axis('off')

    vis_zoom_filt = vis_filtered[ys:ye, xs:xe]
    axes2[2].imshow(cv2.cvtColor(vis_zoom_filt, cv2.COLOR_BGR2RGB))
    axes2[2].set_title(f"Sau lọc (Zoom)")
    axes2[2].axis('off')

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "08_minutiae_zoom.png")
    plt.savefig(path2, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {path2}")

    # HÌNH 3: Legend + Thống kê
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle("Minutiae: Chú thích & Thống kê", fontsize=14, fontweight='bold')

    # Vẽ minutiae lớn trên ảnh gốc để dễ nhìn
    vis_final = draw_minutiae_on_image(img, minutiae_filtered, arrow_len=15)
    axes3[0].imshow(cv2.cvtColor(vis_final, cv2.COLOR_BGR2RGB))
    axes3[0].set_title("Kết quả cuối cùng")
    axes3[0].axis('off')

    # Thống kê
    axes3[1].axis('off')
    stats_text = (
        f"THỐNG KÊ MINUTIAE EXTRACTION\n"
        f"{'━' * 40}\n\n"
        f"  Ảnh             : {SAMPLE_IMAGE}\n"
        f"  Kích thước      : {img.shape[1]}×{img.shape[0]} pixel\n\n"
        f"  TRƯỚC LỌC:\n"
        f"  ├ Termination   : {n_term_raw}\n"
        f"  ├ Bifurcation   : {n_bif_raw}\n"
        f"  └ Tổng          : {len(minutiae_raw)}\n\n"
        f"  SAU LỌC:\n"
        f"  ├ Termination   : {n_term}  (●→  Đỏ)\n"
        f"  ├ Bifurcation   : {n_bif}  (●→  Xanh)\n"
        f"  └ Tổng          : {len(minutiae_filtered)}\n\n"
        f"  Đã loại bỏ      : {len(minutiae_raw) - len(minutiae_filtered)} minutiae giả\n"
        f"  Tỷ lệ giữ lại  : {len(minutiae_filtered)/max(len(minutiae_raw),1)*100:.1f}%\n\n"
        f"  THUẬT TOÁN:\n"
        f"  ├ Crossing Number (CN)\n"
        f"  ├ Border margin: 20 pixel\n"
        f"  └ Distance threshold: 10 pixel"
    )
    axes3[1].text(0.05, 0.95, stats_text, transform=axes3[1].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "08_minutiae_stats.png")
    plt.savefig(path3, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {path3}")

    # =========================================================================
    # TÓM TẮT
    # =========================================================================
    print("\n" + "=" * 60)
    print("TÓM TẮT BƯỚC 7: MINUTIAE EXTRACTION")
    print("=" * 60)
    print(f"  Thuật toán        : Crossing Number (Raymond Thai)")
    print(f"  Trước lọc         : {len(minutiae_raw)} minutiae")
    print(f"    - Termination   : {n_term_raw}")
    print(f"    - Bifurcation   : {n_bif_raw}")
    print(f"  Sau lọc           : {len(minutiae_filtered)} minutiae")
    print(f"    - Termination   : {n_term}")
    print(f"    - Bifurcation   : {n_bif}")
    print(f"  Đã loại bỏ        : {len(minutiae_raw) - len(minutiae_filtered)} giả")
    print("=" * 60)
    print(f"\n  Kết quả đã lưu tại: {OUTPUT_DIR}/")
    print("  → 08_minutiae_overview.png  (Tổng quan pipeline)")
    print("  → 08_minutiae_zoom.png      (Zoom chi tiết)")
    print("  → 08_minutiae_stats.png     (Thống kê + Chú thích)")
    print(f"\n  ★ PIPELINE TRÍCH XUẤT ĐẶC TRƯNG HOÀN TẤT! ★")
    print(f"  Bước tiếp theo: 09_matching.py")
    print(f"  (So khớp vân tay - Fingerprint Matching)")


if __name__ == "__main__":
    process_minutiae()
