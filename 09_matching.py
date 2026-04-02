"""
Bước 8: Fingerprint Matching (So khớp Vân tay)
=================================================
Mục tiêu: So sánh 2 tập minutiae từ 2 ảnh vân tay khác nhau
           → Cho ra điểm tương đồng (Similarity Score) từ 0 đến 1.

Thuật toán (tương đương match.m + score.m + transform.m + transform2.m):
  ┌──────────────────────────────────────────────────────────────────────┐
  │ BƯỚC 1 - Transform (Biến đổi tọa độ):                              │
  │   Chọn 1 minutia làm "gốc tọa độ" (reference point).               │
  │   Tịnh tiến tất cả minutiae khác về hệ tọa độ mới:                 │
  │     x' = (x - x_ref)×cos(θ_ref) + (y - y_ref)×sin(θ_ref)          │
  │     y' = -(x - x_ref)×sin(θ_ref) + (y - y_ref)×cos(θ_ref)         │
  │     θ' = θ - θ_ref                                                  │
  │                                                                      │
  │   → Giúp loại bỏ sự khác biệt do dịch chuyển và xoay giữa          │
  │     2 lần quét vân tay khác nhau.                                    │
  │                                                                      │
  │ BƯỚC 2 - Transform2 (Xoay thêm góc alpha):                         │
  │   Thử xoay thêm ±5° (11 góc) để bù sai số nhỏ trong orientation.   │
  │                                                                      │
  │ BƯỚC 3 - Score (Tính điểm):                                         │
  │   Đếm số cặp minutiae "khớp nhau":                                  │
  │     - Khoảng cách Euclid < 15 pixel                                 │
  │     - Chênh lệch góc < 14°                                          │
  │   Similarity = √(n² / (N1 × N2))                                    │
  │     n = số cặp khớp, N1/N2 = tổng minutiae mỗi ảnh                │
  │                                                                      │
  │ BƯỚC 4 - Brute-force (Thử mọi cặp reference):                      │
  │   Thử TẤT CẢ cặp (minutia_i từ ảnh 1, minutia_j từ ảnh 2)         │
  │   làm reference point → giữ lại cặp cho score CAO NHẤT.            │
  └──────────────────────────────────────────────────────────────────────┘

Ngưỡng phân loại (từ main_total.m):
  Score > 0.48 → CÙNG người (Match)
  Score ≤ 0.48 → KHÁC người (Non-match)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ============================================================================
# CẤU HÌNH
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "FVC2002", "DB1_B")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MATCH_THRESHOLD = 0.48  # Ngưỡng match (giống MATLAB: find(S>0.48))

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
step08 = _import_module("s08", os.path.join(BASE_DIR, "08_minutiae_extraction.py"))


# ============================================================================
# PIPELINE: TRÍCH XUẤT MINUTIAE TỪ 1 ẢNH
# ============================================================================
def extract_features(img_path):
    """
    Chạy toàn bộ pipeline từ ảnh gốc → minutiae.
    Tương đương: ext_finger(img) trong MATLAB.

    Returns:
      minutiae: Array [x, y, type, angle] hoặc None nếu lỗi
      img:      Ảnh gốc grayscale
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None

    # Enhancement
    enhanced, mask, _ = step03.full_enhancement_pipeline(
        img, clip_limit=2.5, grid_size=(8, 8),
        block_size=16, var_threshold=0.005
    )

    # Orientation
    orient_img, reliability = step04.estimate_orientation(enhanced)

    # Frequency
    freq_img, median_freq = step05.ridge_frequency(
        enhanced, mask, orient_img,
        block_size=32, wind_size=5,
        min_wave_length=5, max_wave_length=15
    )
    freq_median = np.where(mask == 1, median_freq, 0).astype(np.float64)

    # Gabor
    gabor_result = step06.gabor_ridge_filter(
        enhanced, orient_img, freq_median, mask,
        kx=0.5, ky=0.5, angle_inc=3
    )

    # Binarize + Thin
    binary = step07.binarize_fingerprint(gabor_result, mask)
    thinned = step07.thin_fingerprint(binary, mask)
    thinned = step07.clean_thinned_image(thinned, mask)

    # Minutiae
    cn_map = step08.compute_crossing_number(thinned)
    minutiae_raw = step08.extract_minutiae(thinned, cn_map, orient_img, mask)
    minutiae = step08.remove_false_minutiae(minutiae_raw, mask, dist_threshold=10)

    return minutiae, img


# ============================================================================
# BƯỚC 8A: TRANSFORM - Biến đổi tọa độ
# ============================================================================
def transform_minutiae(minutiae, ref_index):
    """
    Biến đổi tọa độ minutiae lấy minutia[ref_index] làm gốc.
    Tương đương: transform.m trong MATLAB.

    Mục đích:
      Đưa tất cả minutiae về HỆ TỌA ĐỘ TƯƠNG ĐỐI so với 1 minutia tham chiếu.
      → Loại bỏ ảnh hưởng của vị trí đặt ngón tay trên cảm biến.

    Ma trận xoay R (2D rotation):
      R = [cos(θ)   sin(θ)  0]     [x - x_ref]     [x']
          [-sin(θ)  cos(θ)  0]  ×  [y - y_ref]  =  [y']
          [0        0       1]     [θ - θ_ref]     [θ']

    Returns:
      T: Array [x', y', theta', type] cho mỗi minutia
    """
    n = len(minutiae)
    T = np.zeros((n, 4))

    x_ref = minutiae[ref_index, 0]
    y_ref = minutiae[ref_index, 1]
    th_ref = minutiae[ref_index, 3]

    cos_th = np.cos(th_ref)
    sin_th = np.sin(th_ref)

    # Ma trận xoay 3×3
    R = np.array([
        [cos_th, sin_th, 0],
        [-sin_th, cos_th, 0],
        [0, 0, 1]
    ])

    for i in range(n):
        dx = minutiae[i, 0] - x_ref
        dy = minutiae[i, 1] - y_ref
        dth = minutiae[i, 3] - th_ref
        B = np.array([dx, dy, dth])

        result = R @ B
        T[i, 0] = result[0]  # x'
        T[i, 1] = result[1]  # y'
        T[i, 2] = result[2]  # theta'
        T[i, 3] = minutiae[i, 2]  # type (giữ nguyên)

    return T


# ============================================================================
# BƯỚC 8B: TRANSFORM2 - Xoay thêm góc alpha
# ============================================================================
def transform2_minutiae(T, alpha):
    """
    Xoay thêm tập minutiae đã transform một góc alpha.
    Tương đương: transform2.m trong MATLAB.

    Mục đích:
      Bù sai số nhỏ trong orientation estimation giữa 2 ảnh.
      Thử alpha = -5°, -4°, ..., 0°, ..., +4°, +5° (11 giá trị).

    Returns:
      T_new: Array [x', y', theta', type] đã xoay thêm alpha
    """
    n = len(T)
    T_new = np.zeros((n, 4))

    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    # Ma trận xoay 4×4
    R = np.array([
        [cos_a, sin_a, 0, 0],
        [-sin_a, cos_a, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    for i in range(n):
        B = T[i].copy()
        B[2] -= alpha  # Trừ alpha khỏi theta trước khi xoay
        T_new[i] = R @ B

    return T_new


# ============================================================================
# BƯỚC 8C: SCORE - Tính điểm tương đồng
# ============================================================================
def compute_score(T1, T2, dist_threshold=15, angle_threshold=14):
    """
    Tính điểm tương đồng giữa 2 tập minutiae đã transform.
    Tương đương: score.m trong MATLAB.

    Thuật toán:
      Với mỗi minutia trong T1, tìm minutia gần nhất trong T2 thỏa:
        1. Khoảng cách Euclid < dist_threshold (15 pixel)
        2. Chênh lệch góc < angle_threshold (14°)
      Nếu cả 2 điều kiện thỏa → cặp này "khớp" → n += 1

    Similarity = √(n² / (N1 × N2))
      Giả sử N1=N2=n (khớp hoàn hảo) → S = √(n²/n²) = 1.0
      Giả sử n=0 (không khớp) → S = 0.0

    Returns:
      score: Giá trị 0.0 đến 1.0
    """
    n1 = len(T1)
    n2 = len(T2)

    if n1 == 0 or n2 == 0:
        return 0.0

    matched = 0

    for i in range(n1):
        for j in range(n2):
            # Khoảng cách Euclid
            dx = T1[i, 0] - T2[j, 0]
            dy = T1[i, 1] - T2[j, 1]
            dist = np.sqrt(dx ** 2 + dy ** 2)

            if dist < dist_threshold:
                # Chênh lệch góc (xử lý wraparound)
                d_theta = abs(T1[i, 2] - T2[j, 2]) * 180 / np.pi
                d_theta = min(d_theta, 360 - d_theta)

                if d_theta < angle_threshold:
                    matched += 1
                    break  # Mỗi minutia T1[i] chỉ khớp 1 lần

    score = np.sqrt(matched ** 2 / (n1 * n2))
    return score


# ============================================================================
# BƯỚC 8D: MATCH - So khớp 2 tập minutiae
# ============================================================================
def match_fingerprints(M1, M2, alpha_range=5):
    """
    So khớp 2 tập minutiae. Tương đương: match.m trong MATLAB.

    Thuật toán brute-force:
      1. Với mỗi minutia i trong M1:
         → Transform M1 lấy i làm gốc
         2. Với mỗi minutia j trong M2 (cùng type với i):
            → Transform M2 lấy j làm gốc
            3. Với mỗi góc alpha trong [-5°, +5°]:
               → Transform2 M2 thêm alpha
               → Tính score
               → Giữ lại score cao nhất

    Tham số:
      M1, M2:      Minutiae arrays [x, y, type, angle]
      alpha_range:  Dải góc xoay bổ sung (±5° = 11 góc thử)

    Returns:
      best_score:    Điểm tương đồng cao nhất (0-1)
      best_i, best_j: Index cặp reference tốt nhất
      best_alpha:    Góc xoay bổ sung tốt nhất
    """
    if len(M1) == 0 or len(M2) == 0:
        return 0.0, -1, -1, 0

    # Chỉ giữ minutiae loại Termination (1) và Bifurcation (3)
    # Giống MATLAB: M1=M1(M1(:,3)<5,:)
    M1 = M1[M1[:, 2] < 5]
    M2 = M2[M2[:, 2] < 5]

    n1 = len(M1)
    n2 = len(M2)

    best_score = 0.0
    best_i = 0
    best_j = 0
    best_alpha = 0

    for i in range(n1):
        T1 = transform_minutiae(M1, i)
        for j in range(n2):
            # Chỉ thử cặp cùng type (giống MATLAB: if M1(i,3)==M2(j,3))
            if M1[i, 2] != M2[j, 2]:
                continue

            T2 = transform_minutiae(M2, j)

            for a in range(-alpha_range, alpha_range + 1):
                alpha = a * np.pi / 180  # Chuyển sang radian
                T3 = transform2_minutiae(T2, alpha)
                sm = compute_score(T1, T3)

                if sm > best_score:
                    best_score = sm
                    best_i = i
                    best_j = j
                    best_alpha = a

    return best_score, best_i, best_j, best_alpha


# ============================================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================================
def process_matching():
    print("=" * 60)
    print("  FINGERPRINT MATCHING - So khớp Vân tay")
    print("  Tương đương: main_single.m + match.m + score.m")
    print("=" * 60)

    # === Chọn 2 ảnh để so khớp ===
    # Test 1: Cùng ngón tay (101_1 vs 101_2) → Expect: MATCH (score > 0.48)
    # Test 2: Khác ngón tay (101_1 vs 102_1) → Expect: NON-MATCH (score < 0.48)
    test_pairs = [
        ("101_1.tif", "101_2.tif", "CÙNG ngón tay (101)"),
        ("101_1.tif", "102_1.tif", "KHÁC ngón tay (101 vs 102)"),
        ("101_1.tif", "101_3.tif", "CÙNG ngón tay (101)"),
    ]

    results = []

    for idx, (file1, file2, description) in enumerate(test_pairs):
        print(f"\n{'─' * 60}")
        print(f"TEST {idx + 1}: {file1} vs {file2}")
        print(f"  Mô tả: {description}")
        print(f"{'─' * 60}")

        path1 = os.path.join(DATASET_PATH, file1)
        path2 = os.path.join(DATASET_PATH, file2)

        if not os.path.exists(path1) or not os.path.exists(path2):
            print(f"  Lỗi: Không tìm thấy file ảnh!")
            continue

        # Trích xuất minutiae từ ảnh 1
        print(f"  Trích xuất đặc trưng từ {file1}...")
        t1 = time.time()
        M1, img1 = extract_features(path1)
        t_ext1 = time.time() - t1
        if M1 is None or len(M1) == 0:
            print(f"  Lỗi: Không trích xuất được minutiae từ {file1}")
            continue
        print(f"    → {len(M1)} minutiae ({t_ext1:.1f}s)")

        # Trích xuất minutiae từ ảnh 2
        print(f"  Trích xuất đặc trưng từ {file2}...")
        t2 = time.time()
        M2, img2 = extract_features(path2)
        t_ext2 = time.time() - t2
        if M2 is None or len(M2) == 0:
            print(f"  Lỗi: Không trích xuất được minutiae từ {file2}")
            continue
        print(f"    → {len(M2)} minutiae ({t_ext2:.1f}s)")

        # So khớp
        print(f"  Đang so khớp (brute-force)...")
        t3 = time.time()
        score, bi, bj, ba = match_fingerprints(M1, M2, alpha_range=5)
        t_match = time.time() - t3

        is_match = score > MATCH_THRESHOLD
        verdict = "★ MATCH ★" if is_match else "✗ NON-MATCH"

        print(f"\n  ┌─────────────────────────────────────┐")
        print(f"  │  Score: {score:.4f}  →  {verdict:^15s}   │")
        print(f"  │  Ngưỡng: {MATCH_THRESHOLD}                       │")
        print(f"  │  Best ref: M1[{bi}] ↔ M2[{bj}], α={ba}°   │")
        print(f"  │  Thời gian match: {t_match:.1f}s              │")
        print(f"  └─────────────────────────────────────┘")

        results.append({
            'file1': file1, 'file2': file2, 'desc': description,
            'score': score, 'match': is_match, 'verdict': verdict,
            'M1': M1, 'M2': M2, 'img1': img1, 'img2': img2,
            'bi': bi, 'bj': bj, 'ba': ba,
            't_ext1': t_ext1, 't_ext2': t_ext2, 't_match': t_match,
        })

    # =========================================================================
    # TRỰC QUAN HÓA KẾT QUẢ
    # =========================================================================
    if not results:
        print("Không có kết quả nào để hiển thị.")
        return

    # HÌNH 1: Tổng quan kết quả matching cho mỗi cặp
    n_tests = len(results)
    fig, axes = plt.subplots(n_tests, 3, figsize=(15, 5 * n_tests))
    if n_tests == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Fingerprint Matching Results", fontsize=16, fontweight='bold')

    for idx, res in enumerate(results):
        # Ảnh 1 với minutiae
        vis1 = cv2.cvtColor(res['img1'], cv2.COLOR_GRAY2BGR)
        for m in res['M1']:
            x, y, t = int(m[0]), int(m[1]), int(m[2])
            color = (0, 0, 255) if t == 1 else (255, 0, 0)
            cv2.circle(vis1, (x, y), 3, color, 1, cv2.LINE_AA)

        axes[idx, 0].imshow(cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f"{res['file1']}\n({len(res['M1'])} minutiae)")
        axes[idx, 0].axis('off')

        # Ảnh 2 với minutiae
        vis2 = cv2.cvtColor(res['img2'], cv2.COLOR_GRAY2BGR)
        for m in res['M2']:
            x, y, t = int(m[0]), int(m[1]), int(m[2])
            color = (0, 0, 255) if t == 1 else (255, 0, 0)
            cv2.circle(vis2, (x, y), 3, color, 1, cv2.LINE_AA)

        axes[idx, 1].imshow(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
        axes[idx, 1].set_title(f"{res['file2']}\n({len(res['M2'])} minutiae)")
        axes[idx, 1].axis('off')

        # Kết quả
        axes[idx, 2].axis('off')
        bg_color = '#d4edda' if res['match'] else '#f8d7da'
        text_color = '#155724' if res['match'] else '#721c24'
        result_text = (
            f"{res['verdict']}\n\n"
            f"Score: {res['score']:.4f}\n"
            f"Threshold: {MATCH_THRESHOLD}\n\n"
            f"{res['desc']}"
        )
        axes[idx, 2].text(0.5, 0.5, result_text,
                          transform=axes[idx, 2].transAxes,
                          fontsize=13, ha='center', va='center',
                          color=text_color, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.8',
                                    facecolor=bg_color, alpha=0.9))

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "09_matching_results.png")
    plt.savefig(path1, dpi=200, bbox_inches='tight')
    print(f"\nĐã lưu: {path1}")

    # HÌNH 2: Bảng tổng kết
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 3 + n_tests))
    ax2.axis('off')
    fig2.suptitle("Fingerprint Matching - Summary", fontsize=16, fontweight='bold')

    table_data = []
    for res in results:
        table_data.append([
            res['file1'],
            res['file2'],
            f"{res['score']:.4f}",
            res['verdict'],
            f"{res['t_ext1']:.1f}s",
            f"{res['t_ext2']:.1f}s",
            f"{res['t_match']:.1f}s",
        ])

    col_labels = ['Image 1', 'Image 2', 'Score', 'Result',
                   'Extract 1', 'Extract 2', 'Match Time']

    table = ax2.table(cellText=table_data, colLabels=col_labels,
                       loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Tô màu header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#343a40')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Tô màu kết quả
    for i, res in enumerate(results):
        row = i + 1
        if res['match']:
            table[row, 3].set_facecolor('#d4edda')
        else:
            table[row, 3].set_facecolor('#f8d7da')

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "09_matching_summary.png")
    plt.savefig(path2, dpi=200, bbox_inches='tight')
    print(f"Đã lưu: {path2}")

    # =========================================================================
    # TÓM TẮT CUỐI CÙNG
    # =========================================================================
    print("\n" + "=" * 60)
    print("  ★★★ PIPELINE HOÀN TẤT ★★★")
    print("=" * 60)
    print(f"\n  TOÀN BỘ QUY TRÌNH NHẬN DẠNG VÂN TAY:")
    print(f"  ┌──────────────────────────────────────┐")
    print(f"  │ 01. Visualize         (Xem ảnh)      │")
    print(f"  │ 02. Preprocessing     (Tiền xử lý)   │")
    print(f"  │ 03. Enhancement       (Tăng cường)    │")
    print(f"  │ 04. Orientation Field (Hướng vân)     │")
    print(f"  │ 05. Frequency Est.    (Tần số vân)    │")
    print(f"  │ 06. Gabor Filtering   (Lọc Gabor)     │")
    print(f"  │ 07. Binarize + Thin   (Nhị phân+Mảnh)│")
    print(f"  │ 08. Minutiae Extract  (Trích Minutiae)│")
    print(f"  │ 09. Matching          (So khớp)  ★    │")
    print(f"  └──────────────────────────────────────┘")
    print(f"\n  KẾT QUẢ SO KHỚP:")
    for res in results:
        icon = "✓" if res['match'] else "✗"
        print(f"    {icon} {res['file1']} vs {res['file2']}: "
              f"Score={res['score']:.4f} → {res['verdict']}")
    print(f"\n  Ngưỡng phân loại: {MATCH_THRESHOLD}")
    print(f"  (Score > {MATCH_THRESHOLD} = MATCH, Score ≤ {MATCH_THRESHOLD} = NON-MATCH)")
    print("=" * 60)


if __name__ == "__main__":
    process_matching()
