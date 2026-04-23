"""
Bước 9: Fingercode Matching (1-vs-1 Verification)
=================================================
Mục tiêu: So sánh trực tiếp 2 ảnh vân tay bằng vector Fingercode.
Thay vì quét vét cạn O(N^2) các điểm minutiae như trước đây, 
ta chỉ việc tính khoảng cách (Euclidean Distance / L2) hoặc 
độ tương đồng Cosine Similarity giữa 2 vector có chiều cố định.

Ứng dụng: Xác thực (Verification 1-1) - Xác nhận 2 ảnh có phải 
của cùng một ngón tay hay không.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ============================================================================
# CẤU HÌNH
# ============================================================================
import config

BASE_DIR = config.BASE_DIR
DATASET_PATH = config.DATASET_PATH
OUTPUT_DIR = config.OUTPUT_DIR

# Ngưỡng phân loại khoảng cách (Tùy thuộc vào việc normalize vector)
L2_MATCH_THRESHOLD = 0.50 # Distance < Threshold => MATCH

# ============================================================================
# IMPORT TỪ CÁC BƯỚC TRƯỚC
# ============================================================================
from importlib.util import spec_from_file_location, module_from_spec
def _import_module(name, filepath):
    spec = spec_from_file_location(name, filepath)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

step08 = _import_module("s08", os.path.join(BASE_DIR, "08_fingercode_extraction.py"))
extract_features = step08.extract_features

# ============================================================================
# BƯỚC 9: KHOẢNG CÁCH VECTOR (VECTOR DISTANCE)
# ============================================================================
def calculate_distance(v1, v2):
    """Tính L2 distance (Euclidean) giữa 2 vector numpy"""
    return np.linalg.norm(v1 - v2)

# ============================================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================================
def process_matching():
    print("=" * 60)
    print("  FINGERCODE MATCHING - Xác Thực 1-vs-1 bằng Khoảng Cách Vector")
    print("=" * 60)

    # Test pairs: 1 Cùng người, 1 Khác người
    test_pairs = [
        ("101_1.tif", "101_2.tif", "CÙNG ngón tay (101)"),
        ("101_1.tif", "102_1.tif", "KHÁC ngón tay (101 vs 102)"),
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

        print(f"  Trích xuất Fingercode từ {file1}...")
        t1 = time.time()
        V1, img1 = extract_features(path1)
        t_ext1 = time.time() - t1
        
        if V1 is None or len(V1) == 0:
            print("  Lỗi trích xuất V1")
            continue

        print(f"  Trích xuất Fingercode từ {file2}...")
        t2 = time.time()
        V2, img2 = extract_features(path2)
        t_ext2 = time.time() - t2
        
        if V2 is None or len(V2) == 0:
            print("  Lỗi trích xuất V2")
            continue

        # So khớp bằng Euclidean Distance
        print(f"  Đang tính Vector Distance...")
        t3 = time.time()
        dist = calculate_distance(V1, V2)
        t_match = time.time() - t3

        is_match = dist < L2_MATCH_THRESHOLD
        verdict = "★ MATCH ★" if is_match else "✗ NON-MATCH"

        print(f"\n  ┌─────────────────────────────────────┐")
        print(f"  │  L2 Distance: {dist:.4f}  →  {verdict:^11s} │")
        print(f"  │  Threshold  : {L2_MATCH_THRESHOLD:.4f}                 │")
        print(f"  │  Match Time : {t_match*1000:.2f} ms             │")
        print(f"  └─────────────────────────────────────┘")

        results.append({
            'file1': file1, 'file2': file2, 'desc': description,
            'dist': dist, 'match': is_match, 'verdict': verdict,
            'img1': img1, 'img2': img2,
            't_ext1': t_ext1, 't_ext2': t_ext2, 't_match': t_match,
        })

    # =========================================================================
    # TRỰC QUAN HÓA KẾT QUẢ
    # =========================================================================
    if not results: return

    n_tests = len(results)
    fig, axes = plt.subplots(n_tests, 3, figsize=(15, 5 * n_tests))
    if n_tests == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Fingercode 1-vs-1 Matching Results", fontsize=16, fontweight='bold')

    for idx, res in enumerate(results):
        axes[idx, 0].imshow(res['img1'], cmap="gray")
        axes[idx, 0].set_title(f"{res['file1']}")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(res['img2'], cmap="gray")
        axes[idx, 1].set_title(f"{res['file2']}")
        axes[idx, 1].axis('off')

        axes[idx, 2].axis('off')
        bg_color = '#d4edda' if res['match'] else '#f8d7da'
        text_color = '#155724' if res['match'] else '#721c24'
        result_text = (
            f"{res['verdict']}\n\n"
            f"L2 Distance: {res['dist']:.4f}\n"
            f"Threshold: < {L2_MATCH_THRESHOLD}\n\n"
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
    print(f"\nĐã lưu visualization: {path1}")


if __name__ == "__main__":
    process_matching()
