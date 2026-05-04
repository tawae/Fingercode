"""
Bước 12: Đánh giá Hệ thống (ROC, FAR/FRR, EER)
=================================================
Mục tiêu:
  1. Tạo cặp Genuine (cùng người) và Impostor (khác người)
  2. Tính FAR / FRR tại nhiều ngưỡng (threshold)
  3. Vẽ đường cong ROC và FAR/FRR
  4. Tìm EER (Equal Error Rate)
  5. Lưu kết quả đánh giá ra file

Chiến lược tạo cặp:
  - Genuine: Ảnh Real (trong DB) vs Ảnh Altered-Easy (cùng người, cùng ngón)
  - Impostor: Ảnh Real (trong DB) vs Ảnh Altered-Easy (khác người)
"""

import os
import re
import glob
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import config

# ============================================================================
# CẤU HÌNH
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_REAL_DIR    = config.DATASET_PATH  # SOCOFing/Real
DATASET_ALTERED_DIR = os.path.join(BASE_DIR, "SOCOFing", "Altered", "Altered-Easy")
OUTPUT_DIR = config.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# IMPORT
# ============================================================================
from importlib.util import spec_from_file_location, module_from_spec

def _import_module(name, filepath):
    spec = spec_from_file_location(name, filepath)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Dùng Fingercode (step08) nhất quán với hệ thống DB
step08 = _import_module("s08", os.path.join(BASE_DIR, "08_fingercode_extraction.py"))
extract_features = step08.extract_features


# ============================================================================
# PARSE FILENAME
# ============================================================================
def parse_filename(filename):
    """Parse SOCOFing filename → (person_id, gender, hand, finger)"""
    m = re.match(r"(\d+)__([MF])_([A-Za-z]+)_([A-Za-z]+)_finger", filename)
    if m:
        return m.group(1), m.group(2), m.group(3), m.group(4)
    return None, None, None, None


# ============================================================================
# TẠO CẶP GENUINE VÀ IMPOSTOR
# ============================================================================
def build_pairs():
    """
    Tạo danh sách cặp Genuine và Impostor.
    Genuine: TOÀN BỘ cặp Real[person_X, finger_Y] vs Altered[person_X, finger_Y]
    Impostor: Random cặp khác người, số lượng = số genuine (cân bằng thống kê)
    """
    print("Đang quét thư mục Real và Altered-Easy...")

    # Index ảnh Real theo (person_id, finger_key)
    real_files = glob.glob(os.path.join(DATASET_REAL_DIR, "*.BMP"))
    real_index = {}
    for fp in real_files:
        pid, _, hand, finger = parse_filename(os.path.basename(fp))
        if pid:
            real_index[(pid, f"{hand}_{finger}")] = fp

    # Index ảnh Altered theo (person_id, finger_key) → lấy 1 biến thể (_CR)
    altered_files = glob.glob(os.path.join(DATASET_ALTERED_DIR, "*_CR.BMP"))
    altered_index = {}
    for fp in altered_files:
        pid, _, hand, finger = parse_filename(os.path.basename(fp))
        if pid:
            altered_index[(pid, f"{hand}_{finger}")] = fp

    # Lọc chỉ giữ user_id 1-50 (50 người đầu, mỗi người 10 ảnh)
    MAX_USER_ID_EVAL = 50
    real_index    = {k: v for k, v in real_index.items()    if int(k[0]) <= MAX_USER_ID_EVAL}
    altered_index = {k: v for k, v in altered_index.items() if int(k[0]) <= MAX_USER_ID_EVAL}

    # Tìm giao giữa Real và Altered
    common_keys = list(set(real_index.keys()) & set(altered_index.keys()))
    random.seed(42)
    random.shuffle(common_keys)

    print(f"  Tìm thấy {len(common_keys)} cặp Real↔Altered có chung (person, finger)")

    # --- Genuine pairs: lấy TẤT CẢ (không giới hạn) ---
    genuine_pairs = []
    for key in common_keys:
        genuine_pairs.append((real_index[key], altered_index[key], "genuine"))

    # --- Impostor pairs: số lượng = genuine, đảm bảo cân bằng thống kê ---
    num_impostor_target = len(genuine_pairs)
    impostor_pairs = []
    count = 0
    attempts = 0
    while count < num_impostor_target and attempts < num_impostor_target * 10:
        attempts += 1
        k1 = random.choice(common_keys)
        k2 = random.choice(common_keys)
        if k1[0] != k2[0]:  # Khác person
            impostor_pairs.append((real_index[k1], altered_index[k2], "impostor"))
            count += 1

    print(f"  Genuine pairs: {len(genuine_pairs)}")
    print(f"  Impostor pairs: {len(impostor_pairs)}")
    return genuine_pairs, impostor_pairs


# ============================================================================
# TÍNH SCORE CHO CÁC CẶP
# ============================================================================
def compute_scores(pairs, label):
    """Trích xuất Fingercode + tính similarity L2 (cùng công thức với DB search_top_k)."""
    scores = []
    errors = 0

    for real_path, altered_path, _ in tqdm(pairs, desc=f"Matching {label}", unit="pair"):
        try:
            v1, _ = extract_features(real_path)
            v2, _ = extract_features(altered_path)

            if v1 is None or v2 is None or len(v1) == 0 or len(v2) == 0:
                errors += 1
                continue

            # score = 1/(1+L2), nhất quán với similarity trong hệ thống FAISS
            dist = float(np.linalg.norm(
                np.array(v1, dtype=np.float32) - np.array(v2, dtype=np.float32)
            ))
            score = 1.0 / (1.0 + dist)
            scores.append(score)
        except Exception:
            errors += 1

    print(f"  {label}: {len(scores)} scores tính được, {errors} lỗi bỏ qua")
    return np.array(scores)


# ============================================================================
# TÍNH FAR / FRR
# ============================================================================
def compute_far_frr(genuine_scores, impostor_scores, thresholds):
    """
    FAR(t) = Tỷ lệ impostor bị chấp nhận nhầm (score > t)
    FRR(t) = Tỷ lệ genuine bị từ chối nhầm (score <= t)
    """
    far_list = []
    frr_list = []

    for t in thresholds:
        far = np.sum(impostor_scores > t) / len(impostor_scores) if len(impostor_scores) > 0 else 0
        frr = np.sum(genuine_scores <= t) / len(genuine_scores) if len(genuine_scores) > 0 else 0
        far_list.append(far)
        frr_list.append(frr)

    return np.array(far_list), np.array(frr_list)


def find_eer(far, frr, thresholds):
    """Tìm EER: điểm giao giữa FAR và FRR."""
    diff = np.abs(far - frr)
    idx = np.argmin(diff)
    eer = (far[idx] + frr[idx]) / 2
    return eer, thresholds[idx]


# ============================================================================
# VẼ BIỂU ĐỒ
# ============================================================================
def plot_evaluation(genuine_scores, impostor_scores, far, frr, thresholds, eer, eer_threshold):
    """Vẽ 4 biểu đồ đánh giá và lưu file."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Fingerprint System Evaluation Report", fontsize=16, fontweight='bold')

    # --- 1. Score Distribution ---
    ax = axes[0, 0]
    ax.hist(genuine_scores, bins=30, alpha=0.7, color='green', label=f'Genuine (n={len(genuine_scores)})', density=True)
    ax.hist(impostor_scores, bins=30, alpha=0.7, color='red', label=f'Impostor (n={len(impostor_scores)})', density=True)
    ax.axvline(x=eer_threshold, color='blue', linestyle='--', label=f'EER Threshold={eer_threshold:.3f}')
    ax.set_xlabel('Matching Score')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution (Genuine vs Impostor)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 2. FAR / FRR Curve ---
    ax = axes[0, 1]
    ax.plot(thresholds, far, 'r-', linewidth=2, label='FAR (False Accept Rate)')
    ax.plot(thresholds, frr, 'b-', linewidth=2, label='FRR (False Reject Rate)')
    ax.axvline(x=eer_threshold, color='gray', linestyle='--', alpha=0.7)
    ax.plot(eer_threshold, eer, 'ko', markersize=10, label=f'EER = {eer:.4f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Error Rate')
    ax.set_title('FAR / FRR vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 3. ROC Curve ---
    ax = axes[1, 0]
    tpr = 1 - frr  # True Positive Rate = 1 - FRR
    fpr = far       # False Positive Rate = FAR
    # Tính AUC (diện tích dưới đường cong)
    sorted_idx = np.argsort(fpr)
    auc = np.trapezoid(tpr[sorted_idx], fpr[sorted_idx])

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate (FAR)')
    ax.set_ylabel('True Positive Rate (1 - FRR)')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- 4. Summary Text ---
    ax = axes[1, 1]
    ax.axis('off')
    summary = (
        f"═══════════════════════════════════════\n"
        f"  EVALUATION SUMMARY\n"
        f"═══════════════════════════════════════\n\n"
        f"  Genuine Pairs : {len(genuine_scores)}\n"
        f"  Impostor Pairs: {len(impostor_scores)}\n\n"
        f"  Genuine Score  (mean): {np.mean(genuine_scores):.4f}\n"
        f"  Genuine Score  (std) : {np.std(genuine_scores):.4f}\n"
        f"  Impostor Score (mean): {np.mean(impostor_scores):.4f}\n"
        f"  Impostor Score (std) : {np.std(impostor_scores):.4f}\n\n"
        f"  EER           : {eer:.4f} ({eer*100:.2f}%)\n"
        f"  EER Threshold : {eer_threshold:.4f}\n"
        f"  AUC (ROC)     : {auc:.4f}\n\n"
        f"  FAR@threshold=0.40: {far[np.argmin(np.abs(thresholds - 0.40))]:.4f}\n"
        f"  FRR@threshold=0.40: {frr[np.argmin(np.abs(thresholds - 0.40))]:.4f}\n"
        f"  FAR@threshold=0.48: {far[np.argmin(np.abs(thresholds - 0.48))]:.4f}\n"
        f"  FRR@threshold=0.48: {frr[np.argmin(np.abs(thresholds - 0.48))]:.4f}\n"
        f"═══════════════════════════════════════"
    )
    ax.text(0.05, 0.5, summary, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "12_evaluation_report.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nĐã lưu biểu đồ: {save_path}")
    plt.close()

    return auc


# ============================================================================
# LƯU KẾT QUẢ ĐÁNH GIÁ RA FILE TXT
# ============================================================================
def save_report(genuine_scores, impostor_scores, far, frr, thresholds, eer, eer_threshold, auc):
    """Lưu báo cáo đánh giá chi tiết ra file text."""
    report_path = os.path.join(OUTPUT_DIR, "12_evaluation_results.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  FINGERPRINT SYSTEM - EVALUATION REPORT\n")
        f.write(f"  Ngày: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. CẤU HÌNH ĐÁNH GIÁ\n")
        f.write(f"   Gallery     : Real images (SOCOFing)\n")
        f.write(f"   Probe       : Altered-Easy (biến thể _CR)\n")
        f.write(f"   Matching    : KD-Tree + Poincare alignment\n")
        f.write(f"   Genuine Pairs : {len(genuine_scores)}\n")
        f.write(f"   Impostor Pairs: {len(impostor_scores)}\n\n")

        f.write("2. PHÂN BỐ SCORE\n")
        f.write(f"   Genuine  — Mean: {np.mean(genuine_scores):.4f}, "
                f"Std: {np.std(genuine_scores):.4f}, "
                f"Min: {np.min(genuine_scores):.4f}, "
                f"Max: {np.max(genuine_scores):.4f}\n")
        f.write(f"   Impostor — Mean: {np.mean(impostor_scores):.4f}, "
                f"Std: {np.std(impostor_scores):.4f}, "
                f"Min: {np.min(impostor_scores):.4f}, "
                f"Max: {np.max(impostor_scores):.4f}\n\n")

        f.write("3. CHỈ SỐ HIỆU NĂNG\n")
        f.write(f"   EER           : {eer:.4f} ({eer*100:.2f}%)\n")
        f.write(f"   EER Threshold : {eer_threshold:.4f}\n")
        f.write(f"   AUC (ROC)     : {auc:.4f}\n\n")

        f.write("4. FAR/FRR TẠI CÁC NGƯỠNG QUAN TRỌNG\n")
        f.write(f"   {'Threshold':<12} {'FAR':<12} {'FRR':<12}\n")
        f.write(f"   {'-'*36}\n")
        for t_val in [0.20, 0.30, 0.40, 0.48, 0.50, 0.60, 0.70]:
            idx = np.argmin(np.abs(thresholds - t_val))
            f.write(f"   {t_val:<12.2f} {far[idx]:<12.4f} {frr[idx]:<12.4f}\n")

        f.write(f"\n5. BẢNG FAR/FRR CHI TIẾT (mỗi 0.05)\n")
        f.write(f"   {'Threshold':<12} {'FAR':<12} {'FRR':<12} {'TPR(1-FRR)':<12}\n")
        f.write(f"   {'-'*48}\n")
        for t_val in np.arange(0, 1.05, 0.05):
            idx = np.argmin(np.abs(thresholds - t_val))
            f.write(f"   {t_val:<12.2f} {far[idx]:<12.4f} {frr[idx]:<12.4f} {1-frr[idx]:<12.4f}\n")

    print(f"Đã lưu báo cáo: {report_path}")


# ============================================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  FINGERPRINT SYSTEM EVALUATION")
    print("  ROC Curve / FAR / FRR / EER")
    print("=" * 60)

    t_start = time.time()

    # 1. Tạo cặp
    genuine_pairs, impostor_pairs = build_pairs()

    # 2. Tính score
    print("\n[PHA 1] Tính score cho Genuine pairs...")
    genuine_scores = compute_scores(genuine_pairs, "Genuine")

    print("\n[PHA 2] Tính score cho Impostor pairs...")
    impostor_scores = compute_scores(impostor_pairs, "Impostor")

    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        print("Lỗi: Không đủ dữ liệu để đánh giá!")
        exit(1)

    # 3. Tính FAR / FRR
    thresholds = np.linspace(0, 1, 500)
    far, frr = compute_far_frr(genuine_scores, impostor_scores, thresholds)
    eer, eer_threshold = find_eer(far, frr, thresholds)

    print(f"\n{'='*40}")
    print(f"  EER = {eer:.4f} ({eer*100:.2f}%)")
    print(f"  EER Threshold = {eer_threshold:.4f}")
    print(f"{'='*40}")

    # 4. Vẽ biểu đồ
    auc = plot_evaluation(genuine_scores, impostor_scores, far, frr, thresholds, eer, eer_threshold)

    # 5. Lưu báo cáo
    save_report(genuine_scores, impostor_scores, far, frr, thresholds, eer, eer_threshold, auc)

    t_total = time.time() - t_start
    print(f"\nTổng thời gian đánh giá: {t_total:.1f}s")