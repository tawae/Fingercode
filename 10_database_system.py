"""
Bước 10: Hệ thống Cơ sở Dữ liệu Vân tay (Fingerprint Database System)
=========================================================================
Mục tiêu: Xây dựng hệ thống hoàn chỉnh với 2 pha:
  1. Enrollment (Đăng ký): Nạp ảnh → Trích xuất minutiae → Lưu vào DB
  2. Matching  (Nhận dạng): Ảnh truy vấn → Trích xuất → So khớp với DB → Trả kết quả

Thiết kế Database (SQLite):
  ┌─────────────────────────────────────────────────────────────────┐
  │ Bảng: Users                                                     │
  │ ├── user_id      INTEGER PRIMARY KEY AUTOINCREMENT              │
  │ ├── name         TEXT NOT NULL                                  │
  │ ├── role         TEXT DEFAULT 'Student'                         │
  │ └── created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP            │
  │                                                                 │
  │ Bảng: Fingerprint_Templates                                    │
  │ ├── template_id  INTEGER PRIMARY KEY AUTOINCREMENT              │
  │ ├── user_id      INTEGER FOREIGN KEY → Users(user_id)           │
  │ ├── finger_index TEXT (e.g. 'right_thumb')                      │
  │ ├── minutiae_data TEXT (JSON: [{x,y,type,angle}, ...])          │
  │ ├── minutiae_count INTEGER                                      │
  │ ├── source_image TEXT (đường dẫn ảnh gốc)                       │
  │ └── created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP            │
  │                                                                 │
  │ Quan hệ: Users (1) ──── (N) Fingerprint_Templates              │
  │ (1 người có thể đăng ký nhiều ngón tay)                        │
  └─────────────────────────────────────────────────────────────────┘

Lưu trữ minutiae: Dạng JSON (Cách 1)
  Ví dụ: [{"x": 120, "y": 95, "type": 1, "angle": 1.57}, ...]
  → SELECT 1 lần lấy hết template → Load vào numpy → Chạy matching
"""

import sqlite3
import json
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt

# ============================================================================
# CẤU HÌNH
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "FVC2002", "DB1_B")
DB_PATH = os.path.join(BASE_DIR, "fingerprint.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MATCH_THRESHOLD = 0.48

# ============================================================================
# IMPORT PIPELINE TỪ CÁC BƯỚC TRƯỚC
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
step09 = _import_module("s09", os.path.join(BASE_DIR, "09_matching.py"))

extract_features = step09.extract_features
match_fingerprints = step09.match_fingerprints


# ============================================================================
# LỚP DATABASE: Quản lý kết nối và thao tác SQL
# ============================================================================
class FingerprintDatabase:
    """
    Lớp quản lý cơ sở dữ liệu vân tay.
    Sử dụng SQLite - không cần cài server, dữ liệu lưu trong 1 file .db
    """

    def __init__(self, db_path):
        """Khởi tạo kết nối đến database."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self):
        """Mở kết nối đến database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Cho phép truy cập cột bằng tên
        self.cursor = self.conn.cursor()
        # Bật Foreign Key support (SQLite mặc định TẮT)
        self.cursor.execute("PRAGMA foreign_keys = ON")
        print(f"  Đã kết nối database: {self.db_path}")

    def close(self):
        """Đóng kết nối."""
        if self.conn:
            self.conn.close()
            print("  Đã đóng kết nối database.")

    # ────────────────────────────────────────────────────────────────────
    # TẠO BẢNG (CREATE)
    # ────────────────────────────────────────────────────────────────────
    def create_tables(self):
        """
        Tạo các bảng trong database nếu chưa tồn tại.
        """
        # Bảng Users: Thông tin người dùng
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS Users (
                user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                role        TEXT DEFAULT 'Student',
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Bảng Fingerprint_Templates: Dữ liệu sinh trắc
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS Fingerprint_Templates (
                template_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         INTEGER NOT NULL,
                finger_index    TEXT DEFAULT 'unknown',
                minutiae_data   TEXT NOT NULL,
                minutiae_count  INTEGER NOT NULL,
                source_image    TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES Users(user_id)
                    ON DELETE CASCADE
            )
        """)

        self.conn.commit()
        print("  Đã tạo bảng Users và Fingerprint_Templates.")

    # ────────────────────────────────────────────────────────────────────
    # PHA ĐĂNG KÝ (ENROLLMENT)
    # ────────────────────────────────────────────────────────────────────
    def add_user(self, name, role="Student"):
        """
        Thêm người dùng mới vào bảng Users.
        INSERT INTO Users (name, role) VALUES (?, ?)

        Returns: user_id của người dùng vừa thêm
        """
        self.cursor.execute(
            "INSERT INTO Users (name, role) VALUES (?, ?)",
            (name, role)
        )
        self.conn.commit()
        user_id = self.cursor.lastrowid
        print(f"  [INSERT] User: '{name}' (ID={user_id}, Role={role})")
        return user_id

    def enroll_fingerprint(self, user_id, minutiae, finger_index="unknown",
                           source_image=""):
        """
        Đăng ký vân tay: Chuyển minutiae thành JSON → INSERT vào DB.

        Pha Enrollment:
          1. Ảnh vân tay → Pipeline trích xuất → minutiae array
          2. Chuyển minutiae array → JSON string
          3. INSERT INTO Fingerprint_Templates (...)

        Cấu trúc JSON:
          [{"x": 120, "y": 95, "type": 1, "angle": 1.5708}, ...]

        Tham số:
          user_id:      ID người dùng (Foreign Key → Users)
          minutiae:     Numpy array [x, y, type, angle] từ pipeline
          finger_index: Tên ngón tay (e.g. "right_thumb")
          source_image: Đường dẫn ảnh gốc (để tham chiếu)
        """
        # Chuyển numpy array → list of dicts → JSON string
        minutiae_list = []
        for m in minutiae:
            minutiae_list.append({
                "x": float(m[0]),
                "y": float(m[1]),
                "type": int(m[2]),
                "angle": float(m[3])
            })

        json_data = json.dumps(minutiae_list)
        count = len(minutiae_list)

        self.cursor.execute("""
            INSERT INTO Fingerprint_Templates
                (user_id, finger_index, minutiae_data, minutiae_count, source_image)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, finger_index, json_data, count, source_image))

        self.conn.commit()
        template_id = self.cursor.lastrowid
        print(f"  [INSERT] Template ID={template_id}: "
              f"User={user_id}, Finger={finger_index}, "
              f"Minutiae={count}, Image={os.path.basename(source_image)}")
        return template_id

    # ────────────────────────────────────────────────────────────────────
    # PHA NHẬN DẠNG (MATCHING / IDENTIFICATION)
    # ────────────────────────────────────────────────────────────────────
    def get_all_templates(self):
        """
        Lấy tất cả template từ database.
        SELECT ... FROM Fingerprint_Templates JOIN Users

        Returns: list of dict với thông tin user + minutiae
        """
        self.cursor.execute("""
            SELECT
                ft.template_id,
                ft.user_id,
                u.name,
                u.role,
                ft.finger_index,
                ft.minutiae_data,
                ft.minutiae_count,
                ft.source_image
            FROM Fingerprint_Templates ft
            JOIN Users u ON ft.user_id = u.user_id
        """)

        results = []
        for row in self.cursor.fetchall():
            # Parse JSON → numpy array
            minutiae_list = json.loads(row["minutiae_data"])
            minutiae_array = np.array([
                [m["x"], m["y"], m["type"], m["angle"]]
                for m in minutiae_list
            ]) if minutiae_list else np.array([]).reshape(0, 4)

            results.append({
                "template_id": row["template_id"],
                "user_id": row["user_id"],
                "name": row["name"],
                "role": row["role"],
                "finger_index": row["finger_index"],
                "minutiae": minutiae_array,
                "minutiae_count": row["minutiae_count"],
                "source_image": row["source_image"],
            })

        return results

    def identify(self, query_minutiae, alpha_range=5):
        """
        Nhận dạng: So khớp minutiae truy vấn với TẤT CẢ template trong DB.

        Pha Matching:
          1. Trích xuất minutiae từ ảnh truy vấn
          2. SELECT tất cả template từ DB
          3. So khớp query vs mỗi template → Tính score
          4. Trả về danh sách xếp hạng theo score giảm dần

        Returns:
          results: List of (name, score, match, template_info) sắp xếp theo score
        """
        templates = self.get_all_templates()

        if not templates:
            print("  Cảnh báo: Database trống, không có template nào!")
            return []

        print(f"  Đang so khớp với {len(templates)} template trong database...")

        results = []
        for tmpl in templates:
            print(f"    So khớp với {tmpl['name']} ({tmpl['finger_index']})...", end="")
            t0 = time.time()

            score, _, _, _ = match_fingerprints(
                query_minutiae, tmpl["minutiae"], alpha_range=alpha_range
            )

            elapsed = time.time() - t0
            is_match = score > MATCH_THRESHOLD
            icon = "★" if is_match else "·"
            print(f" Score={score:.4f} {icon} ({elapsed:.1f}s)")

            results.append({
                "name": tmpl["name"],
                "user_id": tmpl["user_id"],
                "finger_index": tmpl["finger_index"],
                "template_id": tmpl["template_id"],
                "score": score,
                "match": is_match,
                "source_image": tmpl["source_image"],
            })

        # Sắp xếp theo score giảm dần
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    # ────────────────────────────────────────────────────────────────────
    # TRUY VẤN THÔNG TIN (QUERY)
    # ────────────────────────────────────────────────────────────────────
    def get_all_users(self):
        """SELECT * FROM Users"""
        self.cursor.execute("SELECT * FROM Users")
        return [dict(row) for row in self.cursor.fetchall()]

    def get_user_templates(self, user_id):
        """SELECT templates cho 1 user cụ thể."""
        self.cursor.execute(
            "SELECT * FROM Fingerprint_Templates WHERE user_id = ?",
            (user_id,)
        )
        return [dict(row) for row in self.cursor.fetchall()]

    def get_stats(self):
        """Thống kê database."""
        self.cursor.execute("SELECT COUNT(*) FROM Users")
        n_users = self.cursor.fetchone()[0]
        self.cursor.execute("SELECT COUNT(*) FROM Fingerprint_Templates")
        n_templates = self.cursor.fetchone()[0]
        return n_users, n_templates


# ============================================================================
# CHƯƠNG TRÌNH CHÍNH: Demo Enrollment + Matching
# ============================================================================
def main():
    print("=" * 60)
    print("  HỆ THỐNG NHẬN DẠNG VÂN TAY")
    print("  Fingerprint Recognition System")
    print("=" * 60)

    # === Khởi tạo Database ===
    # Xóa DB cũ nếu có để tạo mới (demo)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    db = FingerprintDatabase(DB_PATH)
    db.connect()
    db.create_tables()

    # =========================================================================
    # PHA 1: ENROLLMENT (Đăng ký)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  PHA 1: ENROLLMENT (Đăng ký vân tay)")
    print("=" * 60)

    # Dữ liệu giả lập: Mỗi subject (101-110) = 1 người, mỗi người 8 ảnh
    enrollment_data = [
        {"name": "Nguyen Van A", "role": "Student",  "subject": "101", "samples": ["1", "2"]},
        {"name": "Tran Thi B",   "role": "Student",  "subject": "102", "samples": ["1", "2"]},
        {"name": "Le Van C",     "role": "Lecturer", "subject": "103", "samples": ["1", "2"]},
        {"name": "Pham Thi D",   "role": "Student",  "subject": "104", "samples": ["1"]},
    ]

    for person in enrollment_data:
        print(f"\n{'─' * 50}")
        print(f"  Đăng ký: {person['name']} ({person['role']})")
        print(f"{'─' * 50}")

        # Thêm user
        user_id = db.add_user(person["name"], person["role"])

        # Đăng ký từng mẫu vân tay
        for sample in person["samples"]:
            filename = f"{person['subject']}_{sample}.tif"
            img_path = os.path.join(DATASET_PATH, filename)

            if not os.path.exists(img_path):
                print(f"  [SKIP] Không tìm thấy {filename}")
                continue

            print(f"  Đang trích xuất từ {filename}...")
            minutiae, _ = extract_features(img_path)

            if minutiae is not None and len(minutiae) > 0:
                db.enroll_fingerprint(
                    user_id=user_id,
                    minutiae=minutiae,
                    finger_index=f"right_thumb_sample{sample}",
                    source_image=img_path
                )
            else:
                print(f"  [FAIL] Không trích xuất được minutiae từ {filename}")

    # Thống kê sau enrollment
    n_users, n_templates = db.get_stats()
    print(f"\n  ✓ Enrollment hoàn tất: {n_users} users, {n_templates} templates")

    # =========================================================================
    # PHA 2: MATCHING (Nhận dạng / Xác thực)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  PHA 2: MATCHING (Nhận dạng vân tay)")
    print("=" * 60)

    # Test Case 1: Ảnh của Nguyen Van A (101_3.tif) - nên match với 101
    # Test Case 2: Ảnh của người lạ (105_1.tif) - không nên match với ai
    queries = [
        ("101_3.tif", "Ảnh của Nguyen Van A (mẫu khác)"),
        ("105_1.tif", "Ảnh người lạ (không có trong DB)"),
    ]

    all_results = []

    for query_file, description in queries:
        print(f"\n{'─' * 50}")
        print(f"  QUERY: {query_file}")
        print(f"  Mô tả: {description}")
        print(f"{'─' * 50}")

        query_path = os.path.join(DATASET_PATH, query_file)
        if not os.path.exists(query_path):
            print(f"  Lỗi: Không tìm thấy {query_file}")
            continue

        # Trích xuất minutiae từ ảnh truy vấn
        print(f"  Đang trích xuất đặc trưng từ {query_file}...")
        query_minutiae, query_img = extract_features(query_path)

        if query_minutiae is None or len(query_minutiae) == 0:
            print(f"  Lỗi: Không trích xuất được minutiae")
            continue

        print(f"  → {len(query_minutiae)} minutiae")

        # So khớp với database
        results = db.identify(query_minutiae, alpha_range=5)

        if results:
            best = results[0]
            if best["match"]:
                print(f"\n  ┌─────────────────────────────────────────┐")
                print(f"  │  ★ NHẬN DẠNG THÀNH CÔNG ★                │")
                print(f"  │  Danh tính: {best['name']:<27s} │")
                print(f"  │  Score:     {best['score']:.4f}                       │")
                print(f"  │  Ngón:      {best['finger_index']:<27s} │")
                print(f"  └─────────────────────────────────────────┘")
            else:
                print(f"\n  ┌─────────────────────────────────────────┐")
                print(f"  │  ✗ KHÔNG TÌM THẤY DANH TÍNH             │")
                print(f"  │  Score cao nhất: {best['score']:.4f} (< {MATCH_THRESHOLD})       │")
                print(f"  │  Người gần nhất: {best['name']:<22s} │")
                print(f"  └─────────────────────────────────────────┘")

        all_results.append({
            "query": query_file,
            "description": description,
            "results": results,
            "query_img": query_img,
        })

    # =========================================================================
    # HIỂN THỊ DATABASE
    # =========================================================================
    print("\n" + "=" * 60)
    print("  NỘI DUNG DATABASE")
    print("=" * 60)

    users = db.get_all_users()
    print(f"\n  Bảng Users ({len(users)} records):")
    print(f"  {'ID':<5s} {'Name':<20s} {'Role':<12s} {'Created'}")
    print(f"  {'─'*5} {'─'*20} {'─'*12} {'─'*20}")
    for u in users:
        print(f"  {u['user_id']:<5d} {u['name']:<20s} {u['role']:<12s} {u['created_at']}")

    templates = db.get_all_templates()
    print(f"\n  Bảng Fingerprint_Templates ({len(templates)} records):")
    print(f"  {'TID':<5s} {'UID':<5s} {'Name':<18s} {'Finger':<25s} {'#Min':<6s} {'Image'}")
    print(f"  {'─'*5} {'─'*5} {'─'*18} {'─'*25} {'─'*6} {'─'*15}")
    for t in templates:
        img_name = os.path.basename(t["source_image"]) if t["source_image"] else "N/A"
        print(f"  {t['template_id']:<5d} {t['user_id']:<5d} {t['name']:<18s} "
              f"{t['finger_index']:<25s} {t['minutiae_count']:<6d} {img_name}")

    # =========================================================================
    # TRỰC QUAN HÓA KẾT QUẢ
    # =========================================================================
    if all_results:
        n_queries = len(all_results)
        fig, axes = plt.subplots(n_queries, 1, figsize=(14, 5 * n_queries))
        if n_queries == 1:
            axes = [axes]

        fig.suptitle("Fingerprint Identification Results",
                     fontsize=16, fontweight='bold')

        for idx, qr in enumerate(all_results):
            ax = axes[idx]
            ax.axis('off')

            # Tạo text kết quả
            lines = [f"QUERY: {qr['query']}  ({qr['description']})\n"]
            lines.append(f"{'Rank':<6s} {'Name':<20s} {'Score':<10s} {'Result':<15s} {'Finger'}")
            lines.append("─" * 70)

            for rank, r in enumerate(qr['results'], 1):
                icon = "★ MATCH" if r['match'] else "  -"
                lines.append(
                    f"  {rank:<4d} {r['name']:<20s} {r['score']:<10.4f} {icon:<15s} {r['finger_index']}"
                )

            if qr['results'] and qr['results'][0]['match']:
                bg = '#d4edda'
                title = f"✓ Identified: {qr['results'][0]['name']}"
            else:
                bg = '#f8d7da'
                title = "✗ No match found"

            text = "\n".join(lines)
            ax.text(0.02, 0.95, text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor=bg, alpha=0.8))
            ax.set_title(title, fontsize=13, fontweight='bold',
                         color='green' if '✓' in title else 'red')

        plt.tight_layout()
        path1 = os.path.join(OUTPUT_DIR, "10_database_results.png")
        plt.savefig(path1, dpi=200, bbox_inches='tight')
        print(f"\nĐã lưu: {path1}")

    # =========================================================================
    # TÓM TẮT CUỐI CÙNG
    # =========================================================================
    db.close()

    print("\n" + "=" * 60)
    print("  ★★★ HỆ THỐNG HOÀN TẤT ★★★")
    print("=" * 60)
    print(f"\n  Database file: {DB_PATH}")
    print(f"  Users:         {n_users}")
    print(f"  Templates:     {n_templates}")
    print(f"\n  Cấu trúc Database:")
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │ Users                                        │")
    print(f"  │ ├── user_id (PK)                             │")
    print(f"  │ ├── name                                     │")
    print(f"  │ ├── role                                     │")
    print(f"  │ └── created_at                               │")
    print(f"  │                    1:N                        │")
    print(f"  │ Fingerprint_Templates                        │")
    print(f"  │ ├── template_id (PK)                         │")
    print(f"  │ ├── user_id (FK → Users)                     │")
    print(f"  │ ├── finger_index                             │")
    print(f"  │ ├── minutiae_data (JSON)                     │")
    print(f"  │ ├── minutiae_count                           │")
    print(f"  │ ├── source_image                             │")
    print(f"  │ └── created_at                               │")
    print(f"  └─────────────────────────────────────────────┘")
    print(f"\n  Lưu trữ minutiae: JSON")
    print(f'  VD: [{{"x":120,"y":95,"type":1,"angle":1.57}}, ...]')
    print(f"\n  2 Luồng hoạt động:")
    print(f"  ├── Enrollment: Ảnh → Pipeline → INSERT minutiae JSON")
    print(f"  └── Matching:   Ảnh → Pipeline → SELECT all → So khớp → Kết quả")


if __name__ == "__main__":
    main()
