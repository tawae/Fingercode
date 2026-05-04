"""
Bước 10: Cơ sở Dữ liệu & Tìm kiếm Faiss IVF
=========================================================================
Mục tiêu:
  1. Loại bỏ cấu trúc User, chuyển sang một bảng Fingerprints đồng nhất.
  2. Bổ sung trường `cluster_id` để mô tả nhãn phân cụm khi dùng IVF.
  3. Áp dụng FAISS `IndexIVFFlat` phân cụm và search top-5
     thay cho thao tác for vét cạn từng fingerprint.
  4. Thêm các trường `user_id`, `sex`, `finger_index` trích xuất từ tên ảnh.
     Định dạng tên ảnh: <user_id>__<sex>_<hand>_<finger_type>_finger.BMP
     Ví dụ: 1__M_Left_index_finger.BMP  ->  user_id=1, sex=M, finger_index=Left_index
  5. Chỉ nạp 3000 ảnh đầu tiên với user_id từ 1 đến 3000.
"""

import sqlite3
import json
import numpy as np
import cv2
import os
import faiss
import time
import matplotlib.pyplot as plt

# ============================================================================
# CẤU HÌNH
# ============================================================================
import config

BASE_DIR = config.BASE_DIR
DATASET_PATH = config.DATASET_PATH
DB_PATH = config.DB_PATH
OUTPUT_DIR = config.OUTPUT_DIR
FAISS_INDEX_PATH = config.FAISS_INDEX_PATH
TARGET_KNN = config.TARGET_KNN
FAISS_DIM = config.FAISS_DIM

# ============================================================================
# IMPORT TỪ BƯỚC FEATURE EXTRACTION (08)
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
# TIỆN ÍCH: PARSE TÊN FILE ẢNH
# ============================================================================
def parse_filename(filename):
    """
    Trích xuất user_id, sex, finger_index từ tên file ảnh.
    Định dạng: <user_id>__<sex>_<hand>_<finger_type>_finger[.ext]
    Ví dụ: '1__M_Left_index_finger.BMP'  ->  (1, 'M', 'Left_index')
    Trả về (None, None, None) nếu không parse được.
    """
    # Bỏ phần mở rộng file
    name = os.path.splitext(filename)[0]   # '1__M_Left_index_finger'
    parts = name.split('__')               # ['1', 'M_Left_index_finger']
    if len(parts) != 2:
        return None, None, None
    
    user_id_str = parts[0].strip()
    rest = parts[1].strip()                # 'M_Left_index_finger'
    
    # Tách sex (ký tự đầu trước '_') và phần còn lại
    rest_parts = rest.split('_')           # ['M', 'Left', 'index', 'finger']
    if len(rest_parts) < 4:
        return None, None, None
    
    sex = rest_parts[0]                    # 'M'
    # finger_index = hand + '_' + finger_type (bỏ đuôi '_finger')
    hand = rest_parts[1]                   # 'Left'
    finger_type = rest_parts[2]            # 'index'
    finger_index = f"{hand}_{finger_type}"  # 'Left_index'
    
    try:
        user_id = int(user_id_str)
    except ValueError:
        return None, None, None
    
    return user_id, sex, finger_index

# ============================================================================
# DATABSE MANAGER QUẢN LÝ SQLITE + FAISS
# ============================================================================
class FingerprintVectorDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.index = None
    
    def connect(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        print(f"  Đã kết nối database: {self.db_path}")
        
        self._init_tables()
        self._load_or_create_index()

    def close(self):
        if self.conn:
            self.conn.close()
            print("  Đã đóng kết nối database.")

    def _init_tables(self):
        """Đảm bảo bảng Fingerprints tồn tại. KHÔNG xóa dữ liệu cũ.
        Việc reset DB được xử lý bởi os.remove(DB_PATH) trong main() trước khi connect().
        """
        # Dọn các bảng schema cũ (legacy) nếu còn sót
        self.cursor.execute("DROP TABLE IF EXISTS Fingerprint_Templates")
        self.cursor.execute("DROP TABLE IF EXISTS Users")

        # Tạo bảng Fingerprints chỉ khi chưa tồn tại — KHÔNG DROP dữ liệu hiện có
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS Fingerprints (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id        INTEGER NOT NULL,
                sex            TEXT NOT NULL,
                finger_index   TEXT NOT NULL,
                source_image   TEXT NOT NULL,
                feature_vector TEXT NOT NULL,
                cluster_id     INTEGER DEFAULT -1,
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _load_or_create_index(self):
        """Khởi tạo FAISS Index. Nạp từ đĩa nếu có."""
        if os.path.exists(FAISS_INDEX_PATH):
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            print(f"  Đã load FAISS Index từ {FAISS_INDEX_PATH} (Tổng vector: {self.index.ntotal})")
        else:
            print("  FAISS Index chưa tồn tại. Sẽ tạo lúc Build/Enrollment.")
            self.index = None

    # ────────────────────────────────────────────────────────────────────
    # PHA 1: ENROLLMENT (ĐĂNG KÝ VÀ BUILD INDEX HÀNG LOẠT)
    # ────────────────────────────────────────────────────────────────────
    def batch_enroll_and_build(self, feature_data_list):
        """
        Nạp một danh sách các tuple [(source_image, user_id, sex, finger_index, feature_vector), ...]
        Xây dựng Index IVF, lấy cluster ID và insert vào SQLite.
        """
        num_vectors = len(feature_data_list)
        if num_vectors == 0:
            return

        d = len(feature_data_list[0][4])  # dimension (index 4 = feature_vector)
        nlist = max(2, min(5, num_vectors // 5))
        # Array vectors float32 của Faiss
        xb = np.array([item[4] for item in feature_data_list], dtype=np.float32)

        # Định nghĩa Index IVF
        quantizer = faiss.IndexFlatL2(d)
        index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

        # Train index
        print(f"  [FAISS] Đang huấn luyện IndexIVFFlat với {nlist} cụm trên {num_vectors} vectors...")
        index_ivf.train(xb)

        # Tìm Cụm (Cluster ID) của mỗi vector để lưu vào database
        _, cluster_ids_db = quantizer.search(xb, 1)

        inserted_ids = []
        for i, (source_img, uid, sex, finger_idx, vector) in enumerate(feature_data_list):
            cluster_assign = int(cluster_ids_db[i][0])
            json_vector = json.dumps(vector.tolist())

            # Ghi vào SQLite
            self.cursor.execute("""
                INSERT INTO Fingerprints (user_id, sex, finger_index, source_image, feature_vector, cluster_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (uid, sex, finger_idx, source_img, json_vector, cluster_assign))
            self.conn.commit()

            inserted_id = self.cursor.lastrowid
            inserted_ids.append(inserted_id)
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [DB] {i+1}/{num_vectors}: {os.path.basename(source_img)}"
                      f" -> ID={inserted_id}, user_id={uid}, sex={sex},"
                      f" finger={finger_idx}, Cluster={cluster_assign}")

        # Nạp dữ liệu vào FAISS index với index.add_with_ids
        ids_array = np.array(inserted_ids, dtype=np.int64)
        index_ivf.add_with_ids(xb, ids_array)

        # Lưu index và gán lên class
        faiss.write_index(index_ivf, FAISS_INDEX_PATH)
        self.index = index_ivf
        print(f"  [FAISS] Hoàn tất build và ghi đĩa FAISS Index (kích thước {self.index.ntotal} records).")


    # ────────────────────────────────────────────────────────────────────
    # PHA 2: MATCHING VÀ LẤY VỀ KNN (TÌM TOP-5)
    # ────────────────────────────────────────────────────────────────────
    def search_top_k(self, query_vector, k=TARGET_KNN):
        """
        Tìm K dấu vân tay gần nhất từ vector truy vấn.
        """
        if self.index is None or not self.index.is_trained:
            print("  Lỗi: FAISS Index chưa sẵn sàng!")
            return []
            
        # Thao tác bắt buộc với IVF để cải thiện tìm kiếm: tăng nprobe
        self.index.nprobe = 3 # Quét 3 centroid gần nhất (do tập dữ liệu nhỏ)
        
        # Chuyển đổi và search
        xq = np.array([query_vector], dtype=np.float32)
        
        start_time = time.time()
        distances, indices = self.index.search(xq, k)
        elapsed = time.time() - start_time
        
        results = []
        for idx_order, db_id in enumerate(indices[0]):
            if db_id == -1: continue # FAISS trả -1 nếu không đủ kết quả k
            
            # Retrieve from SQL
            self.cursor.execute(
                "SELECT user_id, sex, finger_index, source_image, cluster_id "
                "FROM Fingerprints WHERE id = ?", (int(db_id),)
            )
            row = self.cursor.fetchone()
            if row:
                dist_score = distances[0][idx_order]
                # distance nhỏ -> càng giống. Đổi ra dạng score cho dễ nhìn
                sim_score = 1.0 / (1.0 + dist_score)

                results.append({
                    "id": db_id,
                    "user_id": row["user_id"],
                    "sex": row["sex"],
                    "finger_index": row["finger_index"],
                    "source_image": row["source_image"],
                    "cluster_id": row["cluster_id"],
                    "distance": dist_score,
                    "similarity": sim_score
                })
                
        print(f"  [FAISS Search] Mất {elapsed*1000:.2f}ms để hoàn tất IVF Query.")
        return results

    def get_all_records(self):
        """Thống kê chi tiết Database"""
        self.cursor.execute(
            "SELECT id, user_id, sex, finger_index, source_image, cluster_id, created_at "
            "FROM Fingerprints ORDER BY id"
        )
        return [dict(row) for row in self.cursor.fetchall()]

# ============================================================================
# CHƯƠNG TRÌNH CHÍNH TỔNG HỢP (ENROLL + MATCHING)
# ============================================================================
def main():
    print("=" * 60)
    print("  HỆ THỐNG IVF VÀ ĐẶC TRƯNG CỐ ĐỊNH (FINGERCODE)")
    print("=" * 60)

    # Đặt lại hệ thống Demo
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)

    db = FingerprintVectorDB(DB_PATH)
    db.connect()

    # == PHA 1: ENROLLMENT ==
    print("\n" + "=" * 60)
    print("  PHA 1: ENROLLMENT VÀ QUẢN LÝ VECTOR TẬP TRUNG")
    print("=" * 60)

    # Lấy toàn bộ file ảnh trong DATASET_PATH
    valid_extensions = ('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp')
    all_files = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(valid_extensions)]
    all_files.sort()

    # Lọc chỉ lấy ảnh có user_id từ 1 đến 300 (300 người × 10 ngón = 3000 ảnh)
    MAX_USER_ID = 300  # 300 người × 10 ảnh/người = 3000 ảnh
    filtered_files = []
    for fname in all_files:
        uid, sex, finger_idx = parse_filename(fname)
        if uid is not None and 1 <= uid <= MAX_USER_ID:
            filtered_files.append((fname, uid, sex, finger_idx))

    # QUAN TRỌNG: sort theo user_id (số nguyên) để insert đúng thứ tự tuyến tính.
    # all_files.sort() chỉ sort lexicographic nên "100" đứng trước "2",
    # dẫn đến user_id=100 nhận id=1, user_id=1 nhận id=1101 (sai hoàn toàn).
    filtered_files.sort(key=lambda x: (x[1], x[0]))  # sort by (user_id, filename)

    print(f"  Tìm thấy {len(all_files)} ảnh trong dataset.")
    print(f"  Sau khi lọc user_id 1-{MAX_USER_ID}: {len(filtered_files)} ảnh sẽ được nạp vào Database.")
    print(f"  Thứ tự nạp: user_id {filtered_files[0][1]} → {filtered_files[-1][1]}")

    extracted_data = []  # [(source_image, user_id, sex, finger_index, feature_vector), ...]

    for idx, (sample, uid, sex, finger_idx) in enumerate(filtered_files, 1):
        img_path = os.path.join(DATASET_PATH, sample)
        if not os.path.exists(img_path):
            print(f"  [SKIP] Không thấy ảnh: {sample}")
            continue

        if idx % 100 == 1:
            print(f"  [{idx}/{len(filtered_files)}] Đang trích xuất: {sample}")

        vector, _ = extract_features(img_path)
        if vector is not None and len(vector) > 0:
            extracted_data.append((img_path, uid, sex, finger_idx, vector))
        else:
            print(f"  [FAIL] Lỗi trích xuất trên {sample}")

    # Gửi qua DB Controller xử lý insert và xây dựng Faiss IVF
    if extracted_data:
        db.batch_enroll_and_build(extracted_data)

    # In thống kê bảng (hiển thị 20 dòng đầu)
    records = db.get_all_records()
    print(f"\n  SQL Table Fingerprints ({len(records)} records) - Hiển thị 20 dòng đầu:")
    print(f"  {'ID':<5} {'UserID':<8} {'Sex':<5} {'Finger':<15} {'Source File':<40} {'Cluster':<8}")
    print(f"  {'-'*5} {'-'*8} {'-'*5} {'-'*15} {'-'*40} {'-'*8}")
    for item in records[:20]:
        filename = os.path.basename(item["source_image"])
        print(f"  {item['id']:<5} {item['user_id']:<8} {item['sex']:<5} "
              f"{item['finger_index']:<15} {filename:<40} {item['cluster_id']:<8}")
    if len(records) > 20:
        print(f"  ... (còn {len(records) - 20} dòng nữa)")

    # == PHA 2: MATCHING (QUERY TÌM TOP-5) ==
    print("\n" + "=" * 60)
    print("  PHA 2: MATCHING (KNN-SEARCH BẰNG FAISS IVF)")
    print("=" * 60)
    
    test_queries = [
        ("101__M_Left_index_finger.BMP", "user_id=101, đã có trong DB"),
        ("500__M_Right_thumb_finger.BMP", "user_id=500, đã có trong DB"),
    ]
    
    all_query_results = []
    for q_file, desc in test_queries:
        print(f"\n{'─' * 50}")
        print(f"  QUERY IMAGE: {q_file} ({desc})")
        print(f"{'─' * 50}")
        
        q_path = os.path.join(DATASET_PATH, q_file)
        if not os.path.exists(q_path):
            print(f"  => [SKIP] Tệp {q_path} không tồn tại!")
            continue
            
        t0 = time.time()
        q_vec, q_img = extract_features(q_path)
        t_ext = time.time() - t0
        print(f"  [Time] Trích xuất vector: {t_ext*1000:.1f}ms")
        
        if q_vec is None:
            continue
            
        # Gọi Search (Không dùng Loop O(N)!)
        top_k_list = db.search_top_k(q_vec, k=TARGET_KNN)
        
        print(f"\n  > --- KẾT QUẢ TOP {TARGET_KNN} TƯƠNG ĐỒNG NHẤT --- <")
        for i, item in enumerate(top_k_list, 1):
            file_base = os.path.basename(item['source_image'])
            print(f"    R {i}: {file_base:<40} | user_id={item['user_id']:<5}"
                  f" sex={item['sex']} finger={item['finger_index']:<15}"
                  f" | Sim: {item['similarity']:.4f} | L2: {item['distance']:.4f}"
                  f" | Cluster: {item['cluster_id']}")

        all_query_results.append({
            "query": q_file,
            "desc": desc,
            "img": q_img,
            "top_k": top_k_list
        })
        
    db.close()
    
    # == VISUALIZE KẾT QUẢ TƯƠNG ĐỒNG ==
    if all_query_results:
        print("\n  Tạo visualization báo cáo...")
        n_queries = len(all_query_results)
        fig, axes = plt.subplots(n_queries, TARGET_KNN + 1, figsize=(15, 3 * n_queries))
        if n_queries == 1: axes = [axes]
        
        for i, qr in enumerate(all_query_results):
            ax_q = axes[i][0]
            ax_q.imshow(qr["img"], cmap="gray")
            ax_q.set_title(f"Query: {qr['query']}\n(Input)")
            ax_q.axis('off')
            
            for j in range(TARGET_KNN):
                ax_res = axes[i][j+1]
                if j < len(qr["top_k"]):
                    match = qr["top_k"][j]
                    res_img = cv2.imread(match["source_image"], cv2.IMREAD_GRAYSCALE)
                    if res_img is not None:
                        ax_res.imshow(res_img, cmap="gray")
                        s_name = os.path.basename(match['source_image'])
                        ax_res.set_title(
                            f"#{j+1}: {s_name}\n"
                            f"UserID={match['user_id']} | {match['sex']} | {match['finger_index']}\n"
                            f"Sim: {match['similarity']:.3f} | Cluster: {match['cluster_id']}"
                        )
                ax_res.axis('off')
                
        plt.tight_layout()
        path_res = os.path.join(OUTPUT_DIR, "10_database_ivf_results.png")
        plt.savefig(path_res, dpi=200, bbox_inches='tight')
        print(f"  Đã xuất đồ họa: {path_res}")
        
    print("\n" + "=" * 60)
    print("   ★★★ HOÀN THIỆN XÂY DỰNG FINGERCODE + FAISS IVF ★★★")
    print("=" * 60)


if __name__ == "__main__":
    main()
