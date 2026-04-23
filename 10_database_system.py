"""
Bước 10: Cơ sở Dữ liệu & Tìm kiếm Faiss IVF
=========================================================================
Mục tiêu:
  1. Loại bỏ cấu trúc User, chuyển sang một bảng Fingerprints đồng nhất.
  2. Bổ sung trường `cluster_id` để mô tả nhãn phân cụm khi dùng IVF.
  3. Áp dụng FAISS `IndexIVFFlat` phân cụm và search top-5 
     thay cho thao tác for vét cạn từng fingerprint.
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
        """Tạo cấu trúc bảng mới (chỉ một bảng Fingerprints)."""
        self.cursor.execute("DROP TABLE IF EXISTS Fingerprint_Templates")
        self.cursor.execute("DROP TABLE IF EXISTS Users")
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS Fingerprints (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
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
        Nạp một danh sách các tuple [(source_image, feature_vector), ...]
        Xây dựng Index IVF, lấy cluster ID và insert vào SQLite.
        """
        # Nếu số lượng ít quá (dưới 10), faiss sẽ khó build IVF.
        # Nhưng để demo thuật toán IVF, ta chốt cố định nlist nhỏ, ví dụ 2.
        num_vectors = len(feature_data_list)
        if num_vectors == 0:
            return
            
        d = len(feature_data_list[0][1])  # dimension
        nlist = max(2, min(5, num_vectors // 5))  # Ví dụ: tối thiểu 2 cụm, kích thước tùy ý
        
        # Array vectors float32 của Faiss
        xb = np.array([item[1] for item in feature_data_list], dtype=np.float32)
        
        # Định nghĩa Index IVF
        quantizer = faiss.IndexFlatL2(d)
        index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        
        # Train index
        print(f"  [FAISS] Đang huấn luyện IndexIVFFlat với {nlist} cụm...")
        index_ivf.train(xb)
        
        # Tìm Cụm (Cluster ID) của mỗi vector để lưu vào database
        _, cluster_ids_db = quantizer.search(xb, 1)
        
        inserted_ids = []
        for i, (source_img, vector) in enumerate(feature_data_list):
            cluster_assign = int(cluster_ids_db[i][0])
            json_vector = json.dumps(vector.tolist())
            
            # Ghi vào SQLite
            self.cursor.execute("""
                INSERT INTO Fingerprints (source_image, feature_vector, cluster_id)
                VALUES (?, ?, ?)
            """, (source_img, json_vector, cluster_assign))
            self.conn.commit()
            
            inserted_id = self.cursor.lastrowid
            inserted_ids.append(inserted_id)
            print(f"  [DB] Đã lưu {os.path.basename(source_img)} -> ID={inserted_id}, Cluster={cluster_assign}")
            
        # Nạp dữ liệu vào FAISS index với index.add_with_ids
        # Chú ý ids phải là int64 array và đúng số lượng
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
            self.cursor.execute("SELECT source_image, cluster_id FROM Fingerprints WHERE id = ?", (int(db_id),))
            row = self.cursor.fetchone()
            if row:
                dist_score = distances[0][idx_order]
                # Nếu normalized vectors, Cosine Similarity liên hệ trực tiếp qua L2 distance, 
                # distance nhỏ -> càng giống. Đổi ra dạng score (vd: 1 / (1 + distance)) cho dễ nhìn
                sim_score = 1.0 / (1.0 + dist_score)
                
                results.append({
                    "id": db_id,
                    "source_image": row["source_image"],
                    "cluster_id": row["cluster_id"],
                    "distance": dist_score,
                    "similarity": sim_score
                })
                
        print(f"  [FAISS Search] Mất {elapsed*1000:.2f}ms để hoàn tất IVF Query.")
        return results

    def get_all_records(self):
        """Thống kê chi tiết Database"""
        self.cursor.execute("SELECT id, source_image, cluster_id, created_at FROM Fingerprints")
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

    # Tự động lấy toàn bộ file ảnh trong DATASET_PATH (hỗ trợ .tif, .jpg, .png)
    valid_extensions = ('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp')
    all_files = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(valid_extensions)]
    all_files.sort() # Sắp xếp để dễ theo dõi
    
    # Bạn có thể chọn dùng config.ENROLL_SAMPLES (danh sách chọn lọc) 
    # hoặc dùng all_files (tất cả ảnh trong thư mục)
    # Ở đây tôi sẽ ưu tiên lấy tất cả ảnh nếu danh sách trong config là rỗng
    enroll_samples = config.ENROLL_SAMPLES if config.ENROLL_SAMPLES else all_files
    
    print(f"  Tìm thấy {len(all_files)} ảnh trong dataset.")
    print(f"  Sẽ tiến hành đăng ký {len(enroll_samples)} ảnh vào Database...")
    
    extracted_data = [] # Lưu trữ [(path, feature_vec), ...] để làm batch_enroll
    
    for sample in enroll_samples:
        img_path = os.path.join(DATASET_PATH, sample)
        if not os.path.exists(img_path):
            print(f"  [SKIP] Khong thay anh: {sample}")
            continue
            
        print(f"  Đang trích xuất Fingercode cho ảnh {sample}...")
        vector, _ = extract_features(img_path)
        if vector is not None and len(vector) > 0:
            extracted_data.append((img_path, vector))
        else:
            print(f"  [FAIL] Lỗi trích xuất trên {sample}")
            
    # Gửi qua DB Controller xử lý insert và xây dựng Faiss IVF
    if extracted_data:
        db.batch_enroll_and_build(extracted_data)

    # In thông tin bảng
    records = db.get_all_records()
    print(f"\n  SQL Table Fingerprints ({len(records)} records):")
    print(f"  {'ID':<5} {'Source File':<20} {'Cluster ID':<10}")
    print(f"  {'-'*5} {'-'*20} {'-'*10}")
    for item in records:
         filename = os.path.basename(item["source_image"])
         print(f"  {item['id']:<5} {filename:<20} {item['cluster_id']:<10}")

    # == PHA 2: MATCHING (QUERY TÌM TOP-5) ==
    print("\n" + "=" * 60)
    print("  PHA 2: MATCHING (KNN-SEARCH BẰNG FAISS IVF)")
    print("=" * 60)
    
    test_queries = [
        ("101_4.tif", "Cùng vân 101, chưa tồn tại trong Db"), 
        ("106_2.tif", "Cùng vân 106, chưa tồn tại trong Db")
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
             print(f"    R {i}: {file_base:<15} | Sim Score: {item['similarity']:.4f} "
                   f"| L2: {item['distance']:.4f} | Cluster: {item['cluster_id']}")

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
                        ax_res.set_title(f"#{j+1}: {s_name}\nSim: {match['similarity']:.3f}\nCluster: {match['cluster_id']}")
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
