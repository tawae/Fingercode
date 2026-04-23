import os

# ============================================================================
# TỆP CẤU HÌNH HỆ THỐNG
# ============================================================================

# Thư mục gốc chứa mã nguồn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Đường dẫn thư mục Dataset (Tập ảnh vân tay)
# Thay đổi biến này khi muốn chạy trên bộ dữ liệu khác.
DATASET_PATH = os.path.join(BASE_DIR, "..", "FVC2002", "DB1_B")

# 2. Đường dẫn lưu trữ SQLite Database
DB_PATH = os.path.join(BASE_DIR, "fingerprint.db")

# 3. Thư mục đầu ra lưu các file kết quả hoặc log
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 4. Đường dẫn lưu FAISS Index
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_ivf.index")

# 5. Cấu hình thuật toán FAISS (Tìm kiếm K-Nearest Neighbors)
TARGET_KNN = 5          # Số lượng kết quả tương đồng nhất muốn trả về
FAISS_DIM = 320         # Kích thước vector đặc trưng (5 bands * 8 sectors * 8 Gabor angles)

# 6. Tập ảnh mặc định để đăng ký ban đầu (Enrollment) khi build database
ENROLL_SAMPLES = [
    # "101_1.tif", "101_2.tif", "101_3.tif", 
    # "102_1.tif", "102_2.tif", 
    # "103_1.tif", "103_2.tif", 
    # "104_1.tif", "105_1.tif", "106_1.tif"
]

# Đảm bảo thư mục đầu ra tồn tại
os.makedirs(OUTPUT_DIR, exist_ok=True)
