# Fingerprint Recognition System in Python

Dự án này là một hệ thống nhận dạng vân tay hoàn chỉnh được cài đặt bằng Python. Dự án chuyển đổi các thuật toán xử lý vân tay kinh điển (thường dùng trong MATLAB) sang Python, bao gồm toàn bộ quy trình: từ tiền xử lý ảnh, lọc Gabor, trích xuất đặc trưng (Minutiae), so khớp (Matching), cho đến quản lý dữ liệu với SQLite.

## Mục lục
- [Luồng Hệ Thống (System Pipeline)](#-luồng-hệ-thống-system-pipeline)
- [Thư viện sử dụng](#-thư-viện-sử-dụng)
- [Cài đặt (Installation)](#-cài-đặt-installation)
- [Cấu trúc Thư mục & Dữ liệu](#-cấu-trúc-thư-mục--dữ-liệu)
- [Cách sử dụng (Usage)](#-cách-sử-dụng-usage)
- [Cấu trúc Cơ sở dữ liệu](#-cấu-trúc-cơ-sở-dữ-liệu)

---

## Luồng Hệ Thống (System Pipeline)

Hệ thống được chia thành 10 bước (tương ứng với 10 file script), thực thi tuần tự để biến một bức ảnh vân tay thô thành thông tin định danh:

1. **Visualize (`01_visualize_fingerprint.py`):** Đọc ảnh vân tay (Grayscale) và vẽ biểu đồ Histogram để phân biệt đường vân (Ridge) và nền (Valley).
2. **Preprocessing (`02_preprocessing.py`):** Chuẩn hóa độ sáng (Normalization) và cắt bỏ nền thừa (Segmentation) dựa trên phương sai (Variance).
3. **Enhancement (`03_enhancement.py`):** Tăng cường độ tương phản cục bộ sử dụng thuật toán CLAHE (thay vì FFT như nguyên bản) để làm rõ đường vân.
4. **Orientation Field (`04_orientation_field.py`):** Ước lượng hướng của đường vân tại từng pixel bằng bộ lọc Sobel và ma trận Hiệp phương sai (Covariance).
5. **Frequency Estimation (`05_frequency_estimation.py`):** Ước lượng tần số (khoảng cách giữa các đường vân) bằng cách xoay block ảnh và phân tích hình sin của các điểm ảnh.
6. **Gabor Filter (`06_gabor_filter.py`):** Áp dụng bộ lọc Gabor 2D dựa trên Hướng (bước 4) và Tần số (bước 5) để làm mịn vân tay và loại bỏ nhiễu.
7. **Binarize & Thinning (`07_binarize_thin.py`):** Nhị phân hóa ảnh về dạng Đen/Trắng và làm mảnh đường vân (Skeletonization) xuống kích thước 1 pixel.
8. **Minutiae Extraction (`08_minutiae_extraction.py`):** Sử dụng thuật toán **Crossing Number** để tìm các điểm đặc trưng: Điểm kết thúc (Termination) và Điểm rẽ nhánh (Bifurcation). Loại bỏ các điểm giả mạo (False Minutiae).
9. **Matching (`09_matching.py`):** So khớp 2 tập hợp Minutiae thông qua các phép biến đổi không gian (Translate & Rotate) và tính điểm tương đồng (Similarity Score).
10. **Database System (`10_database_system.py`):** Hệ thống cơ sở dữ liệu SQLite mô phỏng ứng dụng thực tế với 2 pha: **Enrollment** (Đăng ký vân tay) và **Matching** (Nhận dạng người dùng).

---

## Thư viện sử dụng

Dự án phụ thuộc vào các thư viện Python sau:
- `numpy`: Xử lý mảng và ma trận toán học.
- `opencv-python` (`cv2`): Đọc, ghi và xử lý các phép toán trên ảnh cơ bản (Sobel, CLAHE, Thresholding...).
- `matplotlib`: Trực quan hóa dữ liệu, vẽ biểu đồ và hiển thị kết quả các bước.
- `scipy`: Xoay ảnh (rotate), tìm đỉnh sóng (maximum_filter1d) và tính toán khoảng cách Euclid.
- `scikit-image` (`skimage`): Cung cấp thuật toán làm mảnh khung xương (`skeletonize`).

---

## Cài đặt (Installation)

**1. Clone repository:**
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

**2. Tạo môi trường ảo (Khuyến nghị):**

```bash
# Active môi trường
python -m venv venv

#trước mỗi lần chạy, cần activate môi trường ảo như sau:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

**3. Cài đặt các thư viện yêu cầu:**

```bash
pip install -r requirements.txt
```
**4. Cấu trúc Thư mục & Dữ liệu**

Mã nguồn được thiết lập để đọc dữ liệu vân tay từ dataset FVC2002 (DB1_B). Bạn cần sắp xếp thư mục theo đúng cấu trúc sau để code không bị lỗi đường dẫn:

```bash
Workspace/
│
├── FVC2002/
│   └── DB1_B/
│       ├── 101_1.tif
│       ├── 101_2.tif
│       └── ... (Các ảnh vân tay khác)
│
└── Python_Implement_FingerPrint_Project/    <-- (Thư mục Repo của bạn)
    ├── .gitignore
    ├── 01_visualize_fingerprint.py
    ├── 02_preprocessing.py
    ├── ...
    ├── 10_database_system.py
    └── requirements.txt
```
(Code sử dụng os.path.join(BASE_DIR, "..", "FVC2002", "DB1_B") để tìm ảnh).

**5. Cách sử dụng (Usage)**

Bạn có thể chạy độc lập từng script để xem kết quả của từng bước. Kết quả (các biểu đồ, hình ảnh so sánh) sẽ được tự động lưu vào thư mục output/ nằm trong project.

Chạy để hiểu quy trình (Từ bước 1 đến 9):

```bash
python 01_visualize_fingerprint.py
python 03_enhancement.py
...
python 09_matching.py
```

Chạy hệ thống hoàn chỉnh (Bước 10):

```bash
python 10_database_system.py
```
Khi chạy script số 10, hệ thống sẽ tự động tạo file fingerprint.db (SQLite), đăng ký một số dữ liệu mẫu giả lập và thực hiện truy vấn so khớp để in kết quả ra Terminal.

**6. Cấu trúc Cơ sở dữ liệu**

Hệ thống ở bước 10 sử dụng SQLite với cấu trúc tối ưu:

Bảng Users: Chứa thông tin định danh (Tên, Chức vụ...).

Bảng Fingerprint_Templates:

Liên kết với Users qua Khóa ngoại (user_id).

Lưu trữ các Minutiae đã trích xuất dưới dạng chuỗi JSON ([{"x": 120, "y": 95, "type": 1, "angle": 1.57}, ...]) giúp tối ưu hóa dung lượng thay vì lưu ảnh gốc.

Khi có truy vấn nhận dạng, hệ thống gọi dữ liệu từ DB, parse JSON thành mảng NumPy và tính toán Matching Score.