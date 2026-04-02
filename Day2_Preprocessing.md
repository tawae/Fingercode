# Day 2: Làm Sạch & Tách Nền (Preprocessing)

## 1. Tại sao cần bước này?
Ảnh vân tay gốc thường không hoàn hảo: quá sáng, quá tối, hoặc dính nhiều nền giấy trắng xung quanh.
Nếu đưa ảnh "bẩn" này vào xử lý ngay, máy tính sẽ tìm ra hàng tá đặc trưng giả ở vùng nền giấy.

Hôm nay chúng ta giải quyết 2 vấn đề:
1.  **Normalization (Chuẩn hóa):** Đưa độ sáng của toàn bộ ảnh về một mức chuẩn (trung bình 0, phương sai 1). Giúp xử lý ngón tay ướt (quá đen) hoặc khô (quá mờ) như nhau.
2.  **Segmentation (Tách nền):** Cắt bỏ phần rìa trắng thừa thãi, chỉ giữ lại vùng có vân tay (ROI - Region of Interest).

## 2. Giải thuật Tách Nền (Segmentation)
Làm sao máy tính biết đâu là vân tay, đâu là giấy trắng?
*   **Nguyên lý:** Vùng có vân tay thì màu sắc thay đổi liên tục (Đen -> Trắng -> Đen...). Tức là **Phương sai (Variance)** cao.
*   **Cách làm:**
    1. Chia ảnh thành các ô vuông nhỏ (ví dụ 16x16 pixel).
    2. Tính phương sai của từng ô.
    3. Nếu phương sai > ngưỡng (threshold) -> Là Vân tay.
    4. Ngược lại -> Là Nền -> Xóa (gán bằng 0).

## 3. Bài Tập Thực Hành
Chạy script `02_preprocessing.py` để xem kết quả tách nền.

```bash
python Python_Implementation/02_preprocessing.py
```

### Kết quả mong đợi:
Bạn sẽ thấy 4 hình:
1.  Ảnh gốc.
2.  Ảnh sau khi chuẩn hóa (độ tương phản tốt hơn).
3.  Vùng mặt nạ (Mask): Màu trắng là vân tay, màu đen là nền.
4.  Ảnh kết quả: Chỉ còn lại vân tay, nền đã bị xóa sạch.
