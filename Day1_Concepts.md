# Day 1: Khởi động - "Vân Tay Dưới Góc Nhìn Máy Tính"

## 1. Mục Tiêu của Ngày 1
Hôm nay chúng ta sẽ ***không*** viết bất kỳ dòng thuật toán phức tạp nào. Mục tiêu duy nhất là:
1.  Hiểu **Input** của hệ thống (Ảnh vân tay trông như thế nào dưới dạng số?).
2.  Thiết lập môi trường Python để sẵn sàng code.
3.  Phân biệt rõ **Ridge (Vân)** và **Valley (Rãnh)** - hai khái niệm cốt lõi.

## 2. Chuẩn Bị Môi Trường Python
Bạn cần cài các thư viện sau để bắt đầu "nhìn" ảnh vân tay.
Chạy lệnh sau trong terminal (CMD/PowerShell) tại thư mục dự án:

```bash
pip install numpy matplotlib opencv-python
```

## 3. Lý Thuyết Cốt Lỗi: Ridge vs Valley
Hãy tưởng tượng ngón tay bạn là một thửa ruộng bậc thang.
*   **Ridge (Đường Vân - Màu Đen):** Là phần da *nhô lên*, chạm vào bề mặt máy quét. Nó in hình lên giấy/máy quét.
    *   Trong ảnh xám (Grayscale), giá trị của nó thấp (gần 0 - màu đen).
*   **Valley (Rãnh - Màu Trắng):** Là phần da *lõm xuống*, nằm giữa các đường vân.
    *   Trong ảnh xám, giá trị của nó cao (gần 255 - màu trắng).

> **Máy tính chỉ thấy một ma trận số.** Nhiệm vụ của chúng ta là tìm các đường màu đen (Ridge) và các điểm đặc biệt trên đó (Minutiae).

## 4. Bài Tập Thực Hành (Python Script)
Tôi đã tạo file `01_visualize_fingerprint.py`.
Hãy chạy nó bằng lệnh:
```bash
python 01_visualize_fingerprint.py
```

### Điều gì sẽ xảy ra?
Script sẽ hiển thị 1 ảnh vân tay từ dataset FVC2002 và biểu đồ Histogram (phân bố độ sáng).
*   Bạn sẽ thấy biểu đồ có 2 đỉnh (bimodal):
    *   Một đỉnh ở vùng tối (Ridge).
    *   Một đỉnh ở vùng sáng (Valley/Nền trắng).
*   Đây là bước đầu tiên để hiểu tại sao chúng ta cần "Nhị phân hóa" (biến ảnh thành 0 và 1) ở các bước sau.

## 5. Mini App trên điện thoại? (Khả thi!)
Bạn muốn có app trên điện thoại để demo?
*   **Phương án:** Chúng ta sẽ viết Backend bằng Python (FastAPI) xử lý vân tay, và Frontend là một Web App đơn giản (Streamlit) chạy trên Laptop.
*   **Cách demo:** Bạn bật Web App trên Laptop -> Dùng điện thoại truy cập vào địa chỉ IP của Laptop qua Wifi -> Dùng Camera điện thoại chụp vân tay -> Gửi về Laptop xử lý -> Kết quả hiện ngay trên điện thoại.
*   Cách này **nhanh và dễ** hơn nhiều so với viết Android/iOS App native (vốn rất khó tích hợp thư viện xử lý ảnh xịn sò).

---
**Ngày mai (Day 2):** Chúng ta sẽ học cách làm cho ảnh vân tay "nét" hơn (Preprocessing).
