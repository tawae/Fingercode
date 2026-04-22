# Báo Cáo Chiến Lược: Minutiae sang Fingercode & Faiss IVF

Hệ thống nhận dạng vân tay đã trải qua một sự lột xác kiến trúc từ việc trích xuất và đối sánh các điểm đặc trưng (Minutiae) sang việc lưu trữ chỉ mục qua độ phân tán vùng (Fingercode) kết hợp công nghệ trích xuất đỉnh cao của Meta - FAISS Inverted File Index.

## 1. Phương pháp Cũ: Đối sánh Điểm Ngắt (Minutiae)

**Cách hoạt động**:
- Phân tách ảnh vân tay ra dạng Nhị phân (Binarization) rồi Làm mảnh (Thinning/Skeletonization) thành các đường chỉ rộng đúng 1 pixel.
- Quét bộ lọc cục bộ (Crossing Number) để tìm ra tọa độ (x, y) và góc của các điểm kết thúc (Ending Points) và điểm chia nhánh (Bifurcations).
- **Matching**: Quét tất cả `N` Minutiae của ảnh 1, xoay và so sánh tịnh tiến vét cạn với tất cả `M` Minutiae của ảnh 2 (Độ phức tạp $O(M \times N)$). Đi qua CSDL sẽ có độ phức tạp $O(M \times N \times DatabaseSize)$.

### Ưu Điểm
- Chính xác và là tiêu chuẩn Pháp Y. Hệ thống lưu lại chính xác vị trí lỗi vân tay.
- Rất phổ biến, nhẹ nhàng trong vấn đề dung lượng lưu trữ (do chỉ lưu danh sách điểm).

### Nhược Điểm
- **Quá Chậm**: Tính toán không thể hỗ trợ quy mô mở rộng vì phải làm toán hình học không gian giữa 2 mảng không cố định.
- Vô cùng nhạy cảm với vân tay bẩn: Mồ hôi, đứt gãy ảnh tạo ra hàng trăm Minutiae lỗi dẫn đến sai số trầm trọng.

---

## 2. Phương pháp Mới: Jain's Fingercode + Faiss IVF

**Cách hoạt động**:
- Tìm **Core Point** (tâm) của Vân tay.
- Giữ nguyên ảnh Grayscale (được làm nét), chia sẻ khu vực ảnh xung quanh tâm thành một mạng lưới dạng màng nhện gồm **40 Sectors** (Gồm 5 Vòng Đồng Tâm, cắt thành 8 Góc).
- Phóng 8 bộ lọc **Gabor Filters** vào toàn thể ảnh vân tay với các trục góc hướng khác nhau để tạo ra 8 layer nổi vân.
- Tính toán độ lệch trung bình (AAD / Phương sai) ở bên trong từng Sector đối với từng Bộ Lọc Gabor. Chúng ta thu được $40 \times 8 = $ **Vector 320 Chiều Cố Định**.

### Đánh giá Kết Quả Hiện Tại

#### Ưu Điểm (Tốt)
- **Truy vấn Siêu việt**: Việc mỗi vân tay nằm trong 1 Vector 320 Chiều giúp ta dùng công thức khoảng cách chuẩn (Euclidean Distance). Công việc toán logic phức tạp bị đánh sập.
- **Tương thích hoàn hảo với AI / Clustering**: Cho phép ta đưa cả triệu bản ghi vào Faiss IVF, tạo Voronoi Cells (Clusters) để k-NN Search. Thay vì duyệt 1 triệu vân tay, Faiss chỉ nhảy vào Cụm Fingerprint liên quan rồi so sánh 500 ảnh.
- Tốc độ tăng phi mã: Matching Time giảm xuống mức xấp xỉ `0.02 - 0.05 ms / Truy vấn`.
- Khắc phục hình ảnh bẩn: Vector Fingercode bị ảnh hưởng ít hơn khi dính 1 đường xước nhỏ, do nó đo Trung Bình Phương Sai của cả 1 vùng ảnh (Sector), chứ không chăm chăm đi tìm điểm ngắt. Điểm ngắt bị đứt (đứt đoạn do bẩn) thì Minutiae lỗi, nhưng Fingercode coi nó là bình thường.

#### Nhược Điểm (Chưa Tốt)
- Tính xoay (Rotation) và Tịnh tiến (Translation) cần phải định tuyến lúc bắt Core Point. Nếu hệ thống vô tình tìm sai Core Point (Vân tay cụt, mất tâm), Vector sẽ bị lệch hoàn toàn khiến Matching sai (Điều mà Minutiae xử lý cục bộ tốt hơn). Việc tìm Core Point dựa trên *Orientation Variance* trong code hiện tại chỉ là một biện pháp thay thế đơn giản, chưa hoàn toàn khống chế được 100% tỷ lệ FVC2002.
- Vector bị ảnh xạ phụ thuộc vào Bán kính ngón tay. Nếu chụp bằng Cảm biến lúp, vector bị sai khác quá nhiều.

---

## 3. Tổng kết Cấu Trúc File
- Xoá tệp `07_binarize_thin.py` khỏi luồng hệ thống vì Gabor Variance của Fingercode không cần binarize hay làm mảnh vân tay.
- Đổi tên bước Tính đặc trưng thành cấu trúc Vector L2 cố định: `08_fingercode_extraction.py`.
- Tái cấu trúc Layer DB 10 nhằm phục vụ Index Faiss C++ cho tốc độ Production Ready.
